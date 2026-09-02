"""Tests for Pyramid single-count state machine, watchdog recovery, and side isolation (TASK C-1, H-2)."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.core.executor import TripleBarrierExecutor
from app.core.quoter import PMMQuoter
from app.core.worker import PMMWorker
from app.models.config import PairConfig
from app.models.state import (
    FillRecord,
    OrderSide,
    PositionSide,
)


@pytest.fixture(autouse=True)
def mock_db_ops():
    with patch("app.core.worker.db.save_fill", new_callable=AsyncMock), \
         patch("app.core.worker.db.record_pnl", new_callable=AsyncMock), \
         patch("app.core.worker.db.save_order", new_callable=AsyncMock), \
         patch("app.core.executor.db.save_fill", new_callable=AsyncMock), \
         patch("app.core.executor.db.record_pnl", new_callable=AsyncMock):
        yield


@pytest.mark.asyncio
async def test_pyramid_fill_counted_exactly_once():
    """
    Verify pyramid order does NOT mutate position before real fill arrives:
    1. Initial fill 1.0 @ 100.0 -> tracker.amount = 1.0
    2. Price surges -> pyramid triggers and dispatches order -> tracker.amount MUST STILL BE 1.0 (no premature mutation).
    3. Real fill event (cid=q_pyr_long_...) arrives via worker.on_fill -> tracker.long_pos.amount == 1.5,
       executor.state.sl_qty == 1.5, sl_price >= 100.0 * 1.0035.
    """
    cfg = PairConfig(
        symbol="SOL/USDT:USDT",
        leverage=5,
        order_amount_usdt=100.0,
        allocated_margin_usdt=21.0,
        take_profit=0.010,
        trailing_tp_enabled=True,
        trailing_tp_activation_pct=0.008,
        trailing_tp_callback_pct=0.003,
        favorable_pyramiding_enabled=True,
        pyramiding_size_pct=0.50,
        pyramiding_trigger_natr_mult=0.65,
        min_holding_sec=0.0,
    )
    mock_gw = MagicMock()
    mock_gw.get_market_precision.return_value = (2, 3, 0.01, 0.001)
    mock_gw.get_market_limits.return_value = (0.0, 5.0)
    mock_gw.create_entry_market_order = AsyncMock(return_value={"id": "pyr_order_1", "status": "open"})
    mock_gw.create_exit_order = AsyncMock(return_value={"id": "exit_ord_1", "status": "closed"})
    mock_gw.cancel_order = AsyncMock()

    worker = PMMWorker(cfg, mock_gw)
    entry_p = 100.0
    t0 = 1000000.0

    # 1. Fill Initial Level 0 LONG position (1.0 SOL @ $100.0)
    fill_0 = FillRecord(
        id="fill_init",
        order_id="ord_init",
        symbol="SOL/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=entry_p,
        amount=1.0,
        quote_amount=100.0,
        timestamp=t0,
        realized_pnl=0.0,
    )
    with patch("time.time", return_value=t0):
        await worker.on_fill(fill_0)

    assert worker.tracker.long_pos.amount == 1.0
    assert worker.executor_long.state.pyramid_filled_count == 0
    assert worker.executor_long.state.pending_pyramid_client_id is None

    # 2. Price surges up to 101.20 within 60s
    t_tick = t0 + 30.0
    price_now = 101.20
    worker.market_state.price_history_60s.append((t0, 100.0))
    worker.market_state.price_history_60s.append((t_tick, price_now))
    worker.market_state.current_natr_15m = 0.012

    with patch("time.time", return_value=t_tick):
        await worker.on_ticker_update(bid=price_now - 0.01, ask=price_now + 0.01, mark=price_now)

    # Order dispatched, but position amount MUST NOT have changed yet!
    mock_gw.create_entry_market_order.assert_awaited_once()
    assert worker.executor_long.state.pending_pyramid_client_id is not None
    assert worker.tracker.long_pos.amount == 1.0  # Guardrail 4: not mutated prematurely
    assert worker.executor_long.state.remaining_qty == 1.0
    assert worker.executor_long.state.pyramid_filled_count == 0

    # 3. Real private WS fill event arrives
    pyr_cid = worker.executor_long.state.pending_pyramid_client_id
    fill_pyr = FillRecord(
        id="fill_pyr_1",
        order_id="pyr_order_1",
        client_order_id=pyr_cid,
        symbol="SOL/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=101.20,
        amount=0.5,
        quote_amount=50.60,
        timestamp=t_tick + 0.2,
        realized_pnl=0.0,
    )
    with patch("time.time", return_value=t_tick + 0.2):
        await worker.on_fill(fill_pyr)

    # Now state should be properly updated exactly once
    assert worker.tracker.long_pos.amount == 1.5
    assert worker.executor_long.state.remaining_qty == 1.5
    assert worker.executor_long.state.sl_qty == 1.5
    assert worker.executor_long.state.pyramid_filled_count == 1
    assert worker.executor_long.state.pending_pyramid_client_id is None
    assert worker.executor_long.state.is_guaranteed_sl_locked is True
    assert worker.executor_long.state.sl_price >= (entry_p * 1.0035)


@pytest.mark.asyncio
async def test_pyramid_watchdog_recovers_missing_fill():
    """
    If pyramid order is sent but no fill event arrives within 10s:
    - Watchdog triggers reconcile_with_exchange
    - Pending pyramid client ID is cleared
    """
    cfg = PairConfig(
        symbol="SOL/USDT:USDT",
        leverage=5,
        order_amount_usdt=100.0,
        allocated_margin_usdt=21.0,
        take_profit=0.010,
        trailing_tp_enabled=True,
        trailing_tp_activation_pct=0.008,
        trailing_tp_callback_pct=0.003,
        favorable_pyramiding_enabled=True,
        pyramiding_size_pct=0.50,
        pyramiding_trigger_natr_mult=0.65,
        min_holding_sec=0.0,
    )
    mock_gw = MagicMock()
    mock_gw.get_market_precision.return_value = (2, 3, 0.01, 0.001)
    mock_gw.get_market_limits.return_value = (0.0, 5.0)
    mock_gw.create_entry_market_order = AsyncMock(return_value={"id": "pyr_order_1", "status": "open"})
    mock_gw.create_exit_order = AsyncMock(return_value={"id": "exit_ord_1", "status": "closed"})
    mock_gw.cancel_order = AsyncMock()
    mock_gw.fetch_positions = AsyncMock(return_value=[])

    worker = PMMWorker(cfg, mock_gw)
    t0 = 1000000.0

    fill_0 = FillRecord(
        id="fill_init",
        order_id="ord_init",
        symbol="SOL/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=100.0,
        amount=1.0,
        quote_amount=100.0,
        timestamp=t0,
        realized_pnl=0.0,
    )
    with patch("time.time", return_value=t0):
        await worker.on_fill(fill_0)

    # Trigger pyramid
    t_pyr = t0 + 10.0
    worker.market_state.price_history_60s.append((t0, 100.0))
    worker.market_state.price_history_60s.append((t_pyr, 101.20))
    worker.market_state.current_natr_15m = 0.012

    with patch("time.time", return_value=t_pyr):
        await worker.on_ticker_update(bid=101.19, ask=101.21, mark=101.20)

    assert worker.executor_long.state.pending_pyramid_client_id is not None

    # Advance time > 10s (e.g. +11s) without fill
    t_timeout = t_pyr + 11.0
    with patch("time.time", return_value=t_timeout), \
         patch.object(worker.tracker, "reconcile_with_exchange", new_callable=AsyncMock) as mock_recon:
        await worker.executor_long.check_runtime_barriers(101.20)
        mock_recon.assert_awaited_once()

    assert worker.executor_long.state.pending_pyramid_client_id is None
    assert worker.executor_long.state.pending_pyramid_started_at == 0.0


@pytest.mark.asyncio
async def test_trailing_callback_override_not_leaking_across_sides():
    """
    Ensure TASK H-2: Tightening callback during pyramid overrides local executor only
    and does not mutate shared PairConfig.
    """
    cfg = PairConfig(
        symbol="SOL/USDT:USDT",
        leverage=5,
        take_profit=0.010,
        trailing_tp_enabled=True,
        trailing_tp_callback_pct=0.003,
        favorable_pyramiding_enabled=True,
    )
    mock_gw = MagicMock()
    mock_gw.get_market_precision.return_value = (2, 3, 0.01, 0.001)
    mock_gw.get_market_limits.return_value = (0.0, 5.0)
    mock_gw.create_entry_market_order = AsyncMock(return_value={"id": "pyr_order_1", "status": "open"})
    mock_gw.create_exit_order = AsyncMock(return_value={"id": "exit_ord_1", "status": "closed"})
    mock_gw.cancel_order = AsyncMock()

    worker = PMMWorker(cfg, mock_gw)
    t0 = 1000000.0

    fill_long = FillRecord(
        id="fill_l",
        order_id="ord_l",
        symbol="SOL/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=100.0,
        amount=1.0,
        quote_amount=100.0,
        timestamp=t0,
        realized_pnl=0.0,
    )
    await worker.on_fill(fill_long)

    # Dispatch pyramid fill for LONG
    pyr_fill = FillRecord(
        id="fill_pyr_l",
        order_id="pyr_ord",
        client_order_id="q_pyr_long_12345",
        symbol="SOL/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=101.2,
        amount=0.5,
        quote_amount=50.6,
        timestamp=t0 + 5.0,
        realized_pnl=0.0,
    )
    await worker.on_fill(pyr_fill)

    # LONG executor should have local override
    assert worker.executor_long._trailing_cb_override == pytest.approx(0.0020, rel=1e-4)
    # SHORT executor must NOT have local override
    assert worker.executor_short._trailing_cb_override is None
    # Shared PairConfig must remain untouched
    assert cfg.trailing_tp_callback_pct == 0.003
