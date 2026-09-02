"""Tests for Fill idempotency and exactly-once execution (TASK C-2)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.core.position_tracker import PositionTracker
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
async def test_duplicate_fill_id_ignored():
    """
    If the same trade fill (identical fill.id) is received multiple times
    (e.g. after WS reconnection), only the first fill is processed.
    """
    cfg = PairConfig(symbol="SOL/USDT:USDT", leverage=5)
    mock_gw = MagicMock()
    tracker = PositionTracker(cfg, mock_gw)

    fill_1 = FillRecord(
        id="trade_dup_101",
        order_id="ord_1",
        client_order_id="q_buy_0",
        symbol="SOL/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=100.0,
        amount=1.0,
        quote_amount=100.0,
        timestamp=1000.0,
        realized_pnl=0.0,
    )

    # First delivery
    res1 = await tracker.on_fill(fill_1)
    assert res1 is True
    assert tracker.long_pos.amount == 1.0

    # Duplicate delivery
    res2 = await tracker.on_fill(fill_1)
    assert res2 is False
    assert tracker.long_pos.amount == 1.0  # Amount must NOT double to 2.0


@pytest.mark.asyncio
async def test_duplicate_exit_fill_not_double_counted_at_worker_level():
    """
    Test that duplicate exit fill events do not cause duplicate executor routing,
    duplicate PnL DB records, or double-counted session_realized_pnl.
    """
    cfg = PairConfig(symbol="SOL/USDT:USDT", leverage=5)
    mock_gw = MagicMock()
    mock_gw.get_market_precision.return_value = (2, 3, 0.01, 0.001)
    mock_gw.get_market_limits.return_value = (0.0, 5.0)
    mock_gw.create_exit_order = AsyncMock(return_value={"id": "exit_1", "status": "closed"})
    mock_gw.cancel_order = AsyncMock()

    worker = PMMWorker(cfg, mock_gw)

    with patch("app.core.executor.db.record_pnl", new_callable=AsyncMock) as mock_record_pnl:
        # Open 2.0 LONG
        entry_fill = FillRecord(
            id="fill_entry_1",
            order_id="ord_entry_1",
            client_order_id="q_buy_0",
            symbol="SOL/USDT:USDT",
            side=OrderSide.BUY,
            position_side=PositionSide.LONG,
            price=100.0,
            amount=2.0,
            quote_amount=200.0,
            timestamp=1000.0,
            realized_pnl=0.0,
        )
        await worker.on_fill(entry_fill)
        assert worker.tracker.long_pos.amount == 2.0
        assert worker.executor_long.state.remaining_qty == 2.0

        # Emit exit fill id="fill_tp_dup" amount 1.0 realized_pnl=1.0
        exit_fill = FillRecord(
            id="fill_tp_dup",
            order_id="ord_tp_dup",
            client_order_id="tp_long_0",
            symbol="SOL/USDT:USDT",
            side=OrderSide.SELL,
            position_side=PositionSide.LONG,
            price=101.0,
            amount=1.0,
            quote_amount=101.0,
            timestamp=1010.0,
            realized_pnl=1.0,
            fee=0.0,
        )
        await worker.on_fill(exit_fill)
        assert worker.session_realized_pnl == pytest.approx(1.0, abs=1e-5)
        assert worker.executor_long.state.remaining_qty == pytest.approx(1.0, abs=1e-5)
        assert mock_record_pnl.call_count == 1

        # Emit duplicate exit fill
        await worker.on_fill(exit_fill)
        assert worker.session_realized_pnl == pytest.approx(1.0, abs=1e-5)
        assert worker.executor_long.state.remaining_qty == pytest.approx(1.0, abs=1e-5)
        assert mock_record_pnl.call_count == 1


@pytest.mark.asyncio
async def test_out_of_order_partial_fills_converge():
    """
    Two partial TP exits (t1 and t2) processed:
    Initial position = 2.0.
    Exit fill A: -0.8 @ 101.0
    Exit fill B: -1.2 @ 102.0
    Total remaining should converge to 0.0 and position closed.
    """
    cfg = PairConfig(symbol="SOL/USDT:USDT", leverage=5)
    mock_gw = MagicMock()
    mock_gw.get_market_precision.return_value = (2, 3, 0.01, 0.001)
    mock_gw.get_market_limits.return_value = (0.0, 5.0)
    mock_gw.create_exit_order = AsyncMock(return_value={"id": "exit_1", "status": "closed"})
    mock_gw.cancel_order = AsyncMock()

    worker = PMMWorker(cfg, mock_gw)

    # Open 2.0 LONG
    init_fill = FillRecord(
        id="fill_entry",
        order_id="ord_entry",
        client_order_id="q_buy_0",
        symbol="SOL/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=100.0,
        amount=2.0,
        quote_amount=200.0,
        timestamp=1000.0,
        realized_pnl=0.0,
    )
    await worker.on_fill(init_fill)
    assert worker.tracker.long_pos.amount == 2.0
    assert worker.executor_long.state.remaining_qty == 2.0

    # Partial exit 1: -0.8
    tp_1 = FillRecord(
        id="fill_tp_1",
        order_id="ord_tp_1",
        client_order_id="tp_long_0",
        symbol="SOL/USDT:USDT",
        side=OrderSide.SELL,
        position_side=PositionSide.LONG,
        price=101.0,
        amount=0.8,
        quote_amount=80.8,
        timestamp=1010.0,
        realized_pnl=0.8,
    )
    await worker.on_fill(tp_1)
    assert worker.tracker.long_pos.amount == pytest.approx(1.2, abs=1e-5)
    assert worker.executor_long.state.remaining_qty == pytest.approx(1.2, abs=1e-5)

    # Partial exit 2: -1.2 (flat)
    tp_2 = FillRecord(
        id="fill_tp_2",
        order_id="ord_tp_2",
        client_order_id="tp_long_1",
        symbol="SOL/USDT:USDT",
        side=OrderSide.SELL,
        position_side=PositionSide.LONG,
        price=102.0,
        amount=1.2,
        quote_amount=122.4,
        timestamp=1020.0,
        realized_pnl=2.4,
    )
    await worker.on_fill(tp_2)
    assert worker.tracker.long_pos.amount == 0.0
    assert worker.executor_long.state.remaining_qty == 0.0
    assert worker.executor_long.state.active is False
