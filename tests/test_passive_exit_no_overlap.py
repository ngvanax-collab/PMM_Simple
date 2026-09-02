"""Unit tests for Hardening Fixes (Phase 6): M-1, M-3, M-8, L-1, L-2."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.core.executor import TripleBarrierExecutor
from app.core.fr_execution.position_tracker import FRPositionTracker
from app.core.gateway import ExchangeGateway
from app.core.position_tracker import PositionTracker
from app.core.quoter import PMMQuoter
from app.models.config import PairConfig
from app.models.state import FillRecord, OrderPurpose, OrderRecord, OrderSide, OrderStatus, OrderType, PositionSide
from app.config import AppSettings, _get_fernet


def _create_pair_config(symbol: str = "BTC/USDT:USDT") -> PairConfig:
    return PairConfig(
        symbol=symbol,
        exchange="binance",
        enabled=True,
        leverage=5,
        margin_mode="isolated",
        order_amount_usdt=30.0,
        bid_spread=0.003,
        ask_spread=0.003,
        order_levels=3,
        take_profit=0.006,
        stop_loss=0.015,
        time_limit=120,
        passive_exit_timeout_sec=60,
        passive_exit_spread_pct=0.0006,
    )


@pytest.mark.asyncio
async def test_passive_exit_cancelled_before_tp_replenishment():
    """
    TASK M-1: Prevent Passive Exit and TP overlap:
    When a partial exit fill occurs while a passive exit order is open,
    the passive exit order must be explicitly cancelled and deactivated
    BEFORE re-placing TP orders to avoid over-exposure.
    """
    cfg = _create_pair_config()
    gw = MagicMock(spec=ExchangeGateway)
    gw.get_market_precision.return_value = (2, 3, 0.01, 0.001)
    gw.get_market_limits.return_value = (0.001, 5.0)
    gw.cancel_order = AsyncMock(return_value=True)
    gw.create_exit_order = AsyncMock(return_value={"id": "new_tp_1", "status": "open"})

    quoter = PMMQuoter(cfg, price_precision=2, amount_precision=3, tick_size=0.01, step_size=0.001)
    pos_tracker = PositionTracker(cfg, gw)
    executor = TripleBarrierExecutor(
        config=cfg,
        position_side=PositionSide.LONG,
        gateway=gw,
        position_tracker=pos_tracker,
        quoter=quoter,
    )

    # Setup active LONG position: 10.0 units @ 100.0
    executor.state.active = True
    executor.state.entry_price = 100.0
    executor.state.total_qty = 10.0
    executor.state.remaining_qty = 10.0

    # Simulate an active passive exit order
    executor.state.passive_exit_active = True
    executor.state.passive_exit_order_id = "pe_long_existing_999"

    # Simulate a partial exit fill of 3.0 units (7.0 units remaining)
    partial_fill = FillRecord(
        id="fill_partial_1",
        order_id="pe_long_existing_999",
        client_order_id="pe_long_123",
        symbol=cfg.symbol,
        side=OrderSide.SELL,
        position_side=PositionSide.LONG,
        price=100.5,
        amount=3.0,
        quote_amount=301.5,
        fee=0.05,
        fee_currency="USDT",
        is_maker=True,
        timestamp=time.time(),
        realized_pnl=1.5,
    )

    with patch("app.core.executor.db.record_pnl", new_callable=AsyncMock):
        await executor.on_exit_fill(partial_fill)

    # 1. Passive exit order must have been cancelled
    gw.cancel_order.assert_any_await(cfg.symbol, "pe_long_existing_999")
    assert executor.state.passive_exit_active is False
    assert executor.state.passive_exit_order_id is None

    # 2. Position remaining quantity updated
    assert pytest.approx(executor.state.remaining_qty, abs=1e-5) == 7.0

    # 3. New TP orders placed for remaining quantity (sum of TP ladder = 7.0)
    assert gw.create_exit_order.call_count >= 1
    total_tp_placed = sum(c[1]["amount"] for c in gw.create_exit_order.call_args_list if c[1]["purpose"] == OrderPurpose.TAKE_PROFIT)
    assert pytest.approx(total_tp_placed, abs=1e-3) == 7.0


@pytest.mark.asyncio
async def test_passive_exit_escalation_to_market_after_two_refreshes():
    """
    TASK M-3: Passive Exit Escalation to MARKET:
    When passive maker exit remains unfilled after > 2 refresh cycles (3 timeouts),
    it escalates to an immediate MARKET exit with OrderPurpose.TIME_LIMIT_EXIT.
    """
    cfg = _create_pair_config()
    gw = MagicMock(spec=ExchangeGateway)
    gw.get_market_precision.return_value = (2, 3, 0.01, 0.001)
    gw.get_market_limits.return_value = (0.001, 5.0)
    gw.cancel_order = AsyncMock(return_value=True)
    gw.create_exit_order = AsyncMock(return_value={"id": "exit_ord_1", "status": "open"})

    quoter = PMMQuoter(cfg, price_precision=2, amount_precision=3, tick_size=0.01, step_size=0.001)
    pos_tracker = PositionTracker(cfg, gw)
    executor = TripleBarrierExecutor(
        config=cfg,
        position_side=PositionSide.LONG,
        gateway=gw,
        position_tracker=pos_tracker,
        quoter=quoter,
    )

    # Setup active LONG position
    executor.state.active = True
    executor.state.entry_price = 100.0
    executor.state.total_qty = 5.0
    executor.state.remaining_qty = 5.0
    executor.state.entry_timestamp = 1000.0

    now = 1200.0  # past time_limit 120s
    # 1. Initial passive exit placement
    with patch("time.time", return_value=now):
        await executor._handle_passive_time_limit_exit(current_price=100.0, best_bid=99.9, best_ask=100.1)
        assert executor.state.passive_exit_active is True
        assert executor.state.passive_exit_order_id is not None
        assert executor.state.passive_exit_refresh_count == 0

    # 2. Timeout cycle 1 (at now + 70s > 60s timeout)
    now += 70.0
    with patch("time.time", return_value=now):
        await executor._handle_passive_time_limit_exit(current_price=100.0, best_bid=99.9, best_ask=100.1)
        # Should cancel previous and place new passive order (refresh count = 1)
        assert executor.state.passive_exit_refresh_count == 1
        assert executor.state.passive_exit_active is True

    # 3. Timeout cycle 2 (at now + 70s) -> refresh count becomes 2 (>= 2) -> ESCALATE TO MARKET
    now += 70.0
    with patch("time.time", return_value=now):
        await executor._handle_passive_time_limit_exit(current_price=100.0, best_bid=99.9, best_ask=100.1)

        # Verify MARKET exit was fired with purpose TIME_LIMIT_EXIT
        last_exit_call = gw.create_exit_order.call_args_list[-1][1]
        assert last_exit_call["order_type"] == OrderType.MARKET
        assert last_exit_call["purpose"] == OrderPurpose.TIME_LIMIT_EXIT
        assert executor.state.passive_exit_refresh_count == 0


def test_secret_key_validation_in_production():
    """
    TASK M-8: Secret key validation:
    - Missing SECRET_KEY in production raises RuntimeError.
    - Missing SECRET_KEY in dev mode generates fallback key safely.
    """
    # In production without secret key -> RuntimeError
    with patch("app.config.settings", AppSettings(app_env="production", secret_key="")):
        with pytest.raises(RuntimeError, match="FATAL: SECRET_KEY"):
            _get_fernet()

    # In dev mode without secret key -> success with fallback
    with patch("app.config.settings", AppSettings(app_env="development", secret_key="")):
        fernet = _get_fernet()
        assert fernet is not None


def test_record_real_funding_deduplication():
    """
    TASK L-2: Real funding accrual polling and deduplication:
    - First record of funding transaction adds amount to position and portfolio realized funding PnL.
    - Duplicate transaction ID is rejected without double counting.
    """
    tracker = FRPositionTracker()
    dual_pos = tracker.get_or_create_position("SOL/USDT:USDT")

    # 1. First transaction
    ok1 = tracker.record_funding_payment(
        funding_id="binance_tx_1001",
        symbol="SOL/USDT:USDT",
        exchange="binance",
        amount=1.25,
        timestamp=1700000000.0,
    )
    assert ok1 is True
    assert pytest.approx(tracker.realized_funding_pnl, abs=1e-5) == 1.25
    assert pytest.approx(dual_pos.total_funding_accrued, abs=1e-5) == 1.25

    # 2. Duplicate transaction
    ok2 = tracker.record_funding_payment(
        funding_id="binance_tx_1001",
        symbol="SOL/USDT:USDT",
        exchange="binance",
        amount=1.25,
        timestamp=1700000000.0,
    )
    assert ok2 is False
    assert pytest.approx(tracker.realized_funding_pnl, abs=1e-5) == 1.25
    assert pytest.approx(dual_pos.total_funding_accrued, abs=1e-5) == 1.25
