"""Unit and Integration Tests for Startup Barrier Reconciliation & Overflow Recovery (Hedge Mode)."""
import time
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.core.executor import TripleBarrierExecutor
from app.core.position_tracker import PositionTracker
from app.core.quoter import PMMQuoter
from app.core.worker import PMMWorker
from app.models.config import PairConfig
from app.models.state import OrderPurpose, OrderSide, OrderType, PositionSide, SidePositionState
from app.persistence.db import db


@pytest.fixture
async def setup_test_env():
    config = PairConfig(
        symbol="SOL/USDT:USDT",
        tp_levels=[[0.01, 0.5], [0.02, 0.5]],
        stop_loss=0.02,
        time_limit=3600,
    )

    mock_gateway = MagicMock()
    mock_gateway.get_market_precision.return_value = (2, 3, 0.01, 0.001)
    mock_gateway.create_exit_order = AsyncMock()
    mock_gateway.cancel_order = AsyncMock(return_value=True)

    async def fake_create_exit(**kwargs):
        return {
            "id": f"ord_{int(time.time()*1000)}_{kwargs.get('order_type')}",
            "amount": kwargs.get("amount"),
            "price": kwargs.get("price"),
            "stopPrice": kwargs.get("stop_price"),
            "params": {"positionSide": kwargs.get("position_side").value},
        }

    mock_gateway.create_exit_order.side_effect = fake_create_exit

    tracker = PositionTracker(config, mock_gateway)
    quoter = PMMQuoter(config, 2, 3, 0.01, 0.001)

    executor_long = TripleBarrierExecutor(
        config=config,
        position_side=PositionSide.LONG,
        gateway=mock_gateway,
        position_tracker=tracker,
        quoter=quoter,
    )

    executor_short = TripleBarrierExecutor(
        config=config,
        position_side=PositionSide.SHORT,
        gateway=mock_gateway,
        position_tracker=tracker,
        quoter=quoter,
    )

    yield config, mock_gateway, tracker, quoter, executor_long, executor_short


@pytest.mark.asyncio
async def test_startup_reconcile_arms_long_barrier(setup_test_env):
    """Test that pre-existing LONG position auto-arms barrier with TP and SL on startup."""
    config, mock_gateway, tracker, quoter, executor_long, _ = setup_test_env

    # Simulate existing LONG position on exchange
    tracker.long_pos.amount = 10.0
    tracker.long_pos.entry_price = 100.0
    tracker.long_pos.current_price = 100.2

    assert executor_long.state.active is False

    # Execute reconcile_barrier with normal mark price (100.2)
    await executor_long.reconcile_barrier(current_mark_price=100.2)

    assert executor_long.state.active is True
    assert executor_long.state.remaining_qty == 10.0
    assert executor_long.state.sl_qty == 10.0
    assert len(executor_long.state.tp_orders) == 2

    # Verify calls to create_exit_order (2 TP limit maker, 0 server STOP_MARKET)
    assert mock_gateway.create_exit_order.call_count == 2
    assert executor_long.state.sl_price == 98.0
    calls = mock_gateway.create_exit_order.call_args_list
    for c in calls:
        assert c.kwargs["order_type"] == OrderType.LIMIT_MAKER
        assert c.kwargs["purpose"] == OrderPurpose.TAKE_PROFIT


@pytest.mark.asyncio
async def test_startup_reconcile_arms_short_barrier(setup_test_env):
    """Test that pre-existing SHORT position auto-arms barrier with TP and Virtual SL on startup."""
    config, mock_gateway, tracker, quoter, _, executor_short = setup_test_env

    # Simulate existing SHORT position on exchange
    tracker.short_pos.amount = 10.0
    tracker.short_pos.entry_price = 100.0
    tracker.short_pos.current_price = 99.8

    assert executor_short.state.active is False

    # Execute reconcile_barrier with normal mark price (99.8)
    await executor_short.reconcile_barrier(current_mark_price=99.8)

    assert executor_short.state.active is True
    assert executor_short.state.remaining_qty == 10.0
    assert executor_short.state.sl_qty == 10.0
    assert executor_short.state.sl_price == 102.0
    assert len(executor_short.state.tp_orders) == 2

    # Verify calls to create_exit_order (2 TP limit maker, 0 server STOP_MARKET)
    assert mock_gateway.create_exit_order.call_count == 2
    calls = mock_gateway.create_exit_order.call_args_list
    for c in calls:
        assert c.kwargs["order_type"] == OrderType.LIMIT_MAKER
        assert c.kwargs["purpose"] == OrderPurpose.TAKE_PROFIT


@pytest.mark.asyncio
async def test_startup_reconcile_overflow_short_sl_breached(setup_test_env):
    """
    Test overflow condition: SHORT position entered at 0.6553 with 2% SL (0.6684).
    Current mark price is 0.7071 (already breached SL).
    Reconcile must execute emergency MARKET exit to stop further loss.
    """
    config, mock_gateway, tracker, quoter, _, executor_short = setup_test_env

    # Set up SHORT position similar to the ASTER scenario
    tracker.short_pos.amount = 113.0
    tracker.short_pos.entry_price = 0.6553
    tracker.short_pos.current_price = 0.7071

    # Execute reconcile_barrier with breached mark price
    await executor_short.reconcile_barrier(current_mark_price=0.7071)

    # Must execute MARKET exit with STOP_LOSS purpose
    calls = mock_gateway.create_exit_order.call_args_list
    assert len(calls) == 1
    exit_call = calls[0].kwargs

    assert exit_call["position_side"] == PositionSide.SHORT
    assert exit_call["side"] == OrderSide.BUY
    assert exit_call["order_type"] == OrderType.MARKET
    assert exit_call["amount"] == 113.0
    assert exit_call["purpose"] == OrderPurpose.STOP_LOSS


@pytest.mark.asyncio
async def test_startup_reconcile_overflow_long_sl_breached(setup_test_env):
    """
    Test overflow condition: LONG position entered at 100.0 with 2% SL (98.0).
    Current mark price is 97.0 (already breached SL).
    Reconcile must execute emergency MARKET exit.
    """
    config, mock_gateway, tracker, quoter, executor_long, _ = setup_test_env

    tracker.long_pos.amount = 10.0
    tracker.long_pos.entry_price = 100.0
    tracker.long_pos.current_price = 97.0

    await executor_long.reconcile_barrier(current_mark_price=97.0)

    calls = mock_gateway.create_exit_order.call_args_list
    assert len(calls) == 1
    exit_call = calls[0].kwargs

    assert exit_call["position_side"] == PositionSide.LONG
    assert exit_call["side"] == OrderSide.SELL
    assert exit_call["order_type"] == OrderType.MARKET
    assert exit_call["amount"] == 10.0
    assert exit_call["purpose"] == OrderPurpose.STOP_LOSS


@pytest.mark.asyncio
async def test_startup_reconcile_overflow_tp_breached(setup_test_env):
    """
    Test overflow condition: LONG position entered at 0.6644 with 1% TP (0.6710).
    Current mark price is 0.7071 (already well beyond TP in profit).
    Reconcile must execute immediate MARKET profit exit.
    """
    config, mock_gateway, tracker, quoter, executor_long, _ = setup_test_env

    tracker.long_pos.amount = 36.0
    tracker.long_pos.entry_price = 0.6644
    tracker.long_pos.current_price = 0.7071

    await executor_long.reconcile_barrier(current_mark_price=0.7071)

    calls = mock_gateway.create_exit_order.call_args_list
    assert len(calls) == 1
    exit_call = calls[0].kwargs

    assert exit_call["position_side"] == PositionSide.LONG
    assert exit_call["side"] == OrderSide.SELL
    assert exit_call["order_type"] == OrderType.MARKET
    assert exit_call["amount"] == 36.0
    assert exit_call["purpose"] == OrderPurpose.TAKE_PROFIT


@pytest.mark.asyncio
async def test_worker_reconcile_barriers_integration():
    """Test full PMMWorker reconcile_barriers method."""
    await db.connect()
    try:
        config = PairConfig(
            symbol="SOL/USDT:USDT",
            tp_levels=[[0.01, 1.0]],
            stop_loss=0.02,
        )
        mock_gw = MagicMock()
        mock_gw.get_market_precision.return_value = (2, 3, 0.01, 0.001)
        mock_gw.create_exit_order = AsyncMock(return_value={"id": "mock_exit_1"})
        mock_gw.cancel_order = AsyncMock(return_value=True)

        real_long = SidePositionState(
            symbol="SOL/USDT:USDT",
            position_side=PositionSide.LONG,
            amount=5.0,
            entry_price=100.0,
            current_price=100.2,
        )
        real_short = SidePositionState(
            symbol="SOL/USDT:USDT",
            position_side=PositionSide.SHORT,
            amount=0.0,
            entry_price=0.0,
            current_price=100.2,
        )
        mock_gw.fetch_positions_hedge = AsyncMock(return_value=(real_long, real_short))

        worker = PMMWorker(config, mock_gw)
        await worker.tracker.reconcile_with_exchange()
        worker.market_state.update_ticker(100.1, 100.3, 100.2)

        await worker.reconcile_barriers()

        assert worker.executor_long.state.active is True
        assert worker.executor_long.state.remaining_qty == 5.0
        assert worker.executor_short.state.active is False
    finally:
        await db.close()

