"""Unit & Integration Tests for Triple Barrier Executor (Hedge Mode & Virtual SL / Passive Exit)."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.core.executor import TripleBarrierExecutor
from app.core.position_tracker import PositionTracker
from app.core.quoter import PMMQuoter
from app.models.config import PairConfig
from app.models.state import FillRecord, OrderPurpose, OrderSide, OrderType, PositionSide
import tempfile
import os
from app.persistence.db import Database


@pytest.fixture
async def setup_executor():
    # Use isolated temp database to avoid locking conflicts with live processes
    temp_f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_path = temp_f.name
    temp_f.close()

    test_db = Database(temp_path)
    await test_db.connect()

    try:
        with patch("app.core.executor.db", test_db):
            config = PairConfig(
                symbol="SOL/USDT:USDT",
                tp_levels=[[0.01, 0.5], [0.02, 0.5]],  # 50% at 1% TP, 50% at 2% TP
                stop_loss=0.02,  # 2% SL
                time_limit=3600,
                trailing_tp_enabled=False,  # Test legacy trailing stop config independently
                trailing_stop={"activation_price": 0.015, "trailing_delta": 0.005},
                min_holding_sec=0.0,
            )

            # Mock gateway
            mock_gateway = MagicMock()
            mock_gateway.get_market_precision.return_value = (2, 3, 0.01, 0.001)
            mock_gateway.create_exit_order = AsyncMock()
            mock_gateway.cancel_order = AsyncMock(return_value=True)

            # Setup mock return for create_exit_order
            async def fake_create_exit(**kwargs):
                return {
                    "id": f"ord_{int(time.time()*1000)}_{kwargs.get('order_type')}",
                    "amount": kwargs.get("amount"),
                    "price": kwargs.get("price"),
                    "stopPrice": kwargs.get("stop_price"),
                    "order_type": kwargs.get("order_type"),
                    "purpose": kwargs.get("purpose"),
                    "params": {"positionSide": kwargs.get("position_side").value},
                }
            mock_gateway.create_exit_order.side_effect = fake_create_exit

            tracker = PositionTracker(config, mock_gateway)
            quoter = PMMQuoter(config, 2, 3, 0.01, 0.001)

            executor = TripleBarrierExecutor(
                config=config,
                position_side=PositionSide.LONG,
                gateway=mock_gateway,
                position_tracker=tracker,
                quoter=quoter,
            )

            yield executor, tracker, mock_gateway
    finally:
        await test_db.close()
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


@pytest.mark.asyncio
async def test_sl_quantity_sync_on_partial_tp(setup_executor):
    """
    CRITICAL HEDGE MODE TEST:
    Verify that when partial TP fills, the Virtual SL quantity
    is immediately updated to match the remaining position size.
    """
    executor, tracker, mock_gateway = setup_executor

    # 1. Simulate entry fill: BUY 10.0 SOL @ $100.00 (positionSide=LONG)
    entry_fill = FillRecord(
        id="fill_entry_1",
        order_id="ord_entry_1",
        symbol="SOL/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=100.0,
        amount=10.0,
        quote_amount=1000.0,
        fee=0.2,
        is_maker=True,
        timestamp=time.time(),
    )
    await tracker.on_fill(entry_fill)
    await executor.on_entry_fill(entry_fill)

    # Check that initial Virtual SL was set with quantity = 10.0
    assert executor.state.active is True
    assert executor.state.remaining_qty == 10.0
    assert executor.state.sl_qty == 10.0
    assert len(executor.state.tp_orders) == 2  # 2 TP tiers

    # 2. Simulate Partial TP #1 Fill: SELL 5.0 SOL @ $101.00
    tp_fill = FillRecord(
        id="fill_tp_1",
        order_id=executor.state.tp_orders[0]["order_id"],
        symbol="SOL/USDT:USDT",
        side=OrderSide.SELL,
        position_side=PositionSide.LONG,
        price=101.0,
        amount=5.0,
        quote_amount=505.0,
        fee=0.1,
        is_maker=True,
        timestamp=time.time(),
        realized_pnl=5.0,
    )
    await tracker.on_fill(tp_fill)
    await executor.on_exit_fill(tp_fill)

    # 3. VERIFY SL INVARIANT & TP RE-PLACEMENT:
    # Remaining qty is 5.0, Virtual SL quantity MUST be 5.0, and TP orders re-placed for 5.0!
    assert tracker.long_pos.amount == 5.0
    assert executor.state.remaining_qty == 5.0
    assert executor.state.sl_qty == 5.0, f"Expected SL qty=5.0 after partial TP, but got {executor.state.sl_qty}"
    assert executor.state.active is True
    assert len(executor.state.tp_orders) == 2
    assert sum(tp["qty"] for tp in executor.state.tp_orders) == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_trailing_stop_activation_and_pullback(setup_executor):
    """Verify Trailing Stop activates at threshold and triggers MARKET exit on pullback."""
    executor, tracker, mock_gateway = setup_executor

    # 1. Entry fill: BUY 10.0 SOL @ 100.0
    entry_fill = FillRecord(
        id="fill_e1",
        order_id="ord_e1",
        symbol="SOL/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=100.0,
        amount=10.0,
        quote_amount=1000.0,
        fee=0.2,
        is_maker=True,
        timestamp=time.time() - 10.0,
    )
    await tracker.on_fill(entry_fill)
    await executor.on_entry_fill(entry_fill)
    executor.state.entry_timestamp = time.time() - 10.0

    # 2. Price rises to 101.0 (activation is 1.5% -> 101.50). Not activated yet.
    await executor.check_runtime_barriers(101.0)
    assert executor.state.trailing_active is False

    # 3. Price reaches 102.0 (>= 101.50 -> activated!). High watermark = 102.0
    await executor.check_runtime_barriers(102.0)
    assert executor.state.trailing_active is True
    assert executor.state.trailing_high_watermark == 102.0

    # 4. Price rises further to 103.0 -> High watermark = 103.0
    await executor.check_runtime_barriers(103.0)
    assert executor.state.trailing_high_watermark == 103.0

    # 5. Price pulls back: delta is 0.5% (0.005) -> Trigger price = 103.0 * (1 - 0.005) = 102.485
    # Current price drops to 102.40 (<= 102.485 -> Trigger trailing stop market exit!)
    mock_gateway.create_exit_order.reset_mock()
    await executor.check_runtime_barriers(102.40)

    # Verify market exit order was dispatched
    assert mock_gateway.create_exit_order.called
    call_args = mock_gateway.create_exit_order.call_args.kwargs
    assert call_args["order_type"] == OrderType.MARKET
    assert call_args["side"] == OrderSide.SELL
    assert call_args["position_side"] == PositionSide.LONG
    assert call_args["purpose"] == OrderPurpose.TRAILING_STOP
    assert executor.state.active is False


@pytest.mark.asyncio
async def test_time_limit_expiration(setup_executor):
    """Verify position exits via Post-Only Passive Maker order when time limit expires."""
    executor, tracker, mock_gateway = setup_executor
    executor.config.time_limit = 10  # 10 seconds time limit

    entry_fill = FillRecord(
        id="fill_e2",
        order_id="ord_e2",
        symbol="SOL/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=100.0,
        amount=5.0,
        quote_amount=500.0,
        fee=0.1,
        is_maker=True,
        timestamp=time.time(),
    )
    await tracker.on_fill(entry_fill)
    await executor.on_entry_fill(entry_fill)

    # Force timestamp to 15s ago
    executor.state.entry_timestamp = time.time() - 15.0

    mock_gateway.create_exit_order.reset_mock()
    await executor.check_runtime_barriers(100.0)

    assert mock_gateway.create_exit_order.called
    call_args = mock_gateway.create_exit_order.call_args.kwargs
    assert call_args["order_type"] == OrderType.LIMIT_MAKER
    assert call_args["purpose"] == OrderPurpose.PASSIVE_TIME_LIMIT_EXIT
    assert executor.state.passive_exit_active is True


@pytest.mark.asyncio
async def test_server_sl_fill_triggers_cooldown(setup_executor):
    """Verify that when an SL fill occurs, progressive cooldown is activated."""
    executor, tracker, mock_gateway = setup_executor

    entry_fill = FillRecord(
        id="fill_e3",
        order_id="ord_e3",
        symbol="SOL/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=100.0,
        amount=10.0,
        quote_amount=1000.0,
        fee=0.2,
        is_maker=True,
        timestamp=time.time(),
    )
    await tracker.on_fill(entry_fill)
    await executor.on_entry_fill(entry_fill)

    assert tracker.is_in_cooldown(PositionSide.LONG) is False

    # Simulate SL execution
    sl_fill = FillRecord(
        id="fill_sl_1",
        order_id="ord_sl_1",
        client_order_id="sl_long_123",
        symbol="SOL/USDT:USDT",
        side=OrderSide.SELL,
        position_side=PositionSide.LONG,
        price=98.0,
        amount=10.0,
        quote_amount=980.0,
        fee=0.4,
        is_maker=False,
        timestamp=time.time(),
        realized_pnl=-20.0,
    )
    await tracker.on_fill(sl_fill)
    await executor.on_exit_fill(sl_fill)

    # Position is flat and cooldown is activated
    assert tracker.long_pos.amount == 0.0
    assert executor.state.active is False
    assert tracker.is_in_cooldown(PositionSide.LONG) is True
