"""Unit Tests for Virtual Local Stop Loss & Passive Maker Time-Limit Exit."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.core.executor import TripleBarrierExecutor
from app.core.position_tracker import PositionTracker
from app.core.quoter import PMMQuoter
from app.models.config import PairConfig
from app.models.state import FillRecord, OrderPurpose, OrderSide, OrderType, PositionSide
import tempfile
import os
from unittest.mock import patch
from app.persistence.db import Database


@pytest.fixture
async def setup_virtual_sl_env():
    temp_f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_path = temp_f.name
    temp_f.close()

    test_db = Database(temp_path)
    await test_db.connect()

    with patch("app.core.executor.db", test_db):
        config = PairConfig(
            symbol="SOL/USDT:USDT",
            tp_levels=[[0.01, 1.0]],  # 1% TP
            stop_loss=0.02,  # 2% SL
            time_limit=60,  # 60s time limit
            min_holding_sec=3.0,  # 3s holding lock
            passive_exit_timeout_sec=60.0,
            passive_exit_spread_pct=0.0006,
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
                "order_type": kwargs.get("order_type"),
                "purpose": kwargs.get("purpose"),
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

        yield executor_long, executor_short, tracker, mock_gateway, config

    await test_db.close()
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_zero_server_sl_orders_placed_on_entry(setup_virtual_sl_env):
    """
    CRITICAL OBJECTIVE 1 TEST:
    Verify that upon entry fill, ZERO STOP_MARKET orders are sent to exchange.
    Only Post-Only Take Profit limit orders are placed.
    """
    executor_long, _, tracker, mock_gateway, _ = setup_virtual_sl_env

    entry_fill = FillRecord(
        id="fill_1",
        order_id="ord_e1",
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
    await executor_long.on_entry_fill(entry_fill)

    # Verify barrier is armed
    assert executor_long.state.active is True
    assert executor_long.state.remaining_qty == 5.0
    assert executor_long.state.sl_price == 98.0  # 100 * (1 - 0.02)

    # Verify calls to create_exit_order: only TAKE_PROFIT, NO STOP_MARKET
    calls = mock_gateway.create_exit_order.call_args_list
    assert len(calls) == 1  # Only 1 TP order
    tp_call = calls[0].kwargs
    assert tp_call["purpose"] == OrderPurpose.TAKE_PROFIT
    assert tp_call["order_type"] == OrderType.LIMIT_MAKER
    assert tp_call["price"] == 100.50  # 1% TP discounted to 0.5% by inventory skew boost (500 USDT / 100 max_long)

    # Ensure NO STOP_MARKET order was sent
    for c in calls:
        assert c.kwargs["order_type"] != OrderType.STOP_MARKET


@pytest.mark.asyncio
async def test_virtual_sl_triggers_market_exit_long(setup_virtual_sl_env):
    """
    Verify Virtual Local SL for LONG triggers Market Exit when mark price <= SL trigger.
    """
    executor_long, _, tracker, mock_gateway, _ = setup_virtual_sl_env

    entry_fill = FillRecord(
        id="fill_2",
        order_id="ord_e2",
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
    await executor_long.on_entry_fill(entry_fill)
    mock_gateway.create_exit_order.reset_mock()

    # Price stays safe at 99.0 (> 98.0 SL)
    await executor_long.check_runtime_barriers(current_price=99.0)
    assert mock_gateway.create_exit_order.call_count == 0

    # Price drops to 97.90 (<= 98.0 SL trigger) -> Virtual SL fires!
    await executor_long.check_runtime_barriers(current_price=97.90)

    # Verify Market exit was executed with STOP_LOSS purpose
    assert mock_gateway.create_exit_order.call_count == 1
    call_kw = mock_gateway.create_exit_order.call_args.kwargs
    assert call_kw["order_type"] == OrderType.MARKET
    assert call_kw["purpose"] == OrderPurpose.STOP_LOSS
    assert call_kw["side"] == OrderSide.SELL
    assert call_kw["position_side"] == PositionSide.LONG
    assert call_kw["amount"] == 10.0

    # Verify LONG side entered progressive cooldown
    assert tracker.is_in_cooldown(PositionSide.LONG) is True
    assert tracker.is_in_cooldown(PositionSide.SHORT) is False


@pytest.mark.asyncio
async def test_virtual_sl_triggers_market_exit_short(setup_virtual_sl_env):
    """
    Verify Virtual Local SL for SHORT triggers Market Exit when mark price >= SL trigger.
    """
    _, executor_short, tracker, mock_gateway, _ = setup_virtual_sl_env

    entry_fill = FillRecord(
        id="fill_3",
        order_id="ord_e3",
        symbol="SOL/USDT:USDT",
        side=OrderSide.SELL,
        position_side=PositionSide.SHORT,
        price=100.0,
        amount=10.0,
        quote_amount=1000.0,
        fee=0.2,
        is_maker=True,
        timestamp=time.time(),
    )
    await tracker.on_fill(entry_fill)
    await executor_short.on_entry_fill(entry_fill)
    mock_gateway.create_exit_order.reset_mock()

    # Price rises to 102.10 (>= 102.0 SL trigger) -> Virtual SL fires!
    await executor_short.check_runtime_barriers(current_price=102.10)

    assert mock_gateway.create_exit_order.call_count == 1
    call_kw = mock_gateway.create_exit_order.call_args.kwargs
    assert call_kw["order_type"] == OrderType.MARKET
    assert call_kw["purpose"] == OrderPurpose.STOP_LOSS
    assert call_kw["side"] == OrderSide.BUY
    assert call_kw["position_side"] == PositionSide.SHORT

    # Verify SHORT entered cooldown while LONG is not in cooldown
    assert tracker.is_in_cooldown(PositionSide.SHORT) is True
    assert tracker.is_in_cooldown(PositionSide.LONG) is False


@pytest.mark.asyncio
async def test_min_holding_lock_protects_whipsaw(setup_virtual_sl_env):
    """
    Verify that within min_holding_sec (3.0s), non-SL exits (e.g. time limit) are blocked,
    but Virtual SL still executes immediately.
    """
    executor_long, _, tracker, mock_gateway, config = setup_virtual_sl_env
    config.time_limit = 1  # 1s time limit for test

    entry_fill = FillRecord(
        id="fill_4",
        order_id="ord_e4",
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
    await executor_long.on_entry_fill(entry_fill)
    mock_gateway.create_exit_order.reset_mock()

    # Position opened 1.5s ago (< 3.0s min_holding_sec)
    executor_long.state.entry_timestamp = time.time() - 1.5

    # Safe price: Time limit would expire, but min_holding_sec holds it!
    await executor_long.check_runtime_barriers(current_price=100.2)
    assert mock_gateway.create_exit_order.call_count == 0  # Protected by holding lock!

    # But if price drops below SL (97.5 <= 98.0), Virtual SL triggers immediately without holding delay!
    await executor_long.check_runtime_barriers(current_price=97.5)
    assert mock_gateway.create_exit_order.call_count == 1
    assert mock_gateway.create_exit_order.call_args.kwargs["purpose"] == OrderPurpose.STOP_LOSS


@pytest.mark.asyncio
async def test_passive_maker_time_limit_exit(setup_virtual_sl_env):
    """
    CRITICAL OBJECTIVE 2 TEST:
    Verify that when hold time exceeds time_limit, executor places a Post-Only
    Limit Maker exit order near breakeven/market instead of a Market Taker exit.
    """
    executor_long, _, tracker, mock_gateway, config = setup_virtual_sl_env
    config.time_limit = 60

    entry_fill = FillRecord(
        id="fill_5",
        order_id="ord_e5",
        symbol="SOL/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=100.0,
        amount=10.0,
        quote_amount=1000.0,
        fee=0.2,
        is_maker=True,
        timestamp=time.time() - 70,  # 70s ago > 60s limit
    )
    await tracker.on_fill(entry_fill)
    await executor_long.on_entry_fill(entry_fill)
    executor_long.state.entry_timestamp = time.time() - 70
    mock_gateway.create_exit_order.reset_mock()

    # Run barrier check with market prices: mark=100.1, best_bid=100.08, best_ask=100.12
    await executor_long.check_runtime_barriers(current_price=100.1, best_bid=100.08, best_ask=100.12)

    # Verify Passive Maker exit order was placed as LIMIT_MAKER (Post-Only GTX)
    assert mock_gateway.create_exit_order.call_count == 1
    call_kw = mock_gateway.create_exit_order.call_args.kwargs
    assert call_kw["order_type"] == OrderType.LIMIT_MAKER
    assert call_kw["purpose"] == OrderPurpose.PASSIVE_TIME_LIMIT_EXIT
    assert call_kw["side"] == OrderSide.SELL
    assert call_kw["position_side"] == PositionSide.LONG
    assert call_kw["amount"] == 10.0
    # Price should be max(100 * (1 + 0.0006) = 100.06, best_ask=100.12) -> 100.12
    assert call_kw["price"] == 100.12
    assert executor_long.state.passive_exit_active is True
