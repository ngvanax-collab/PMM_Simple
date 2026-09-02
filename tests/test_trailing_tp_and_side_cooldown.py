"""Unit Tests for Dynamic Trailing Take Profit & Independent Per-Side Progressive Cooldown."""
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
async def setup_trailing_env():
    temp_f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_path = temp_f.name
    temp_f.close()

    test_db = Database(temp_path)
    await test_db.connect()

    try:
        with patch("app.core.executor.db", test_db):
            config = PairConfig(
                symbol="SOL/USDT:USDT",
                tp_levels=[[0.02, 1.0]],
                trailing_tp_enabled=True,
                trailing_tp_activation_pct=0.008,  # +0.8% to activate
                trailing_tp_callback_pct=0.003,    # 0.3% pullback to trigger
                stop_loss=0.02,
                time_limit=3600,
                min_holding_sec=0.0,
                base_cooldown_sec=300,
                cooldown_multiplier=2.0,
                max_cooldown_sec=3600,
                order_amount_usdt=33.0,
                order_levels=2,
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

            yield executor_long, executor_short, tracker, quoter, mock_gateway, config
    finally:
        await test_db.close()
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


@pytest.mark.asyncio
async def test_dynamic_trailing_tp_long_activation_and_pullback(setup_trailing_env):
    """
    CRITICAL OBJECTIVE 4 TEST (LONG):
    Verify Trailing TP activates at +0.8% profit, tracks peak price upward,
    and executes market exit when price pulls back by 0.3% from peak.
    """
    executor_long, _, tracker, _, mock_gateway, _ = setup_trailing_env

    # 1. Entry fill: BUY 10.0 SOL @ 100.0
    entry_fill = FillRecord(
        id="fill_l1",
        order_id="ord_l1",
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

    # Price at 100.5 (+0.5%): not yet activated (requires +0.8% = 100.80)
    await executor_long.check_runtime_barriers(current_price=100.5)
    assert executor_long.state.trailing_tp_active is False
    assert mock_gateway.create_exit_order.call_count == 0

    # Price rises to 101.50 (+1.5%): Trailing TP activates! Peak = 101.50
    await executor_long.check_runtime_barriers(current_price=101.50)
    assert executor_long.state.trailing_tp_active is True
    assert executor_long.state.peak_price == 101.50

    # Price climbs higher to 102.00: Peak updates to 102.00
    await executor_long.check_runtime_barriers(current_price=102.00)
    assert executor_long.state.peak_price == 102.00

    # Price slightly pulls back to 101.80 (pullback 0.196% < 0.3% trigger at 101.694): hold!
    await executor_long.check_runtime_barriers(current_price=101.80)
    assert mock_gateway.create_exit_order.call_count == 0

    # Price drops to 101.60 (pullback > 0.3% from 102.00): Trailing TP fires!
    await executor_long.check_runtime_barriers(current_price=101.60)
    assert mock_gateway.create_exit_order.call_count == 1
    call_kw = mock_gateway.create_exit_order.call_args.kwargs
    assert call_kw["purpose"] == OrderPurpose.TRAILING_TAKE_PROFIT
    assert call_kw["order_type"] == OrderType.MARKET
    assert call_kw["side"] == OrderSide.SELL
    assert call_kw["position_side"] == PositionSide.LONG


@pytest.mark.asyncio
async def test_dynamic_trailing_tp_short_activation_and_pullback(setup_trailing_env):
    """
    CRITICAL OBJECTIVE 4 TEST (SHORT):
    Verify Trailing TP activates at -0.8% profit, tracks trough price downward,
    and executes market exit when price bounces by 0.3% from trough.
    """
    _, executor_short, tracker, _, mock_gateway, _ = setup_trailing_env

    # 1. Entry fill: SELL 10.0 SOL @ 100.0
    entry_fill = FillRecord(
        id="fill_s1",
        order_id="ord_s1",
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

    # Price drops to 98.50 (-1.5% <= 99.20): Trailing TP activates! Trough = 98.50
    await executor_short.check_runtime_barriers(current_price=98.50)
    assert executor_short.state.trailing_tp_active is True
    assert executor_short.state.trough_price == 98.50

    # Price drops further to 98.00: Trough updates to 98.00
    await executor_short.check_runtime_barriers(current_price=98.00)
    assert executor_short.state.trough_price == 98.00

    # Price bounces up to 98.40 (> 98.0 * (1 + 0.003) = 98.294): Trailing TP fires!
    await executor_short.check_runtime_barriers(current_price=98.40)
    assert mock_gateway.create_exit_order.call_count == 1
    call_kw = mock_gateway.create_exit_order.call_args.kwargs
    assert call_kw["purpose"] == OrderPurpose.TRAILING_TAKE_PROFIT
    assert call_kw["order_type"] == OrderType.MARKET
    assert call_kw["side"] == OrderSide.BUY
    assert call_kw["position_side"] == PositionSide.SHORT


@pytest.mark.asyncio
async def test_dynamic_trailing_tp_short_trough_price_init_inf(setup_trailing_env):
    """
    Verify that when trough_price starts at float('inf'),
    Trailing TP for SHORT properly tracks the descending price and bounces to exit.
    """
    _, executor_short, tracker, _, mock_gateway, _ = setup_trailing_env

    entry_fill = FillRecord(
        id="fill_s2",
        order_id="ord_s2",
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
    assert executor_short.state.trough_price == float('inf')

    # Force activate trailing TP with initial trough_price = float('inf')
    executor_short.state.trailing_tp_active = True
    executor_short.state.trough_price = float('inf')

    # Price moves to 98.0 -> trough_price must become 98.0
    await executor_short.check_runtime_barriers(current_price=98.0)
    assert executor_short.state.trough_price == 98.0

    # Price moves further down to 97.5 -> trough_price updates to 97.5
    await executor_short.check_runtime_barriers(current_price=97.5)
    assert executor_short.state.trough_price == 97.5

    # Price bounces to 97.9 (> 97.5 * 1.003 = 97.7925) -> Trigger market exit
    mock_gateway.create_exit_order.reset_mock()
    await executor_short.check_runtime_barriers(current_price=97.9)
    assert mock_gateway.create_exit_order.call_count == 1
    assert mock_gateway.create_exit_order.call_args.kwargs["purpose"] == OrderPurpose.TRAILING_TAKE_PROFIT


@pytest.mark.asyncio
async def test_independent_side_cooldown_and_quoting_isolation(setup_trailing_env):
    """
    CRITICAL OBJECTIVE 5 TEST:
    Verify that when LONG side hits SL:
    1. Only LONG enters progressive cooldown (is_in_cooldown(LONG) == True, is_in_cooldown(SHORT) == False).
    2. Quoting for BIDs (LONG) is paused, while quoting for ASKs (SHORT) continues normally.
    """
    _, _, tracker, quoter, _, _ = setup_trailing_env

    # 1. Trigger SL for LONG side
    tracker.set_sl_cooldown(PositionSide.LONG)

    assert tracker.is_in_cooldown(PositionSide.LONG) is True
    assert tracker.is_in_cooldown(PositionSide.SHORT) is False
    assert tracker.long_pos.consecutive_sl_count == 1
    assert tracker.short_pos.consecutive_sl_count == 0

    # 2. Compute quotes with pause_long_entry=True, pause_short_entry=False
    bids, asks = quoter.calculate_quotes(
        smoothed_mid=100.0,
        long_value_usdt=0.0,
        short_value_usdt=0.0,
        vol_mult=1.0,
        pause_long_entry=True,   # LONG is in cooldown
        pause_short_entry=False, # SHORT is active!
        available_margin=100.0,
    )

    # BIDs MUST be empty (LONG paused), ASKs MUST be populated (SHORT active)
    assert len(bids) == 0, "Expected 0 BIDs when LONG is in cooldown"
    assert len(asks) > 0, "Expected ASKs to be active when SHORT is normal"
    for ask in asks:
        assert ask.position_side == PositionSide.SHORT
        assert ask.side == OrderSide.SELL


@pytest.mark.asyncio
async def test_reset_cooldown_per_side_isolation(setup_trailing_env):
    """
    Verify that profitable TP for SHORT side ONLY resets SHORT cooldown,
    leaving LONG cooldown and consecutive SL count completely untouched.
    """
    _, _, tracker, _, _, _ = setup_trailing_env

    # Both sides hit SL previously
    tracker.set_sl_cooldown(PositionSide.LONG)
    tracker.set_sl_cooldown(PositionSide.LONG)  # LONG count = 2
    tracker.set_sl_cooldown(PositionSide.SHORT) # SHORT count = 1

    assert tracker.long_pos.consecutive_sl_count == 2
    assert tracker.short_pos.consecutive_sl_count == 1

    # SHORT has a profitable Take Profit fill
    tracker.reset_sl_cooldown(PositionSide.SHORT)

    # Verify SHORT is reset, but LONG remains in cooldown!
    assert tracker.short_pos.consecutive_sl_count == 0
    assert tracker.short_pos.in_cooldown is False

    assert tracker.long_pos.consecutive_sl_count == 2
    assert tracker.long_pos.in_cooldown is True
