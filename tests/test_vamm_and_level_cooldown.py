"""Comprehensive Test Suite for VAMM Dynamic Spreads & 30-Minute Level Cooldown with Uninterrupted TP/SL."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.core.executor import TripleBarrierExecutor
from app.core.quoter import PMMQuoter
from app.core.screener import compute_vamm_parameters
from app.core.worker import PMMWorker
from app.models.config import PairConfig
from app.models.state import (
    FillRecord,
    OrderPurpose,
    OrderRecord,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
)


@pytest.fixture(autouse=True)
def mock_db_ops():
    with patch("app.core.worker.db.save_fill", new_callable=AsyncMock), \
         patch("app.core.worker.db.record_pnl", new_callable=AsyncMock), \
         patch("app.core.executor.db.save_fill", new_callable=AsyncMock), \
         patch("app.core.executor.db.record_pnl", new_callable=AsyncMock):
        yield


# ─────────────────────────────────────────────────────────────
# 1. Test Dynamic VAMM Parameter Generation
# ─────────────────────────────────────────────────────────────

def test_dynamic_vamm_parameter_generation():
    """
    Ensure Spread, Step, TP, Trailing Callback, SL, and Flat Capital Allocation
    are strictly calculated according to the VAMM mathematical specification.
    """
    # Case A: Typical NATR = 1.2% (0.012)
    natr_pct = 1.2
    allocated_margin = 21.0
    leverage = 5
    order_levels = 3
    maker_fee = 0.0002
    taker_fee = 0.0005

    params = compute_vamm_parameters(
        natr_pct=natr_pct,
        allocated_margin=allocated_margin,
        leverage=leverage,
        order_levels=order_levels,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
    )

    natr_dec = 0.012
    expected_s_floor = max(0.0025, 2 * maker_fee + taker_fee + 0.0010)
    expected_base_spread = max(expected_s_floor, 0.30 * natr_dec)
    expected_lvl_spread = 0.45 * natr_dec
    expected_tp = max(0.005, 0.60 * natr_dec)
    expected_trailing_act = expected_tp
    expected_trailing_cb = round(0.35 * expected_tp, 6)
    expected_s_max = expected_base_spread + (order_levels - 1) * expected_lvl_spread
    expected_sl = expected_s_max + 0.60 * natr_dec
    expected_order_amt = 30.0
    expected_gross_cap = 105.0

    assert params["bid_spread"] == pytest.approx(expected_base_spread, rel=1e-5)
    assert params["ask_spread"] == pytest.approx(expected_base_spread, rel=1e-5)
    assert params["order_level_spread"] == pytest.approx(expected_lvl_spread, rel=1e-5)
    assert params["take_profit"] == pytest.approx(expected_tp, rel=1e-5)
    assert params["trailing_tp_enabled"] is True
    assert params["trailing_tp_activation_pct"] == pytest.approx(expected_trailing_act, rel=1e-5)
    assert params["trailing_tp_callback_pct"] == pytest.approx(expected_trailing_cb, rel=1e-5)
    assert params["stop_loss"] == pytest.approx(expected_sl, rel=1e-5)
    assert params["order_amount_usdt"] == expected_order_amt
    assert params["order_level_amount"] == 0.0
    assert params["allocated_margin_usdt"] == 21.0
    assert params["max_long_usdt"] == 96.0
    assert params["max_short_usdt"] == 96.0
    assert params["gross_exposure_cap_usdt"] == expected_gross_cap
    assert params["level_cooldown_sec"] == 1800

    # Case B: Low NATR (0.5%) -> Should be bound by spread floor 0.0025 and min TP 0.005
    params_low = compute_vamm_parameters(natr_pct=0.5)
    assert params_low["bid_spread"] == 0.0025
    assert params_low["take_profit"] == 0.005
    assert params_low["allocated_margin_usdt"] == 21.0
    assert params_low["order_amount_usdt"] == 30.0


# ─────────────────────────────────────────────────────────────
# 2. Test Level Cooldown Suppresses Next Level Order
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_level_cooldown_suppresses_next_level_order():
    """
    When flat: only Level 0 is quoted.
    When Level 0 fills: Level 1 is suppressed for 1800 seconds.
    """
    cfg = PairConfig(
        symbol="XPL/USDT:USDT",
        leverage=5,
        order_amount_usdt=30.0,
        order_levels=3,
        order_level_spread=0.005,
        level_cooldown_sec=1800,
        bid_spread=0.0035,
        ask_spread=0.0035,
        stop_loss=0.018,
    )
    quoter = PMMQuoter(config=cfg, price_precision=4, amount_precision=2)
    mock_gw = MagicMock()
    mock_gw.get_market_precision.return_value = (4, 2, 0.0001, 0.01)
    mock_gw.get_market_limits.return_value = (0.0, 5.0)
    mock_gw.fetch_free_balance = AsyncMock(return_value=1000.0)

    worker = PMMWorker(cfg, mock_gw)
    mid_price = 0.0870
    worker.market_state.best_bid = 0.0869
    worker.market_state.best_ask = 0.0871
    worker.market_state.smoothed_mid = mid_price

    # 1. When flat (filled_levels_count == 0): only Level 0 is quoted
    bids_flat, asks_flat = quoter.calculate_quotes(
        smoothed_mid=mid_price,
        long_value_usdt=0.0,
        short_value_usdt=0.0,
        long_state=worker.tracker.long_pos,
        short_state=worker.tracker.short_pos,
    )
    assert len(bids_flat) == 1
    assert bids_flat[0].level == 0
    assert len(asks_flat) == 1
    assert asks_flat[0].level == 0

    # 2. Level 0 fills on LONG
    fill_time = 1000000.0
    fill_lvl0 = FillRecord(
        id="fill_0",
        order_id="ord_0",
        symbol="XPL/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=0.0867,
        amount=346.0,
        quote_amount=30.0,
        timestamp=fill_time,
        realized_pnl=0.0,
    )
    await worker.tracker.on_fill(fill_lvl0)

    assert worker.tracker.long_pos.filled_levels_count == 1
    assert worker.tracker.long_pos.last_level_fill_time == fill_time
    assert worker.tracker.long_pos.next_allowed_level_time == fill_time + 1800

    # 3. Test during 1800s cooldown (e.g. 500s after fill): Level 1 must be suppressed (bids empty)
    bids_in_cd, asks_in_cd = quoter.calculate_quotes(
        smoothed_mid=mid_price,
        long_value_usdt=worker.tracker.long_pos.notional,
        short_value_usdt=0.0,
        long_state=worker.tracker.long_pos,
        short_state=worker.tracker.short_pos,
        current_time=fill_time + 500.0,
        mark_price=0.0860,  # even if price dipped lower, cooldown must suppress!
    )
    assert len(bids_in_cd) == 0, "Bids must be empty during Level Cooldown"
    assert len(asks_in_cd) == 1, "Opposite side (SHORT) is unaffected and quotes Level 0"


# ─────────────────────────────────────────────────────────────
# 3. Test Trailing TP Executes During Level Cooldown
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trailing_tp_executes_during_level_cooldown():
    """
    Position is in 30-min level cooldown.
    Price moves in profitable direction -> Trailing TP activates, tracks peak, pullbacks, and executes TP.
    Upon position close, level state is completely reset to 0.
    """
    cfg = PairConfig(
        symbol="XPL/USDT:USDT",
        leverage=5,
        order_amount_usdt=30.0,
        order_levels=3,
        level_cooldown_sec=1800,
        take_profit=0.008,
        trailing_tp_enabled=True,
        trailing_tp_activation_pct=0.008,  # +0.8%
        trailing_tp_callback_pct=0.003,    # -0.3% pullback
        min_holding_sec=0.0,
    )
    mock_gw = MagicMock()
    mock_gw.get_market_precision.return_value = (4, 2, 0.0001, 0.01)
    mock_gw.get_market_limits.return_value = (0.0, 5.0)
    mock_gw.create_exit_order = AsyncMock(return_value={"id": "tp_exit_1", "status": "closed"})
    mock_gw.cancel_order = AsyncMock()

    worker = PMMWorker(cfg, mock_gw)
    entry_price = 0.0870
    t0 = 1000000.0

    # Fill Level 0 LONG at t0
    fill_0 = FillRecord(
        id="fill_l0",
        order_id="ord_l0",
        symbol="XPL/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=entry_price,
        amount=345.0,
        quote_amount=30.0,
        timestamp=t0,
        realized_pnl=0.0,
    )
    with patch("time.time", return_value=t0):
        await worker.on_fill(fill_0)

    # Position is in cooldown until t0 + 1800
    assert worker.tracker.long_pos.filled_levels_count == 1
    assert worker.tracker.long_pos.next_allowed_level_time == t0 + 1800

    # Simulate price tick 60s later (within cooldown) reaching Trailing TP activation (+1.0%)
    t_tick = t0 + 60.0
    price_act = entry_price * 1.010  # 0.08787
    with patch("time.time", return_value=t_tick):
        await worker.on_ticker_update(bid=price_act - 0.0001, ask=price_act + 0.0001, mark=price_act)

    assert worker.executor_long.state.trailing_tp_active is True
    assert worker.executor_long.state.peak_price == pytest.approx(price_act, rel=1e-5)

    # Price pulls back by 0.4% from peak (exceeding 0.3% callback) -> Triggers Trailing TP Exit!
    price_pb = price_act * (1.0 - 0.004)
    with patch("time.time", return_value=t_tick + 5.0):
        await worker.on_ticker_update(bid=price_pb - 0.0001, ask=price_pb + 0.0001, mark=price_pb)

    mock_gw.create_exit_order.assert_awaited()
    last_call = mock_gw.create_exit_order.call_args
    assert last_call.kwargs["purpose"] == OrderPurpose.TRAILING_TAKE_PROFIT
    assert last_call.kwargs["side"] == OrderSide.SELL
    assert last_call.kwargs["position_side"] == PositionSide.LONG

    # Simulate exit fill
    exit_fill = FillRecord(
        id="fill_exit_tp",
        order_id="tp_exit_1",
        client_order_id="exit_trailing_take_profit_123",
        symbol="XPL/USDT:USDT",
        side=OrderSide.SELL,
        position_side=PositionSide.LONG,
        price=price_pb,
        amount=345.0,
        quote_amount=30.2,
        timestamp=t_tick + 6.0,
        realized_pnl=0.25,
    )
    with patch("app.persistence.db.db.record_pnl", new_callable=AsyncMock), patch("time.time", return_value=t_tick + 6.0):
        await worker.on_fill(exit_fill)

    # Verify complete reset when flat
    assert worker.tracker.long_pos.amount == 0.0
    assert worker.tracker.long_pos.filled_levels_count == 0
    assert worker.tracker.long_pos.last_level_fill_time == 0.0
    assert worker.tracker.long_pos.next_allowed_level_time == 0.0
    assert worker.tracker.long_pos.next_allowed_level_time == 0.0


# ─────────────────────────────────────────────────────────────
# 4. Test Virtual SL Triggers & Adds Side Cooldown During Level Cooldown
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sl_triggers_and_adds_side_cooldown_during_level_cooldown():
    """
    Position is in 30-min level cooldown.
    Price moves against position and breaches Stop Loss barrier ->
    Virtual SL executes emergency market exit immediately, position resets, and triggers side progressive cooldown.
    """
    cfg = PairConfig(
        symbol="XPL/USDT:USDT",
        leverage=5,
        order_amount_usdt=30.0,
        order_levels=3,
        level_cooldown_sec=1800,
        stop_loss=0.018,  # 1.8%
        base_cooldown_sec=600,
    )
    mock_gw = MagicMock()
    mock_gw.get_market_precision.return_value = (4, 2, 0.0001, 0.01)
    mock_gw.get_market_limits.return_value = (0.0, 5.0)
    mock_gw.create_exit_order = AsyncMock(return_value={"id": "sl_exit_1", "status": "closed"})
    mock_gw.cancel_order = AsyncMock()

    worker = PMMWorker(cfg, mock_gw)
    entry_price = 0.0870
    t0 = 1000000.0

    # Fill Level 0 SHORT
    fill_0 = FillRecord(
        id="fill_s0",
        order_id="ord_s0",
        symbol="XPL/USDT:USDT",
        side=OrderSide.SELL,
        position_side=PositionSide.SHORT,
        price=entry_price,
        amount=345.0,
        quote_amount=30.0,
        timestamp=t0,
        realized_pnl=0.0,
    )
    with patch("time.time", return_value=t0):
        await worker.on_fill(fill_0)

    assert worker.tracker.short_pos.filled_levels_count == 1
    assert worker.tracker.short_pos.next_allowed_level_time == t0 + 1800

    # Price breaches SL for SHORT (mark price >= entry * (1 + 0.018) = 0.088566)
    sl_breach_price = entry_price * 1.020  # +2.0%
    t_sl = t0 + 120.0
    with patch("time.time", return_value=t_sl):
        await worker.on_ticker_update(bid=sl_breach_price - 0.0001, ask=sl_breach_price + 0.0001, mark=sl_breach_price)

    # Virtual SL must have triggered emergency Market exit
    mock_gw.create_exit_order.assert_awaited()
    last_call = mock_gw.create_exit_order.call_args
    assert last_call.kwargs["purpose"] == OrderPurpose.STOP_LOSS
    assert last_call.kwargs["side"] == OrderSide.BUY
    assert last_call.kwargs["position_side"] == PositionSide.SHORT

    # Progressive SL Cooldown activated for SHORT side
    assert worker.tracker.short_pos.consecutive_sl_count == 1
    assert worker.tracker.short_pos.cooldown_until == t_sl + 600

    # Simulate SL fill
    sl_fill = FillRecord(
        id="fill_exit_sl",
        order_id="sl_exit_1",
        client_order_id="exit_stop_loss_123",
        symbol="XPL/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.SHORT,
        price=sl_breach_price,
        amount=345.0,
        quote_amount=30.6,
        timestamp=t_sl + 1.0,
        realized_pnl=-0.60,
    )
    with patch("app.persistence.db.db.record_pnl", new_callable=AsyncMock):
        await worker.on_fill(sl_fill)

    # SHORT position is flat -> level state reset
    assert worker.tracker.short_pos.amount == 0.0
    assert worker.tracker.short_pos.filled_levels_count == 0


# ─────────────────────────────────────────────────────────────
# 5. Test Post-Cooldown Conditional Order Placement
# ─────────────────────────────────────────────────────────────

def test_post_cooldown_conditional_order_placement():
    """
    After 1800s cooldown expires:
    - If mark_price meets threshold (SHORT: mark_price >= entry * (1 + order_level_spread)) -> quotes Level 1.
    - If mark_price has NOT met threshold -> does NOT quote Level 1.
    """
    cfg = PairConfig(
        symbol="XPL/USDT:USDT",
        leverage=5,
        order_amount_usdt=30.0,
        order_levels=3,
        order_level_spread=0.005,  # 0.5%
        level_cooldown_sec=1800,
        bid_spread=0.0035,
        ask_spread=0.0035,
    )
    quoter = PMMQuoter(config=cfg, price_precision=4, amount_precision=2)

    entry_price = 0.0870
    t0 = 1000000.0

    mock_gw = MagicMock()
    mock_gw.get_market_precision.return_value = (4, 2, 0.0001, 0.01)
    worker = PMMWorker(cfg, mock_gw)

    # Set up active SHORT position with Level 0 filled
    short_state = worker.tracker.short_pos
    short_state.amount = 345.0
    short_state.entry_price = entry_price
    short_state.notional = 30.0
    short_state.filled_levels_count = 1
    short_state.last_level_fill_time = t0
    short_state.next_allowed_level_time = t0 + 1800

    # Required threshold for SHORT Level 1 = entry_price * (1 + 0.005) = 0.087435
    threshold_price = entry_price * (1.0 + cfg.order_level_spread)

    # Case A: After 1800s (e.g. t0 + 1801), but mark_price = 0.0872 (< threshold) -> NO Level 1 quote!
    t_after_cd = t0 + 1801.0
    bids_below_thresh, asks_below_thresh = quoter.calculate_quotes(
        smoothed_mid=0.0872,
        long_value_usdt=0.0,
        short_value_usdt=short_state.notional,
        short_state=short_state,
        current_time=t_after_cd,
        mark_price=0.0872,
    )
    assert len(asks_below_thresh) == 0, "Must not place Level 1 if price has not reached threshold"

    # Case B: After 1800s, and mark_price = 0.0875 (>= threshold 0.087435) -> Places Level 1 quote!
    bids_above_thresh, asks_above_thresh = quoter.calculate_quotes(
        smoothed_mid=0.0875,
        long_value_usdt=0.0,
        short_value_usdt=short_state.notional,
        short_state=short_state,
        current_time=t_after_cd,
        mark_price=0.0875,
    )
    assert len(asks_above_thresh) == 1, "Must place Level 1 when price condition is met"
    assert asks_above_thresh[0].level == 1
