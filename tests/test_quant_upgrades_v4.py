"""Comprehensive Test Suite for 4-Pillar Quantitative MM Upgrades:
1. Dynamic VAMM Order Levels & Inverted Sizing
2. Trend Bias 1h Regime Detection & Counter-Entry Blocking
3. Trend Bias Exit Priority Preservation (Virtual SL / TP / Trailing TP)
4. Favorable Momentum Pyramiding & Guaranteed Profit Stop Loss
"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.core.executor import TripleBarrierExecutor
from app.core.market_state import MarketState, calculate_atr_from_candles, calculate_ema
from app.core.quoter import PMMQuoter
from app.core.vamm_calculator import compute_dynamic_vamm, determine_order_levels
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
         patch("app.core.worker.db.save_order", new_callable=AsyncMock), \
         patch("app.core.executor.db.save_fill", new_callable=AsyncMock), \
         patch("app.core.executor.db.record_pnl", new_callable=AsyncMock):
        yield


# ─────────────────────────────────────────────────────────────
# 1. Test Dynamic Levels & Inverted Sizing
# ─────────────────────────────────────────────────────────────

def test_dynamic_levels_and_inverted_sizing():
    """
    Test determine_order_levels and compute_dynamic_vamm:
    - H <= 0.35 and NATR <= 1.40% -> N = 3, weights = [0.50, 0.30, 0.20]
    - H > 0.44 -> N = 1, weights = [1.00]
    - Otherwise (e.g. H = 0.40) -> N = 2, weights = [0.60, 0.40]
    Verify Quoter allocates capital with inverted sizing.
    """
    # Case A: Low Hurst (0.30) & NATR (1.2%) -> N = 3
    n_3 = determine_order_levels(hurst=0.30, natr_15m_pct=1.20)
    assert n_3 == 3
    params_3 = compute_dynamic_vamm(natr_15m_pct=1.20, hurst=0.30, allocated_margin=21.0, leverage=5)
    assert params_3["order_levels"] == 3
    assert params_3["level_weights"] == [0.50, 0.30, 0.20]
    assert params_3["level_notionals"] == [52.50, 31.50, 21.00]

    # Case B: High Hurst (0.48) -> N = 1
    n_1 = determine_order_levels(hurst=0.48, natr_15m_pct=1.20)
    assert n_1 == 1
    params_1 = compute_dynamic_vamm(natr_15m_pct=1.20, hurst=0.48, allocated_margin=21.0, leverage=5)
    assert params_1["order_levels"] == 1
    assert params_1["level_weights"] == [1.00]
    assert params_1["level_notionals"] == [105.00]

    # Case C: Moderate Hurst (0.40) -> N = 2
    n_2 = determine_order_levels(hurst=0.40, natr_15m_pct=1.20)
    assert n_2 == 2
    params_2 = compute_dynamic_vamm(natr_15m_pct=1.20, hurst=0.40, allocated_margin=21.0, leverage=5)
    assert params_2["order_levels"] == 2
    assert params_2["level_weights"] == [0.60, 0.40]
    assert params_2["level_notionals"] == [63.00, 42.00]

    # Test Quoter applies inverted sizing across all 3 levels
    cfg = PairConfig(
        symbol="SOL/USDT:USDT",
        leverage=5,
        allocated_margin_usdt=21.0,
        order_levels=3,
        inverted_sizing_enabled=True,
        trend_bias_enabled=False,
    )
    quoter = PMMQuoter(config=cfg, price_precision=2, amount_precision=3, tick_size=0.01, step_size=0.001)
    bids, asks = quoter.calculate_quotes(
        smoothed_mid=100.0,
        long_value_usdt=0.0,
        short_value_usdt=0.0,
        level_notionals=params_3["level_notionals"],
    )
    assert len(bids) == 3  # All 3 levels generated with inverted sizing
    assert bids[0].notional == pytest.approx(52.50, rel=0.05)  # 50%
    assert bids[1].notional == pytest.approx(31.50, rel=0.05)  # 30%
    assert bids[2].notional == pytest.approx(21.00, rel=0.05)  # 20%


# ─────────────────────────────────────────────────────────────
# 2. Test Trend Bias Blocks Counter Orders
# ─────────────────────────────────────────────────────────────

def test_trend_bias_blocks_counter_orders():
    """
    Test Trend Bias regime detection & entry quote blocking:
    - Close > EMA50 + (0.5 * ATR) -> BULLISH: Asks = [] (No Short Entries), Bids active.
    - Close < EMA50 - (0.5 * ATR) -> BEARISH: Bids = [] (No Long Entries), Asks active.
    - Otherwise -> NEUTRAL: Both Bids and Asks active.
    """
    cfg = PairConfig(
        symbol="SOL/USDT:USDT",
        leverage=5,
        allocated_margin_usdt=21.0,
        order_levels=3,
        trend_bias_enabled=True,
        trend_ema_period=50,
        trend_atr_buffer_mult=0.5,
    )
    market_state = MarketState(cfg)

    # 1. Generate synthetic 1h candles
    # Base price = 100.0, high=101.0, low=99.0 (ATR ~ 2.0)
    candles_bullish = []
    for i in range(100):
        # Last candle closes at 105.0, well above EMA50 ~ 100.0 + 0.5*2.0 = 101.0
        close_p = 105.0 if i == 99 else 100.0
        candles_bullish.append([i * 3600, 100.0, close_p + 1.0, close_p - 1.0, close_p, 1000.0])

    regime_bull = market_state.update_trend_bias(candles_bullish)
    assert regime_bull == "BULLISH"
    assert market_state.trend_bias_regime == "BULLISH"

    quoter = PMMQuoter(config=cfg, price_precision=2, amount_precision=3, tick_size=0.01, step_size=0.001)

    # In BULLISH regime: Asks must be completely suppressed!
    bids_bull, asks_bull = quoter.calculate_quotes(
        smoothed_mid=105.0,
        long_value_usdt=0.0,
        short_value_usdt=0.0,
        trend_bias_regime="BULLISH",
    )
    assert len(bids_bull) > 0, "Bids (LONG entry) must remain active in Bullish regime"
    assert len(asks_bull) == 0, "Asks (SHORT entry) must be blocked in Bullish regime"

    # 2. Bearish candles
    candles_bearish = []
    for i in range(100):
        close_p = 94.0 if i == 99 else 100.0
        candles_bearish.append([i * 3600, 100.0, close_p + 1.0, close_p - 1.0, close_p, 1000.0])

    regime_bear = market_state.update_trend_bias(candles_bearish)
    assert regime_bear == "BEARISH"

    bids_bear, asks_bear = quoter.calculate_quotes(
        smoothed_mid=94.0,
        long_value_usdt=0.0,
        short_value_usdt=0.0,
        trend_bias_regime="BEARISH",
    )
    assert len(bids_bear) == 0, "Bids (LONG entry) must be blocked in Bearish regime"
    assert len(asks_bear) > 0, "Asks (SHORT entry) must remain active in Bearish regime"

    # 3. Neutral candles
    candles_neutral = []
    for i in range(100):
        candles_neutral.append([i * 3600, 100.0, 101.0, 99.0, 100.0, 1000.0])

    regime_neutral = market_state.update_trend_bias(candles_neutral)
    assert regime_neutral == "NEUTRAL"

    bids_neut, asks_neut = quoter.calculate_quotes(
        smoothed_mid=100.0,
        long_value_usdt=0.0,
        short_value_usdt=0.0,
        trend_bias_regime="NEUTRAL",
    )
    assert len(bids_neut) > 0
    assert len(asks_neut) > 0


# ─────────────────────────────────────────────────────────────
# 3. Test Trend Bias Preserves Exit Orders
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trend_bias_preserves_exit_orders():
    """
    When Trend Bias blocks new entry orders for a side (e.g. SHORT blocked in BULLISH regime),
    an existing open SHORT position MUST retain 100% protection (Virtual SL and Take Profit).
    """
    cfg = PairConfig(
        symbol="SOL/USDT:USDT",
        leverage=5,
        order_amount_usdt=30.0,
        order_levels=3,
        stop_loss=0.02,  # 2.0%
        take_profit=0.01,  # 1.0%
        trend_bias_enabled=True,
    )
    mock_gw = MagicMock()
    mock_gw.get_market_precision.return_value = (2, 3, 0.01, 0.001)
    mock_gw.get_market_limits.return_value = (0.0, 5.0)
    mock_gw.create_exit_order = AsyncMock(return_value={"id": "exit_1", "status": "closed"})
    mock_gw.cancel_order = AsyncMock()

    worker = PMMWorker(cfg, mock_gw)
    entry_price = 100.0
    t0 = 1000000.0

    # 1. Open SHORT position
    fill_short = FillRecord(
        id="fill_s1",
        order_id="ord_s1",
        symbol="SOL/USDT:USDT",
        side=OrderSide.SELL,
        position_side=PositionSide.SHORT,
        price=entry_price,
        amount=1.0,
        quote_amount=100.0,
        timestamp=t0,
        realized_pnl=0.0,
    )
    with patch("time.time", return_value=t0):
        await worker.on_fill(fill_short)

    assert worker.tracker.short_pos.amount == 1.0
    assert worker.executor_short.state.active is True
    assert worker.executor_short.state.sl_price == 102.0  # entry * (1 + 0.02)

    # 2. Set market to BULLISH regime -> SHORT entries are blocked
    worker.market_state.trend_bias_regime = "BULLISH"
    worker.tracker.short_pos.is_trend_blocked = True

    bids, asks = worker.quoter.calculate_quotes(
        smoothed_mid=101.0,
        long_value_usdt=0.0,
        short_value_usdt=100.0,
        trend_bias_regime="BULLISH",
    )
    assert len(asks) == 0, "SHORT entry quotes must be blocked"

    # 3. Price spikes against SHORT to 102.50 (breaching SL 102.00)
    t_sl = t0 + 10.0
    with patch("time.time", return_value=t_sl):
        await worker.on_ticker_update(bid=102.49, ask=102.51, mark=102.50)

    # Virtual SL MUST trigger market exit despite entry being blocked
    mock_gw.create_exit_order.assert_awaited()
    last_call = mock_gw.create_exit_order.call_args
    assert last_call.kwargs["purpose"] == OrderPurpose.STOP_LOSS
    assert last_call.kwargs["side"] == OrderSide.BUY
    assert last_call.kwargs["position_side"] == PositionSide.SHORT
    assert last_call.kwargs["amount"] == 1.0


# ─────────────────────────────────────────────────────────────
# 4. Test Favorable Momentum Pyramiding & Guaranteed Profit SL
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_favorable_pyramiding_and_breakeven_sl():
    """
    Test Favorable Momentum Pyramiding:
    - Open initial LONG position @ 100.0 (size = 1.0).
    - Price moves favorably (+1.2% >= activation 0.8%).
    - 60s micro-momentum >= 0.65 * NATR_15m (0.65 * 0.012 = 0.0078).
    - Bot executes MARKET entry for +50% (+0.5 size).
    - Stop Loss is immediately locked to Guaranteed Profit (above breakeven / initial entry).
    - Trailing callback tightened to 20% of TP_base.
    - When position closes, pyramiding and guaranteed SL state cleanly reset to 0.
    """
    cfg = PairConfig(
        symbol="SOL/USDT:USDT",
        leverage=5,
        order_amount_usdt=100.0,
        allocated_margin_usdt=21.0,
        order_levels=3,
        take_profit=0.010,  # 1.0%
        trailing_tp_enabled=True,
        trailing_tp_activation_pct=0.008,  # +0.8%
        trailing_tp_callback_pct=0.003,
        favorable_pyramiding_enabled=True,
        pyramiding_size_pct=0.50,
        pyramiding_trigger_natr_mult=0.65,
        min_holding_sec=0.0,
    )
    mock_gw = MagicMock()
    mock_gw.get_market_precision.return_value = (2, 3, 0.01, 0.001)
    mock_gw.get_market_limits.return_value = (0.0, 5.0)
    mock_gw.create_entry_market_order = AsyncMock(return_value={"id": "pyr_order_1", "status": "closed"})
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
    assert worker.executor_long.state.is_guaranteed_sl_locked is False

    # 2. Price surges up to 101.20 (+1.2% >= 0.8% activation) within 60s
    t_tick = t0 + 30.0
    price_now = 101.20
    # Simulate 60s price history surge: min price was 100.0, now 101.20 -> surge_up = +1.20% (>= 0.65 * 1.2% = 0.78%)
    worker.market_state.price_history_60s.append((t0, 100.0))
    worker.market_state.price_history_60s.append((t_tick, price_now))
    worker.market_state.current_natr_15m = 0.012

    with patch("time.time", return_value=t_tick):
        await worker.on_ticker_update(bid=price_now - 0.01, ask=price_now + 0.01, mark=price_now)

    # 3. Verify Pyramiding Market Entry Order was placed
    mock_gw.create_entry_market_order.assert_awaited()
    pyr_call = mock_gw.create_entry_market_order.call_args
    assert pyr_call.kwargs["symbol"] == "SOL/USDT:USDT"
    assert pyr_call.kwargs["side"] == OrderSide.BUY
    assert pyr_call.kwargs["position_side"] == PositionSide.LONG
    assert pyr_call.kwargs["amount"] == 0.5  # 50% of initial 1.0

    # Deliver real fill event via worker.on_fill
    pyr_cid = worker.executor_long.state.pending_pyramid_client_id
    fill_pyr = FillRecord(
        id="fill_pyr_real",
        order_id="pyr_order_1",
        client_order_id=pyr_cid,
        symbol="SOL/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=101.20,
        amount=0.5,
        quote_amount=50.60,
        timestamp=t_tick + 0.1,
        realized_pnl=0.0,
    )
    with patch("time.time", return_value=t_tick + 0.1):
        await worker.on_fill(fill_pyr)

    # 4. Verify Position and Guaranteed Profit SL updates
    exec_state = worker.executor_long.state
    assert exec_state.pyramid_filled_count == 1
    assert exec_state.is_guaranteed_sl_locked is True
    assert exec_state.total_qty == 1.5
    assert exec_state.remaining_qty == 1.5

    # New average price = (1.0 * 100.0 + 0.5 * 101.20) / 1.5 = 150.60 / 1.5 = 100.40
    assert exec_state.entry_price == pytest.approx(100.40, rel=1e-4)

    # Guaranteed SL must be above entry_price (100.40 * (1 + s_floor) >= 100.40)
    assert exec_state.sl_price >= 100.40
    assert exec_state.sl_price >= (entry_p * 1.0035)  # >= 100.35

    # Trailing callback must be tightened to 20% of TP_base (0.20 * 0.010 = 0.0020) on local executor
    assert worker.executor_long._trailing_cb_override == pytest.approx(0.0020, rel=1e-4)

    # 5. Simulate closing position
    exit_fill = FillRecord(
        id="fill_close_all",
        order_id="exit_all",
        client_order_id="tp_long_exit",
        symbol="SOL/USDT:USDT",
        side=OrderSide.SELL,
        position_side=PositionSide.LONG,
        price=101.50,
        amount=1.5,
        quote_amount=152.25,
        timestamp=t_tick + 10.0,
        realized_pnl=1.65,
    )
    with patch("time.time", return_value=t_tick + 10.0):
        await worker.on_fill(exit_fill)

    # State must be cleanly reset
    assert worker.tracker.long_pos.amount == 0.0
    assert worker.executor_long.state.pyramid_filled_count == 0
    assert worker.executor_long.state.is_guaranteed_sl_locked is False
    assert worker.tracker.long_pos.pyramid_filled_count == 0
    assert worker.tracker.long_pos.is_guaranteed_sl_locked is False
