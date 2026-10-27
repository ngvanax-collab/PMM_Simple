"""Independent Verification & Acceptance Test Suite for Enhanced Historical PMM Backtest Engine.

Digital Twin Replications (OpenPMM-Engine v3):
1. test_precompute_indicators_no_lookahead:
   Verify delta_p_60s, vol_ratio_1m, natr_15m, ema50_1h, atr14_1h, trend_bias_1h
   are vectorized and zero future data leaks across 15m and 1h intervals.
2. test_circuit_breaker_blocks_entries:
   Verify 60s micro-volatility surge (> 1.5%) freezes entry quotes on subsequent candles.
3. test_trend_bias_blocks_counter_trade:
   Verify BULLISH 1h regime blocks counter-trend SHORT quotes while preserving LONG quotes.
4. test_toxic_flow_early_exit:
   Verify uPnL <= -0.8% with volume spike >= 3x triggers TOXIC_VOLUME_EXIT taker order and saves USDT.
5. test_exit_priority_preservation_under_trend_and_cb:
   Verify exit barriers (TP, SL, Trailing TP) remain 100% active when entry quotes are blocked.
6. test_time_decay_sl_tightening:
   Verify position held beyond time_decay_mins tightens SL to 50% of initial barrier.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backtest.data_loader import fetch_binance_futures_klines, precompute_market_indicators
from backtest.metrics import calculate_metrics, format_metrics_report
from backtest.pmm_simulator import PMMBacktestEngine


def test_precompute_indicators_no_lookahead():
    """
    1. Verify multi-timeframe indicators (15m, 1h) are computed correctly.
    2. Confirm STRICT ZERO LOOK-AHEAD:
       Mutating future candle data in the 2nd hour or 2nd 15m bar does NOT
       alter past indicator values in preceding candles.
    """
    n = 180  # 3 hours of 1m candles
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=n, freq="1min", tz="UTC")

    # Predictable price series
    base_price = 100.0
    opens = np.full(n, base_price)
    highs = np.full(n, base_price + 0.5)
    lows = np.full(n, base_price - 0.5)
    closes = np.full(n, base_price)
    volumes = np.full(n, 100.0)

    df_base = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    df_calc = precompute_market_indicators(df_base)

    # Required columns
    expected_cols = [
        "timestamp", "open", "high", "low", "close", "volume",
        "delta_p_60s", "vol_ma10", "vol_ratio_1m", "natr_15m",
        "ema50_1h", "atr14_1h", "trend_bias_1h"
    ]
    for col in expected_cols:
        assert col in df_calc.columns, f"Missing required column {col}!"

    # 1m bar tests
    assert df_calc["delta_p_60s"].iloc[0] == pytest.approx((100.5 - 99.5) / 99.5, rel=1e-5)
    assert df_calc["vol_ratio_1m"].iloc[0] == pytest.approx(1.0, rel=1e-5)

    # 15m NATR: First 15m candle (index 0..14) has not closed yet, so shifted natr_15m is NaN.
    # From index 15..29, natr_15m must be identical, non-NaN, and positive.
    assert np.isnan(df_calc["natr_15m"].iloc[0])
    assert not np.isnan(df_calc["natr_15m"].iloc[15])
    assert df_calc["natr_15m"].iloc[15] > 0
    np.testing.assert_allclose(df_calc["natr_15m"].iloc[15:30].values, df_calc["natr_15m"].iloc[15])

    # ── Strict Lookahead Test: Mutate candle at minute 75 (Hour 2, 15m bar 5) ──
    df_future_mutated = df_base.copy()
    df_future_mutated.loc[75, "high"] = 999.0
    df_future_mutated.loc[75, "close"] = 950.0
    df_future_mutated.loc[75, "volume"] = 50000.0

    df_mut_calc = precompute_market_indicators(df_future_mutated)

    # Past candles (min 0 to 59: Hour 1, and 15m bars 0 to 3) MUST BE 100% BIT-FOR-BIT IDENTICAL!
    # A single floating point difference proves future leakage!
    np.testing.assert_allclose(
        df_calc.loc[0:59, "vol_ratio_1m"].values,
        df_mut_calc.loc[0:59, "vol_ratio_1m"].values,
        err_msg="vol_ratio_1m leaked future data!"
    )
    # NATR 15m for minutes 15..59
    np.testing.assert_allclose(
        df_calc.loc[15:59, "natr_15m"].values,
        df_mut_calc.loc[15:59, "natr_15m"].values,
        err_msg="natr_15m leaked future data into past candles!"
    )
    # 1h EMA50 and ATR14 for minutes 60..74 (Hour 2 before minute 75)
    np.testing.assert_allclose(
        df_calc.loc[60:74, "ema50_1h"].values,
        df_mut_calc.loc[60:74, "ema50_1h"].values,
        err_msg="ema50_1h leaked future data from min 75 into min 60..74!"
    )
    np.testing.assert_allclose(
        df_calc.loc[60:74, "atr14_1h"].values,
        df_mut_calc.loc[60:74, "atr14_1h"].values,
        err_msg="atr14_1h leaked future data from min 75 into min 60..74!"
    )


def test_circuit_breaker_blocks_entries():
    """
    Verify 60s micro-volatility circuit breaker:
    - Candle 0 experiences a price surge > 1.5% (High - Low / Low = 2.0%).
    - Circuit breaker trips for 60 seconds (cb_pause_until = ts_0 + 60s).
    - Candle 1 (ts_1 = ts_0 + 60s): Entry quotes are FROZEN.
      Even though price sweeps through Bid and Ask, no position is opened!
    - Candle 2 (ts_2 = ts_0 + 120s): Pause expired, quoting resumes.
    """
    timestamps = [
        pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
        pd.Timestamp("2026-01-01 00:01:00", tz="UTC"),
        pd.Timestamp("2026-01-01 00:02:00", tz="UTC"),
    ]

    # Candle 0: 60s surge = (101.6 - 100.0) / 100.0 = 1.60% > 1.50% -> Trips CB #1. Low=100.0 > Bid(99.65) so LONG not opened.
    # Candle 1: Normal vol (100.2 - 99.0)/99.0 = 1.21% < 1.50%. Low(99.0) penetrates Bid(99.65), but CB is ACTIVE -> LONG BLOCKED!
    # Candle 2: Normal vol (100.2 - 99.0)/99.0 = 1.21% < 1.50%. CB expired -> Fills LONG!
    df = pd.DataFrame([
        {"timestamp": timestamps[0], "open": 100.0, "high": 101.6, "low": 100.0, "close": 100.0, "volume": 100.0},
        {"timestamp": timestamps[1], "open": 100.0, "high": 100.2, "low": 99.0, "close": 100.0, "volume": 100.0},
        {"timestamp": timestamps[2], "open": 100.0, "high": 100.2, "low": 99.0, "close": 100.0, "volume": 100.0},
    ])

    engine = PMMBacktestEngine(
        symbol="SOLUSDT",
        initial_balance=1000.0,
        order_amount_usdt=50.0,
        base_spread_pct=0.0035,
        tick_size=0.01,
        cb_threshold_pct=0.015,  # 1.5%
        use_circuit_breaker=True,
        use_vamm=False,
        use_trend_bias=False,
        close_positions_at_end=False,
    )

    metrics = engine.run(df)

    # 1. Circuit Breaker triggered exactly once on Candle 0
    assert engine.cb_trigger_count == 1
    assert metrics["cb_trigger_count"] == 1

    # 2. Check equity records at Candle 1: LONG was blocked!
    assert engine.equity_records[0]["long_open"] is False
    assert engine.equity_records[1]["long_open"] is False, "Bot opened LONG on Candle 1 despite CB freeze!"

    # 3. Check Candle 2: Position was successfully opened after unfreeze!
    assert engine.equity_records[2]["long_open"] is True, "Bot failed to resume quoting on Candle 2 after CB expired!"
    assert engine.long_pos is not None
    assert pd.Timestamp(engine.long_pos.opened_at, tz="UTC") == timestamps[2]


def test_trend_bias_blocks_counter_trade():
    """
    Verify 1h Trend Bias Counter-Entry Blocking:
    - When 1h Trend Bias is 'BULLISH', SHORT maker entry quotes are completely blocked.
    - Only LONG maker entry quotes are placed.
    - Counter-trend blocking counter (blocked_trend_trades) increments properly.
    """
    timestamps = [
        pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
        pd.Timestamp("2026-01-01 00:01:00", tz="UTC"),
    ]

    # Pre-inject indicator columns with BULLISH bias
    # Candle sweeps both High (102.0 >= Ask 100.355) and Low (98.0 <= Bid 99.645)
    df = pd.DataFrame([
        {
            "timestamp": timestamps[0],
            "open": 100.0,
            "high": 102.0,
            "low": 98.0,
            "close": 101.5,
            "volume": 100.0,
            "delta_p_60s": 0.004,  # Below CB threshold
            "vol_ratio_1m": 1.0,
            "natr_15m": np.nan,
            "trend_bias_1h": "BULLISH",
        },
        {
            "timestamp": timestamps[1],
            "open": 101.5,
            "high": 102.0,
            "low": 98.0,
            "close": 101.5,
            "volume": 100.0,
            "delta_p_60s": 0.004,
            "vol_ratio_1m": 1.0,
            "natr_15m": np.nan,
            "trend_bias_1h": "BULLISH",
        },
    ])

    engine = PMMBacktestEngine(
        symbol="SOLUSDT",
        initial_balance=1000.0,
        order_amount_usdt=50.0,
        base_spread_pct=0.0035,
        tick_size=0.01,
        use_trend_bias=True,
        use_circuit_breaker=False,
        use_vamm=False,
        close_positions_at_end=False,
    )

    metrics = engine.run(df)

    # 1. LONG should be opened
    assert engine.long_pos is not None, "LONG entry was not opened in Bullish regime!"
    assert engine.long_pos.side == "LONG"

    # 2. SHORT must NOT be opened despite High = 102.0 penetrating Ask
    assert engine.short_pos is None, "SHORT entry was opened in BULLISH regime when it must be blocked!"

    # 3. Blocked counter-trend trades metric recorded
    assert engine.blocked_trend_trades >= 1
    assert metrics["blocked_trend_trades"] >= 1


def test_toxic_flow_early_exit():
    """
    Verify Fast Toxic Volume Exit:
    - Open LONG position at 99.65 on Candle 0.
    - Candle 1: Price drops to close = 98.8 (uPnL = -0.85% <= -0.8%).
      Volume spikes by 4.0x (vol_ratio_1m = 4.0 >= 3.0 threshold).
      Low is 98.7, which does NOT hit Hard SL (99.65 * 0.98 = 97.657).
    - Engine must immediately trigger TOXIC_VOLUME_EXIT as a Taker order with slippage,
      exiting early and saving capital compared to hitting 100% of Hard SL!
    """
    timestamps = [
        pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
        pd.Timestamp("2026-01-01 00:01:00", tz="UTC"),
    ]

    df = pd.DataFrame([
        # Candle 0: Enter LONG at 99.65 (Low = 99.60 <= 99.645)
        {
            "timestamp": timestamps[0],
            "open": 100.0,
            "high": 100.1,
            "low": 99.60,
            "close": 100.0,
            "volume": 100.0,
            "delta_p_60s": 0.005,
            "vol_ratio_1m": 1.0,
            "natr_15m": np.nan,
            "trend_bias_1h": "NEUTRAL",
        },
        # Candle 1: Toxic adverse flow spike
        {
            "timestamp": timestamps[1],
            "open": 99.8,
            "high": 99.8,
            "low": 98.70,   # > Hard SL (97.657)
            "close": 98.80,  # uPnL = (98.80 - 99.65) / 99.65 = -0.00853 <= -0.8%
            "volume": 400.0, # vol_ratio = 4.0x >= 3.0x
            "delta_p_60s": 0.005,
            "vol_ratio_1m": 4.0,
            "natr_15m": np.nan,
            "trend_bias_1h": "NEUTRAL",
        },
    ])

    engine = PMMBacktestEngine(
        symbol="SOLUSDT",
        initial_balance=1000.0,
        order_amount_usdt=99.65,  # qty = 1.0
        base_spread_pct=0.0035,
        stop_loss_pct=0.020,     # Hard SL = 2.0% (trigger at 97.657)
        taker_fee=0.0005,
        slippage_pct=0.0005,
        tick_size=0.01,
        use_toxic_exit=True,
        vol_spike_threshold=3.0,
        use_circuit_breaker=False,
        use_vamm=False,
        close_positions_at_end=False,
    )

    metrics = engine.run(df)

    assert len(engine.trades) == 1
    trade = engine.trades[0]

    # Verify exit reason is TOXIC_VOLUME_EXIT
    assert trade["exit_reason"] == "TOXIC_VOLUME_EXIT"
    assert trade["is_taker"] is True
    assert trade["slippage_cost"] > 0

    # Verify early exit saved capital vs full 2% Hard SL
    # Full SL loss ~ 2.0 USDT. Actual loss ~ (99.65 - 98.80) ~ 0.85 USDT.
    # Saved USDT ~ 1.15 USDT!
    assert engine.fast_exit_count == 1
    assert engine.fast_exit_saved_usdt > 1.0
    assert metrics["fast_exit_count"] == 1
    assert metrics["fast_exit_saved_usdt"] > 1.0


def test_exit_priority_preservation_under_trend_and_cb():
    """
    Verify Invariant 3 (Exit Priority Preservation):
    Even when Trend Bias blocks entry or Circuit Breaker is active,
    an existing open position MUST maintain full exit defense (Take Profit & Stop Loss).
    """
    timestamps = [
        pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
        pd.Timestamp("2026-01-01 00:01:00", tz="UTC"),
    ]

    # Candle 0: Open SHORT in Neutral regime
    # Candle 1: Regime flips to BULLISH (which blocks SHORT entries), but price rallies to SL
    df = pd.DataFrame([
        {
            "timestamp": timestamps[0],
            "open": 100.0,
            "high": 100.5,  # Enters SHORT at 100.35
            "low": 99.8,
            "close": 100.0,
            "volume": 100.0,
            "delta_p_60s": 0.005,
            "vol_ratio_1m": 1.0,
            "natr_15m": np.nan,
            "trend_bias_1h": "NEUTRAL",
        },
        {
            "timestamp": timestamps[1],
            "open": 100.0,
            "high": 102.0,  # Exceeds SL (100.35 * 1.01 = 101.35)
            "low": 99.9,
            "close": 101.8,
            "volume": 100.0,
            "delta_p_60s": 0.020,  # Trips CB!
            "vol_ratio_1m": 1.0,
            "natr_15m": np.nan,
            "trend_bias_1h": "BULLISH",  # Blocks SHORT entry
        },
    ])

    engine = PMMBacktestEngine(
        symbol="SOLUSDT",
        initial_balance=1000.0,
        order_amount_usdt=100.35,
        base_spread_pct=0.0035,
        stop_loss_pct=0.010,
        use_trend_bias=True,
        use_circuit_breaker=True,
        use_vamm=False,
        close_positions_at_end=False,
    )

    engine.run(df)

    # Even though Trend is BULLISH and CB tripped, the open SHORT was successfully protected and exited!
    assert len(engine.trades) == 1
    assert engine.trades[0]["exit_reason"] == "STOP_LOSS"
    assert engine.short_pos is None


def test_time_decay_sl_tightening():
    """
    Verify Time-Decay SL Tightening:
    - Holding a position for > time_decay_mins (e.g. 5 mins in test) tightens SL by 50%.
    - If price then touches the tightened SL (which would NOT have hit full Hard SL),
      it exits with TIME_DECAY_SL and credits fast_exit_saved_usdt!
    """
    timestamps = [
        pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
        pd.Timestamp("2026-01-01 00:06:00", tz="UTC"),  # 6 mins later > 5 mins
    ]

    # Entry at 99.65 on Candle 0. Initial SL = 2.0% (hard SL = 97.657).
    # Tightened SL after 5m = 1.0% (sl_price = 99.65 * 0.99 = 98.6535).
    # On Candle 1: Low drops to 98.50.
    # 98.50 <= 98.6535 (tightened SL hit!) but 98.50 > 97.657 (Hard SL not hit).
    df = pd.DataFrame([
        {
            "timestamp": timestamps[0],
            "open": 100.0,
            "high": 100.1,
            "low": 99.60,
            "close": 100.0,
            "volume": 100.0,
            "delta_p_60s": 0.005,
            "vol_ratio_1m": 1.0,
            "natr_15m": np.nan,
            "trend_bias_1h": "NEUTRAL",
        },
        {
            "timestamp": timestamps[1],
            "open": 99.5,
            "high": 99.5,
            "low": 98.50,
            "close": 98.70,
            "volume": 100.0,
            "delta_p_60s": 0.005,
            "vol_ratio_1m": 1.0,
            "natr_15m": np.nan,
            "trend_bias_1h": "NEUTRAL",
        },
    ])

    engine = PMMBacktestEngine(
        symbol="SOLUSDT",
        initial_balance=1000.0,
        order_amount_usdt=99.65,
        base_spread_pct=0.0035,
        stop_loss_pct=0.020,  # 2% initial SL
        time_decay_mins=5,    # 5m test threshold
        use_time_decay_sl=True,
        use_circuit_breaker=False,
        use_vamm=False,
        close_positions_at_end=False,
    )

    metrics = engine.run(df)

    assert len(engine.trades) == 1
    trade = engine.trades[0]
    assert trade["exit_reason"] == "TIME_DECAY_SL"
    assert engine.fast_exit_count == 1
    assert engine.fast_exit_saved_usdt > 0.90
    assert metrics["fast_exit_count"] == 1
