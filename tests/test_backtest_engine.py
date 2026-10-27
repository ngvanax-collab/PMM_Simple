"""Unit & Integration Tests for Historical PMM Backtest Engine (Hedge Mode Native)."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backtest.data_loader import fetch_binance_futures_klines, normalize_symbol
from backtest.metrics import calculate_metrics, format_metrics_report
from backtest.pmm_simulator import PMMBacktestEngine, PositionSlot


def test_parquet_caching(tmp_path: Path):
    """
    Ensure Parquet caching saves and loads correctly, preserves data types,
    and achieves high performance in-memory retrieval.
    """
    symbol = "SOLUSDT"
    interval = "1m"
    days = 1

    # Create dummy historical dataset
    n = 1440
    timestamps = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    df_dummy = pd.DataFrame({
        "timestamp": timestamps,
        "open": np.linspace(100.0, 150.0, n),
        "high": np.linspace(100.5, 150.5, n),
        "low": np.linspace(99.5, 149.5, n),
        "close": np.linspace(100.2, 150.2, n),
        "volume": np.random.uniform(10.0, 1000.0, n),
    })

    # Save to Parquet cache in tmp_path
    cache_file = tmp_path / f"{symbol}_{interval}_{days}d.parquet"
    df_dummy.to_parquet(cache_file, engine="pyarrow", compression="snappy", index=False)

    # Benchmark read speed
    t0 = time.perf_counter()
    df_loaded = fetch_binance_futures_klines(
        symbol=symbol,
        interval=interval,
        days=days,
        cache_dir=tmp_path,
        force_download=False,
    )
    t_load = time.perf_counter() - t0

    assert len(df_loaded) == n
    assert list(df_loaded.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert pd.api.types.is_datetime64_any_dtype(df_loaded["timestamp"])
    for col in ["open", "high", "low", "close", "volume"]:
        assert df_loaded[col].dtype == np.float64

    # Parquet load must be sub-second
    assert t_load < 0.20


def test_conservative_fill_logic():
    """
    Verify Conservative Maker Fill Model:
    - Bid fills ONLY if Low <= Bid - 0.5 * TickSize.
      If Low == Bid or Low > Bid - 0.5 * TickSize, it must NOT fill!
    - Ask fills ONLY if High >= Ask + 0.5 * TickSize.
      If High == Ask or High < Ask + 0.5 * TickSize, it must NOT fill!
    """
    tick_size = 0.01
    half_tick = 0.5 * tick_size  # 0.005
    spread = 0.0035  # 0.35%

    engine = PMMBacktestEngine(
        symbol="SOLUSDT",
        initial_balance=1000.0,
        order_amount_usdt=50.0,
        base_spread_pct=spread,
        tick_size=tick_size,
        entry_ref="open",
        close_positions_at_end=False,
    )

    # Reference Open = 100.00
    # Raw Bid = 100.00 * (1 - 0.0035) = 99.65
    # Quantized Bid = 99.65
    # Required Low for fill <= 99.65 - 0.005 = 99.645
    # Raw Ask = 100.00 * (1 + 0.0035) = 100.35
    # Quantized Ask = 100.35
    # Required High for fill >= 100.35 + 0.005 = 100.355

    # Case 1: Low touches Bid exactly (99.65) -> MUST NOT FILL BID!
    # Case 2: High touches Ask exactly (100.35) -> MUST NOT FILL ASK!
    df_touch = pd.DataFrame([{
        "timestamp": pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
        "open": 100.0,
        "high": 100.35,  # Touches Ask exactly
        "low": 99.65,    # Touches Bid exactly
        "close": 100.0,
        "volume": 100.0,
    }])

    engine.run(df_touch)
    assert engine.long_pos is None, "Bid filled when Low only touched Bid without penetrating 0.5*TickSize!"
    assert engine.short_pos is None, "Ask filled when High only touched Ask without penetrating 0.5*TickSize!"

    # Case 3: Low penetrates by only 0.2 * TickSize (Low = 99.648 > 99.645) -> MUST NOT FILL BID!
    df_slight = pd.DataFrame([{
        "timestamp": pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
        "open": 100.0,
        "high": 100.352, # < 100.355
        "low": 99.648,   # > 99.645
        "close": 100.0,
        "volume": 100.0,
    }])

    engine.reset()
    engine.run(df_slight)
    assert engine.long_pos is None, "Bid filled with insufficient penetration!"
    assert engine.short_pos is None, "Ask filled with insufficient penetration!"

    # Case 4: Low penetrates past threshold (Low = 99.64 <= 99.645) -> MUST FILL BID!
    # High penetrates past threshold (High = 100.36 >= 100.355) -> MUST FILL ASK!
    df_fill = pd.DataFrame([{
        "timestamp": pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
        "open": 100.0,
        "high": 100.36,
        "low": 99.64,
        "close": 100.0,
        "volume": 100.0,
    }])

    engine.reset()
    engine.run(df_fill)
    assert engine.long_pos is not None, "Bid did not fill when Low penetrated past 0.5*TickSize!"
    assert engine.short_pos is not None, "Ask did not fill when High penetrated past 0.5*TickSize!"
    assert engine.long_pos.entry_price == 99.65
    assert engine.short_pos.entry_price == 100.35


def test_triple_barrier_long_and_short():
    """
    Verify Triple Barrier execution:
    1. Stop Loss: Taker fee (0.05%) + Slippage penalty (0.05%) applied to exit.
    2. Take Profit: Maker fee (0.02%) applied, no slippage penalty.
    3. Trailing Take Profit: Peak/Trough tracked and closes on pullback.
    """
    tick_size = 0.01
    spread = 0.0035
    tp_pct = 0.010    # 1.0%
    sl_pct = 0.020    # 2.0%
    maker_fee = 0.0002
    taker_fee = 0.0005
    slippage = 0.0005

    # ── Test 1: LONG Take Profit ──
    engine_tp = PMMBacktestEngine(
        symbol="SOLUSDT",
        initial_balance=1000.0,
        order_amount_usdt=99.65,  # Qty = 1.0 at entry 99.65
        base_spread_pct=spread,
        take_profit_pct=tp_pct,
        stop_loss_pct=sl_pct,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
        slippage_pct=slippage,
        tick_size=tick_size,
        entry_ref="open",
        close_positions_at_end=False,
    )

    # Candle 0: Enter LONG at 99.65
    # Candle 1: High reaches 101.0 >= TP_Price (99.65 * 1.01 = 100.65)
    df_long_tp = pd.DataFrame([
        {"timestamp": pd.Timestamp("2026-01-01 00:00:00", tz="UTC"), "open": 100.0, "high": 100.1, "low": 99.60, "close": 100.0, "volume": 100.0},
        {"timestamp": pd.Timestamp("2026-01-01 00:01:00", tz="UTC"), "open": 100.0, "high": 101.0, "low": 99.80, "close": 100.8, "volume": 100.0},
    ])

    engine_tp.run(df_long_tp)
    long_trades = [t for t in engine_tp.trades if t["side"] == "LONG"]
    assert len(long_trades) == 1
    t_tp = long_trades[0]

    assert t_tp["exit_reason"] == "TAKE_PROFIT"
    assert t_tp["is_taker"] is False
    assert t_tp["slippage_cost"] == 0.0
    # Maker exit fee = exit_price * qty * maker_fee
    expected_exit_fee = t_tp["exit_price"] * t_tp["qty"] * maker_fee
    assert t_tp["exit_fee"] == pytest.approx(expected_exit_fee, rel=1e-5)
    assert t_tp["gross_pnl"] > 0
    assert t_tp["net_pnl"] == pytest.approx(t_tp["gross_pnl"] - t_tp["total_fee"], rel=1e-5)

    # ── Test 2: LONG Stop Loss ──
    engine_sl = PMMBacktestEngine(
        symbol="SOLUSDT",
        initial_balance=1000.0,
        order_amount_usdt=99.65,
        base_spread_pct=spread,
        take_profit_pct=tp_pct,
        stop_loss_pct=sl_pct,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
        slippage_pct=slippage,
        tick_size=tick_size,
        entry_ref="open",
        close_positions_at_end=False,
    )

    # Candle 0: Enter LONG at 99.65
    # Candle 1: Low reaches 97.0 <= SL_Price (99.65 * 0.98 = 97.657)
    df_long_sl = pd.DataFrame([
        {"timestamp": pd.Timestamp("2026-01-01 00:00:00", tz="UTC"), "open": 100.0, "high": 100.1, "low": 99.60, "close": 100.0, "volume": 100.0},
        {"timestamp": pd.Timestamp("2026-01-01 00:01:00", tz="UTC"), "open": 100.0, "high": 100.0, "low": 97.00, "close": 97.5, "volume": 100.0},
    ])

    engine_sl.run(df_long_sl)
    long_sl_trades = [t for t in engine_sl.trades if t["side"] == "LONG"]
    assert len(long_sl_trades) == 1
    t_sl = long_sl_trades[0]

    assert t_sl["exit_reason"] == "STOP_LOSS"
    assert t_sl["is_taker"] is True
    assert t_sl["slippage_cost"] > 0.0
    # Taker exit fee = exec_price * qty * taker_fee
    expected_taker_fee = t_sl["exit_price"] * t_sl["qty"] * taker_fee
    assert t_sl["exit_fee"] == pytest.approx(expected_taker_fee, rel=1e-5)
    assert t_sl["net_pnl"] < t_sl["gross_pnl"]

    # ── Test 3: SHORT Take Profit ──
    engine_short_tp = PMMBacktestEngine(
        symbol="SOLUSDT",
        initial_balance=1000.0,
        order_amount_usdt=100.35,  # Qty = 1.0 at entry 100.35
        base_spread_pct=spread,
        take_profit_pct=tp_pct,
        stop_loss_pct=sl_pct,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
        slippage_pct=slippage,
        tick_size=tick_size,
        entry_ref="open",
        close_positions_at_end=False,
    )

    # Candle 0: Enter SHORT at 100.35 (High 100.5)
    # Candle 1: Low drops to 99.0 <= TP_Price (100.35 * 0.99 = 99.3465)
    df_short_tp = pd.DataFrame([
        {"timestamp": pd.Timestamp("2026-01-01 00:00:00", tz="UTC"), "open": 100.0, "high": 100.5, "low": 99.9, "close": 100.0, "volume": 100.0},
        {"timestamp": pd.Timestamp("2026-01-01 00:01:00", tz="UTC"), "open": 100.0, "high": 100.1, "low": 99.0, "close": 99.2, "volume": 100.0},
    ])

    engine_short_tp.run(df_short_tp)
    short_tp_trades = [t for t in engine_short_tp.trades if t["side"] == "SHORT"]
    assert len(short_tp_trades) == 1
    t_short_tp = short_tp_trades[0]
    assert t_short_tp["exit_reason"] == "TAKE_PROFIT"
    assert t_short_tp["is_taker"] is False
    assert t_short_tp["gross_pnl"] > 0

    # ── Test 4: Trailing Take Profit ──
    engine_trailing = PMMBacktestEngine(
        symbol="SOLUSDT",
        initial_balance=1000.0,
        order_amount_usdt=99.65,
        base_spread_pct=spread,
        take_profit_pct=0.05,  # Fixed TP far away (5%)
        stop_loss_pct=0.05,
        trailing_activation_pct=0.01,  # 1% activation
        trailing_callback_pct=0.003,   # 0.3% pullback
        tick_size=tick_size,
        entry_ref="open",
        close_positions_at_end=False,
    )

    # Candle 0: Enter LONG at 99.65
    # Candle 1: Price rallies to High = 102.0 (Activates Trailing TP, Peak = 102.0)
    # Candle 2: Price pulls back to Low = 101.5 <= Peak * (1 - 0.003) = 101.694
    df_trailing = pd.DataFrame([
        {"timestamp": pd.Timestamp("2026-01-01 00:00:00", tz="UTC"), "open": 100.0, "high": 100.1, "low": 99.60, "close": 100.0, "volume": 100.0},
        {"timestamp": pd.Timestamp("2026-01-01 00:01:00", tz="UTC"), "open": 100.0, "high": 102.0, "low": 99.9, "close": 101.9, "volume": 100.0},
        {"timestamp": pd.Timestamp("2026-01-01 00:02:00", tz="UTC"), "open": 101.9, "high": 101.9, "low": 101.5, "close": 101.6, "volume": 100.0},
    ])

    engine_trailing.run(df_trailing)
    trail_trades = [t for t in engine_trailing.trades if t["side"] == "LONG"]
    assert len(trail_trades) == 1
    assert trail_trades[0]["exit_reason"] == "TRAILING_TP"
    assert trail_trades[0]["gross_pnl"] > 0


def test_fee_drag_calculation():
    """
    Verify Fee Drag Ratio calculation:
    Fee Drag (%) = (Total Fees / |Gross Realized PnL|) * 100%
    """
    # Sample trade set with known numbers
    trades = [
        {"gross_pnl": 12.0, "total_fee": 1.2, "slippage_cost": 0.0, "net_pnl": 10.8, "side": "LONG"},
        {"gross_pnl": -4.0, "total_fee": 0.8, "slippage_cost": 0.2, "net_pnl": -5.0, "side": "SHORT"},
    ]
    # Gross PnL = 12.0 + (-4.0) = 8.0 USDT
    # Total Fees = 1.2 + 0.8 = 2.0 USDT
    # Expected Fee Drag = (2.0 / 8.0) * 100 = 25.0%

    dummy_equity = pd.DataFrame([
        {"timestamp": pd.Timestamp("2026-01-01", tz="UTC"), "equity": 1000.0, "balance": 1000.0, "upnl": 0.0},
        {"timestamp": pd.Timestamp("2026-01-02", tz="UTC"), "equity": 1005.8, "balance": 1005.8, "upnl": 0.0},
    ])

    metrics = calculate_metrics(trades, dummy_equity, initial_balance=1000.0)

    assert metrics["gross_pnl"] == pytest.approx(8.0)
    assert metrics["total_fees"] == pytest.approx(2.0)
    assert metrics["fee_drag_pct"] == pytest.approx(25.0)
    assert metrics["net_pnl"] == pytest.approx(10.8 - 5.0)

    # Test edge case: Gross PnL is 0
    trades_zero = [
        {"gross_pnl": 5.0, "total_fee": 0.5, "slippage_cost": 0.0, "net_pnl": 4.5, "side": "LONG"},
        {"gross_pnl": -5.0, "total_fee": 0.5, "slippage_cost": 0.0, "net_pnl": -5.5, "side": "SHORT"},
    ]
    metrics_zero = calculate_metrics(trades_zero, dummy_equity, initial_balance=1000.0)
    assert metrics_zero["gross_pnl"] == pytest.approx(0.0)
    assert metrics_zero["fee_drag_pct"] == 100.0


def test_hedge_mode_native_independence():
    """
    Confirm Hedge Mode Invariant:
    LONG and SHORT slots operate independently. Closing one slot leaves the other intact.
    """
    engine = PMMBacktestEngine(
        symbol="SOLUSDT",
        initial_balance=1000.0,
        order_amount_usdt=50.0,
        base_spread_pct=0.0035,
        take_profit_pct=0.006,
        stop_loss_pct=0.015,
        entry_ref="open",
        close_positions_at_end=False,
    )

    # Candle 0: Open both LONG (Low <= 99.645) and SHORT (High >= 100.355)
    # Candle 1: Price goes up to 101.0 -> Triggers LONG TP (High >= 100.24). SHORT stays open!
    df = pd.DataFrame([
        {"timestamp": pd.Timestamp("2026-01-01 00:00:00", tz="UTC"), "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 100.0},
        {"timestamp": pd.Timestamp("2026-01-01 00:01:00", tz="UTC"), "open": 100.0, "high": 101.0, "low": 99.9, "close": 100.8, "volume": 100.0},
    ])

    engine.run(df)
    # LONG was closed by TP
    assert engine.long_pos is None
    # SHORT is still open and unaffected!
    assert engine.short_pos is not None
    assert engine.short_pos.side == "SHORT"
    assert len(engine.trades) == 1
    assert engine.trades[0]["side"] == "LONG"
