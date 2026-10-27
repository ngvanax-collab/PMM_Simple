"""
Test suite for Historical Trade Reconciliation Benchmark (scripts/reconcile_backtest_vs_live.py).
"""
import pytest
import pandas as pd
from scripts.reconcile_backtest_vs_live import LIVE_TRADES, TOKEN_PRESETS, diagnose_unmatched_trade
from backtest.pmm_simulator import PMMBacktestEngine


def test_live_trades_schema():
    assert len(LIVE_TRADES) == 16
    for t in LIVE_TRADES:
        assert "symbol" in t
        assert "side" in t and t["side"] in ("LONG", "SHORT")
        assert "open" in t
        assert "close" in t
        assert "entry" in t and t["entry"] > 0
        assert "exit" in t and t["exit"] > 0
        assert "pnl" in t


def test_token_presets_configured():
    required_tokens = {"SOLUSDT", "XRPUSDT", "UNIUSDT", "1000PEPEUSDT"}
    assert required_tokens.issubset(TOKEN_PRESETS.keys())
    for token, preset in TOKEN_PRESETS.items():
        assert preset["tick_size"] > 0
        assert preset["qty_step"] > 0
        assert preset["base_spread_pct"] > 0
        assert preset["take_profit_pct"] > 0
        assert preset["stop_loss_pct"] > 0


def test_diagnose_unmatched_trend_bias_block():
    timestamps = pd.date_range("2026-10-19 14:40:00", periods=5, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [100.0] * 5,
        "high": [101.0] * 5,
        "low": [99.0] * 5,
        "close": [100.5] * 5,
        "volume": [1000.0] * 5,
        "trend_bias_1h": ["BULLISH"] * 5,
        "delta_p_60s": [0.005] * 5,
        "vol_ratio_1m": [2.0] * 5,
    })

    engine = PMMBacktestEngine(symbol="SOLUSDT", use_trend_bias=True)
    live_short = {"symbol": "SOLUSDT", "side": "SHORT", "open": "2026-10-19 14:42:00"}
    reason = diagnose_unmatched_trade(
        live_trade=live_short,
        utc_open=timestamps[2],
        df=df,
        engine=engine,
    )
    assert "BLOCKED by 1h Trend Bias" in reason
    assert "BULLISH" in reason
