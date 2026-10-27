"""
Unit tests for 30-Day Survival Stress Replay (scripts/stress_test_30d.py).
"""
import pytest
from scripts.stress_test_30d import calculate_sl_streaks, load_token_preset, TARGET_TOKENS


def test_target_tokens_for_stress_replay():
    required = {"SOLUSDT", "XRPUSDT", "NEARUSDT", "AVAXUSDT"}
    assert required.issubset(TARGET_TOKENS.keys())
    for s in required:
        meta = TARGET_TOKENS[s]
        assert meta["tick_size"] > 0
        assert meta["qty_step"] > 0
        assert meta["default_k_spread"] > 0
        assert meta["default_k_tp"] > 0


def test_calculate_sl_streaks_logic():
    # Sequence: WIN, SL, SL, WIN, SL, SL, SL, WIN
    trades = [
        {"exit_reason": "TAKE_PROFIT", "net_pnl": 0.20},
        {"exit_reason": "STOP_LOSS", "net_pnl": -0.50},
        {"exit_reason": "STOP_LOSS", "net_pnl": -0.50},
        {"exit_reason": "TAKE_PROFIT", "net_pnl": 0.20},
        {"exit_reason": "STOP_LOSS", "net_pnl": -0.50},
        {"exit_reason": "STOP_LOSS", "net_pnl": -0.50},
        {"exit_reason": "STOP_LOSS", "net_pnl": -0.50},
        {"exit_reason": "TAKE_PROFIT", "net_pnl": 0.20},
    ]

    max_hard, max_any = calculate_sl_streaks(trades)
    assert max_hard == 3
    assert max_any == 3

    # Sequence with no 3 consecutive SLs
    safe_trades = [
        {"exit_reason": "STOP_LOSS", "net_pnl": -0.50},
        {"exit_reason": "STOP_LOSS", "net_pnl": -0.50},
        {"exit_reason": "TAKE_PROFIT", "net_pnl": 0.20},
        {"exit_reason": "STOP_LOSS", "net_pnl": -0.50},
        {"exit_reason": "TAKE_PROFIT", "net_pnl": 0.20},
    ]
    max_hard_safe, _ = calculate_sl_streaks(safe_trades)
    assert max_hard_safe == 2


def test_load_token_preset_fallback():
    meta = TARGET_TOKENS["SOLUSDT"]
    preset = load_token_preset("non_existent_file.json", "SOL/USDT:USDT", meta)
    assert preset["k_spread"] == meta["default_k_spread"]
    assert preset["k_tp"] == meta["default_k_tp"]
    assert preset["k_sl_mult"] == meta["default_k_sl_mult"]
    assert preset["time_decay_mins"] == meta["default_time_decay_mins"]
