"""
Unit tests for VAMM Grid Search Scanner and Gaussian Plateau Filter (scripts/grid_search_vamm.py).
"""
import numpy as np
import pytest
from scripts.grid_search_vamm import (
    calculate_raw_fitness,
    check_hard_gates,
    compute_gaussian_plateau_scores,
    TARGET_TOKENS,
    K_SPREAD_VALS,
    K_TP_VALS,
    K_SL_MULT_VALS,
    TIME_DECAY_VALS,
)


def test_grid_search_space_dimensions():
    assert len(K_SPREAD_VALS) == 6
    assert len(K_TP_VALS) == 5
    assert len(K_SL_MULT_VALS) == 3
    assert len(TIME_DECAY_VALS) == 3
    total_combinations = len(K_SPREAD_VALS) * len(K_TP_VALS) * len(K_SL_MULT_VALS) * len(TIME_DECAY_VALS)
    assert total_combinations == 270


def test_target_tokens_spec():
    assert "SOLUSDT" in TARGET_TOKENS
    assert "XRPUSDT" in TARGET_TOKENS
    assert "1000PEPEUSDT" in TARGET_TOKENS
    assert "NEARUSDT" in TARGET_TOKENS
    for tok, meta in TARGET_TOKENS.items():
        assert meta["tick_size"] > 0
        assert meta["qty_step"] > 0
        assert "unified" in meta


def test_raw_fitness_calculation():
    # Winning trade scenario: Sharpe=2.5, Trades=36, FeeDrag=20%, MDD=0.5%
    f = calculate_raw_fitness(net_pnl=5.0, sharpe=2.5, trades=36, fee_drag=20.0, mdd=0.5)
    # expected: 2.5 * sqrt(36) * (1 - 0.20) * (1 / (1 + 2*0.5)) = 2.5 * 6 * 0.8 * 0.5 = 6.0
    assert pytest.approx(f, 0.01) == 6.0

    # Negative PnL returns 0.0
    assert calculate_raw_fitness(net_pnl=-1.0, sharpe=2.0, trades=20, fee_drag=20.0, mdd=0.5) == 0.0

    # Zero Sharpe returns 0.0
    assert calculate_raw_fitness(net_pnl=5.0, sharpe=0.0, trades=20, fee_drag=20.0, mdd=0.5) == 0.0


def test_check_hard_gates():
    # Pass scenario
    assert check_hard_gates(net_pnl=1.0, fee_drag=24.0, trades=20, mdd=1.2) is True

    # PnL <= 0 fails
    assert check_hard_gates(net_pnl=0.0, fee_drag=20.0, trades=20, mdd=1.0) is False

    # Fee drag > 30% fails
    assert check_hard_gates(net_pnl=2.0, fee_drag=35.0, trades=20, mdd=1.0) is False

    # Trades < 15 fails
    assert check_hard_gates(net_pnl=2.0, fee_drag=20.0, trades=14, mdd=1.0) is False

    # MDD > 1.50% fails
    assert check_hard_gates(net_pnl=2.0, fee_drag=20.0, trades=20, mdd=1.6) is False


def test_gaussian_plateau_filter_penalizes_isolated_spike():
    # Create 4D grid of zeros
    grid = np.zeros((6, 5, 3, 3), dtype=np.float64)

    # 1. Single isolated spike at (2, 2, 1, 1) with high fitness 10.0
    grid[2, 2, 1, 1] = 10.0

    # 2. Broad plateau at (3, 2, 1, 1) where center is 7.0 and neighbors are all 7.0
    grid[3, 2, 1, 1] = 7.0
    offsets = [(-1, 0, 0, 0), (1, 0, 0, 0), (0, -1, 0, 0), (0, 1, 0, 0),
               (0, 0, -1, 0), (0, 0, 1, 0), (0, 0, 0, -1), (0, 0, 0, 1)]
    for ds, dt, dsl, dtd in offsets:
        grid[3 + ds, 2 + dt, 1 + dsl, 1 + dtd] = 7.0

    scores = compute_gaussian_plateau_scores(grid)

    isolated_score = scores[2, 2, 1, 1]
    plateau_score = scores[3, 2, 1, 1]

    # The isolated peak of 10.0 is penalized by its zero neighbors
    assert isolated_score < 2.0
    # The broad plateau of 7.0 retains its high score (~7.0)
    assert plateau_score > 6.5
    # The broad plateau decisively outranks the isolated spike!
    assert plateau_score > isolated_score
