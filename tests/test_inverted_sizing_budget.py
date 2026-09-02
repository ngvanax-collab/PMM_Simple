"""Tests for Inverted Sizing Margin Budget Scaling (TASK H-7)."""
import pytest

from app.core.quoter import PMMQuoter
from app.core.vamm_calculator import compute_dynamic_vamm
from app.models.config import PairConfig


def test_inverted_ladder_scales_to_margin_budget():
    """
    Verify inverted sizing respects available margin budget:
    - Config: allocated_margin=21.0, leverage=5 -> side capacity = 52.5 USDT.
    - calculate_quotes with available_margin=21.0:
      * Number of levels > 1
      * Total notional per side <= 52.5 USDT
      * Ratio approximates 50:30:20 (e.g. 26.25, 15.75, 10.50).
    """
    cfg = PairConfig(
        symbol="SOL/USDT:USDT",
        leverage=5,
        allocated_margin_usdt=21.0,
        order_levels=3,
        inverted_sizing_enabled=True,
        bid_spread=0.003,
        ask_spread=0.003,
        order_level_spread=0.005,
    )
    quoter = PMMQuoter(
        config=cfg,
        price_precision=2,
        amount_precision=3,
        tick_size=0.01,
        step_size=0.001,
        min_amount=0.001,
        min_notional=5.0,
    )

    bids, asks = quoter.calculate_quotes(
        smoothed_mid=100.0,
        long_value_usdt=0.0,
        short_value_usdt=0.0,
        available_margin=21.0,
    )

    assert len(bids) == 3
    assert len(asks) == 3

    total_bid_notional = sum(b.notional for b in bids)
    total_ask_notional = sum(a.notional for a in asks)

    # Both sides must fit within 52.5 USDT (21.0 / 2 * 5)
    assert total_bid_notional <= 52.50 + 0.10
    assert total_ask_notional <= 52.50 + 0.10

    # Verify 50:30:20 distribution
    assert bids[0].notional == pytest.approx(26.25, rel=0.05)
    assert bids[1].notional == pytest.approx(15.75, rel=0.05)
    assert bids[2].notional == pytest.approx(10.50, rel=0.05)


def test_inverted_ladder_prunes_levels_below_min_notional():
    """
    When available margin is heavily restricted (e.g. 8.0 USDT @ 5x = 20.0 USDT / side):
    - Level 0 (50%) = 10.0 USDT (> 5.0 min_notional)
    - Level 1 (30%) = 6.0 USDT (> 5.0 min_notional)
    - Level 2 (20%) = 4.0 USDT (< 5.0 min_notional -> pruned)
    - Only 2 levels emitted.
    """
    cfg = PairConfig(
        symbol="SOL/USDT:USDT",
        leverage=5,
        allocated_margin_usdt=21.0,
        order_levels=3,
        inverted_sizing_enabled=True,
        bid_spread=0.003,
        ask_spread=0.003,
        order_level_spread=0.005,
    )
    quoter = PMMQuoter(
        config=cfg,
        price_precision=2,
        amount_precision=3,
        tick_size=0.01,
        step_size=0.001,
        min_amount=0.001,
        min_notional=5.0,
    )

    bids, asks = quoter.calculate_quotes(
        smoothed_mid=100.0,
        long_value_usdt=0.0,
        short_value_usdt=0.0,
        available_margin=8.0,
    )

    assert len(bids) == 2
    assert len(asks) == 2
    assert bids[0].notional >= 5.0
    assert bids[1].notional >= 5.0
    assert sum(b.notional for b in bids) <= 20.0 + 0.10
