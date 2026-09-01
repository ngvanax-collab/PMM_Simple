"""Unit Tests for PMM Quoter (Inventory Skew, Spread Floor, Clamping)."""
import pytest
from app.core.quoter import PMMQuoter
from app.models.config import PairConfig
from app.models.state import OrderSide, PositionSide


@pytest.fixture
def quoter():
    config = PairConfig(
        symbol="SOL/USDT:USDT",
        bid_spread=0.003,  # 0.3%
        ask_spread=0.003,
        minimum_spread=0.001,  # 0.1%
        order_levels=3,
        order_level_spread=0.002,
        order_level_amount=25.0,
        order_amount_usdt=50.0,
        max_long_usdt=300.0,
        max_short_usdt=300.0,
        gross_exposure_cap_usdt=450.0,
        skew_kappa=1.0,
        skew_gamma_net=0.001,
    )
    return PMMQuoter(
        config=config,
        price_precision=2,
        amount_precision=3,
        tick_size=0.01,
        step_size=0.001,
    )


def test_clamp_quote_bounds(quoter):
    mid_price = 100.0
    spread_floor = quoter.calculate_spread_floor(mid_price)

    # Test bid price clamp
    raw_bid_high = 100.5  # crossed above mid
    clamped_bid = quoter.clamp_quote(raw_bid_high, is_bid=True, mid_price=mid_price, spread_floor=spread_floor)
    assert clamped_bid < mid_price, f"Clamped bid {clamped_bid} must be strictly less than mid {mid_price}"

    # Test ask price clamp
    raw_ask_low = 99.5  # crossed below mid
    clamped_ask = quoter.clamp_quote(raw_ask_low, is_bid=False, mid_price=mid_price, spread_floor=spread_floor)
    assert clamped_ask > mid_price, f"Clamped ask {clamped_ask} must be strictly greater than mid {mid_price}"


def test_spread_floor_never_negative(quoter):
    mid_price = 100.0
    floor = quoter.calculate_spread_floor(mid_price)
    assert floor > 0.001, "Spread floor must be strictly positive"


def test_independent_per_side_skew(quoter):
    mid_price = 100.0

    # Scenario A: Neutral (0 LONG, 0 SHORT)
    bids_neutral, asks_neutral = quoter.calculate_quotes(
        smoothed_mid=mid_price,
        long_value_usdt=0.0,
        short_value_usdt=0.0,
    )

    # Scenario B: Holding heavy LONG (250 USDT LONG, 0 SHORT)
    bids_heavy_long, asks_heavy_long = quoter.calculate_quotes(
        smoothed_mid=mid_price,
        long_value_usdt=250.0,
        short_value_usdt=0.0,
    )

    # Holding LONG should push bid further away (lower bid price) to discourage buying more
    assert bids_heavy_long[0].price < bids_neutral[0].price, "Holding LONG must widen bid spread (lower bid price)"

    # Scenario C: Holding heavy SHORT (0 LONG, 250 USDT SHORT)
    bids_heavy_short, asks_heavy_short = quoter.calculate_quotes(
        smoothed_mid=mid_price,
        long_value_usdt=0.0,
        short_value_usdt=250.0,
    )

    # Holding SHORT should push ask further away (higher ask price) to discourage selling more
    assert asks_heavy_short[0].price > asks_neutral[0].price, "Holding SHORT must widen ask spread (higher ask price)"


def test_gross_cap_halts_quoting(quoter):
    mid_price = 100.0
    # Gross cap is 450 USDT. If LONG = 250 and SHORT = 250 -> gross = 500 >= 450
    bids, asks = quoter.calculate_quotes(
        smoothed_mid=mid_price,
        long_value_usdt=250.0,
        short_value_usdt=250.0,
    )
    assert len(bids) == 0, "Bids should be empty when gross exposure cap is breached"
    assert len(asks) == 0, "Asks should be empty when gross exposure cap is breached"


def test_floor_bid_ceil_ask_quantization():
    """Verify that Bid is always floored and Ask is always ceiled to prevent crossing book."""
    cfg = PairConfig(symbol="XRP/USDT:USDT")
    xrp_quoter = PMMQuoter(
        config=cfg,
        price_precision=4,
        amount_precision=1,
        tick_size=0.0001,
        step_size=0.1,
    )

    # Test Bid (floor)
    raw_bid = 0.52349
    floored_bid = xrp_quoter.quantize_price(raw_bid, is_bid=True)
    assert floored_bid == 0.5234, f"Expected 0.5234 but got {floored_bid}"

    # Test Ask (ceil)
    raw_ask = 0.52341
    ceiled_ask = xrp_quoter.quantize_price(raw_ask, is_bid=False)
    assert ceiled_ask == 0.5235, f"Expected 0.5235 but got {ceiled_ask}"


def test_clamp_quote_with_best_bid_ask(quoter):
    """Verify quote prices respect best_bid/best_ask and stay at least 1 tick away from opposite book."""
    mid_price = 100.0
    best_bid = 99.98
    best_ask = 100.02
    spread_floor = quoter.calculate_spread_floor(mid_price)

    # Bid cannot exceed best_bid or best_ask - tick_size (0.01)
    bid_clamped = quoter.clamp_quote(
        raw_price=99.99,
        is_bid=True,
        mid_price=mid_price,
        spread_floor=spread_floor,
        best_bid=best_bid,
        best_ask=best_ask,
    )
    assert bid_clamped <= best_bid
    assert bid_clamped <= (best_ask - quoter.tick_size)

    # Ask cannot be below best_ask or best_bid + tick_size
    ask_clamped = quoter.clamp_quote(
        raw_price=100.01,
        is_bid=False,
        mid_price=mid_price,
        spread_floor=spread_floor,
        best_bid=best_bid,
        best_ask=best_ask,
    )
    assert ask_clamped >= best_ask
    assert ask_clamped >= (best_bid + quoter.tick_size)


def test_clamp_quote_price_floor_breach(quoter):
    """Verify that when price_floor > max_allowed_bid, clamp_quote returns None and quote is skipped."""
    mid_price = 100.0
    spread_floor = quoter.calculate_spread_floor(mid_price)

    # Set price_floor higher than mid_price (e.g. 101.0 > 100.0)
    quoter.config.price_floor = 101.0

    bid_clamped = quoter.clamp_quote(
        raw_price=99.5,
        is_bid=True,
        mid_price=mid_price,
        spread_floor=spread_floor,
    )
    assert bid_clamped is None, "Expected None when price_floor > max_allowed_bid"

    # Verify calculate_quotes skips bids when price_floor breaches
    bids, asks = quoter.calculate_quotes(
        smoothed_mid=mid_price,
        long_value_usdt=0.0,
        short_value_usdt=0.0,
    )
    assert len(bids) == 0, "Bids should be empty when price_floor breaches max_allowed_bid"
    assert len(asks) > 0, "Asks should still be calculated normally"
