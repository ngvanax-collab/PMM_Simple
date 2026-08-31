"""Comprehensive Test Suite for Quantitative Screener and Dynamic Pair Rebalancer."""
import asyncio
import math
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.core.pair_rebalancer import PairRebalancer, RebalancerConfig
from app.core.screener import (
    MarketMetric,
    QuantitativeScreener,
    calculate_hurst_exponent,
    calculate_natr,
    calculate_volatility_ratio,
    compute_pmm_composite_score,
    score_funding_rate,
    score_liquidity,
    score_mean_reversion,
    score_natr_volatility,
)
from app.models.config import PairConfig, ScreenerConfig
from app.models.state import OrderSide, PositionSide, SidePositionState


# ─────────────────────────────────────────────────────────────
# 1. Test Hurst Exponent Calculation
# ─────────────────────────────────────────────────────────────

def test_hurst_exponent_calculation():
    """Verify Hurst Exponent on Mean-Reverting vs Random Walk vs Trending time series."""
    # Synthetic Mean-Reverting series: oscillating sine wave with noise
    mr_series = []
    for i in range(120):
        val = 100.0 + 10.0 * math.sin(i * 0.5) + (i % 2) * 0.5
        mr_series.append(val)

    h_mr = calculate_hurst_exponent(mr_series)
    assert 0.0 <= h_mr < 0.5, f"Mean-reverting series should have H < 0.5, got {h_mr}"

    # Synthetic Trending series: strong upward drift with minor noise
    trend_series = []
    for i in range(120):
        val = 100.0 + (i * 1.5) + ((i % 3) * 0.2)
        trend_series.append(val)

    h_trend = calculate_hurst_exponent(trend_series)
    assert h_trend > 0.5, f"Trending series should have H > 0.5, got {h_trend}"

    # Insufficient length fallback
    short_series = [100.0, 101.0, 102.0]
    h_short = calculate_hurst_exponent(short_series)
    assert h_short == 0.5


# ─────────────────────────────────────────────────────────────
# 2. Test NATR & PMM Composite Scoring Formula
# ─────────────────────────────────────────────────────────────

def test_natr_and_pmm_scoring_formula():
    """Verify NATR calculation and PMM sub-scoring & composite scoring ranges [0, 100]."""
    # Create 30 candles: [timestamp, open, high, low, close, volume]
    candles = []
    base_p = 100.0
    for i in range(30):
        open_p = base_p + (i * 0.1)
        high_p = open_p + 1.5
        low_p = open_p - 1.5
        close_p = open_p + 0.5
        candles.append([1000 + i * 3600, open_p, high_p, low_p, close_p, 1000.0])

    natr = calculate_natr(candles, period=14)
    assert 1.0 <= natr <= 5.0, f"Expected NATR around 2-3%, got {natr}%"

    # Test Sub-Scores
    # 1. Mean Reversion Score (H <= 0.35 -> 100, H -> 0.46 decays to 0)
    assert score_mean_reversion(0.35) == pytest.approx(100.0)  # In optimal zone H <= 0.35
    assert score_mean_reversion(0.10) == pytest.approx(100.0)  # Strong mean-reversion
    assert score_mean_reversion(0.405) == pytest.approx(50.0)  # Midpoint decay
    assert score_mean_reversion(0.46) == pytest.approx(0.0)    # Trend boundary
    assert score_mean_reversion(0.70) == pytest.approx(0.0)    # Trending momentum

    # 2. NATR Volatility Score (Peak 1.0% - 1.5% -> 100, steep penalty > 1.5%)
    assert score_natr_volatility(1.3) == pytest.approx(100.0)  # In optimal zone [1.0%, 1.5%]
    assert score_natr_volatility(0.9) == pytest.approx(80.0)   # [0.8%, 1.0%]
    assert score_natr_volatility(1.65) == pytest.approx(50.0)  # Steep penalty [1.5%, 1.8%]
    assert score_natr_volatility(2.21) == pytest.approx(0.0)   # > 1.8% ceiling breached (PUMP)

    # 3. Liquidity Score
    liq_high = score_liquidity(volume_24h=200_000_000, depth_1pct=200_000, spread_bps=2.0)
    assert liq_high == 100.0
    liq_low = score_liquidity(volume_24h=10_000_000, depth_1pct=20_000, spread_bps=5.0)
    assert liq_low < 50.0

    # 4. Funding Rate Score
    assert score_funding_rate(0.00005) == 100.0  # <= 0.01%
    assert score_funding_rate(0.00010) == 100.0
    assert score_funding_rate(0.00100) == 0.0    # 0.10% extreme funding

    # 5. Composite Score
    score, s_mr, s_natr, s_liq, s_fr = compute_pmm_composite_score(
        hurst=0.35,
        natr_pct=1.3,
        volume_24h=200_000_000,
        depth_1pct=200_000,
        spread_bps=2.0,
        funding_rate=0.00005
    )
    assert score == 100.0
    assert 0.0 <= score <= 100.0


# ─────────────────────────────────────────────────────────────
# 3. Test Screener Filtering & Metric Generation
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_screener_filters_low_volume_and_bad_funding():
    """Verify screener filters out low-volume, blacklisted, and spot pairs."""
    screener = QuantitativeScreener(ScreenerConfig(min_volume_24h_usdt=20_000_000.0))

    mock_gateway = MagicMock()
    mock_gateway._exchange = MagicMock()
    mock_gateway._market_info = {
        "SOL/USDT:USDT": {"active": True, "quote": "USDT", "linear": True, "symbol": "SOL/USDT:USDT"},
        "USDC/USDT:USDT": {"active": True, "quote": "USDT", "linear": True, "symbol": "USDC/USDT:USDT"},  # Blacklisted
        "LOWVOL/USDT:USDT": {"active": True, "quote": "USDT", "linear": True, "symbol": "LOWVOL/USDT:USDT"}, # Low Vol
        "SPOT_PAIR/USDT": {"active": True, "quote": "USDT", "linear": False, "swap": False, "future": False}, # Spot
    }

    async def mock_fetch_ticker_and_mark(symbol):
        return {"bid": 100.0, "ask": 100.05, "mark": 100.02, "percentage": 1.5}

    async def mock_fetch_ohlcv(symbol, timeframe="1h", limit=100):
        # Return 50 sample candles with moderate NATR (~1.3%)
        return [[1000 + i * 3600, 100.0, 100.65, 99.35, 100.0 + (i % 2) * 0.3, 1000.0] for i in range(50)]

    async def mock_fetch_tickers():
        return {
            "SOL/USDT:USDT": {"quoteVolume": 50_000_000.0, "last": 100.0, "bid": 100.0, "ask": 100.05, "percentage": 1.5},
            "USDC/USDT:USDT": {"quoteVolume": 100_000_000.0, "last": 1.0, "bid": 0.9999, "ask": 1.0001, "percentage": 0.01},
            "LOWVOL/USDT:USDT": {"quoteVolume": 5_000_000.0, "last": 10.0, "bid": 9.99, "ask": 10.01, "percentage": 0.5},
        }

    async def mock_fetch_funding_rates():
        return {
            "SOL/USDT:USDT": {"fundingRate": 0.00005},
            "LOWVOL/USDT:USDT": {"fundingRate": 0.00005},
        }

    mock_gateway.fetch_ticker_and_mark = mock_fetch_ticker_and_mark
    mock_gateway._exchange.fetch_tickers = mock_fetch_tickers
    mock_gateway._exchange.fetch_funding_rates = mock_fetch_funding_rates
    mock_gateway._exchange.fetch_ohlcv = mock_fetch_ohlcv

    ranked = await screener.scan_and_rank_all_pairs(mock_gateway)
    symbols = [r.symbol for r in ranked]

    assert "SOL/USDT:USDT" in symbols
    assert "USDC/USDT:USDT" not in symbols  # Filtered by Blacklist
    assert "LOWVOL/USDT:USDT" not in symbols  # Filtered by Volume < 20M
    assert "SPOT_PAIR/USDT" not in symbols  # Filtered by contract type


# ─────────────────────────────────────────────────────────────
# 4. Test Volatility Ceiling & Pump-Dump Filters
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_screener_filters_high_natr_tokens():
    """
    CRITICAL FILTER TEST:
    Verify token with high NATR 14 (e.g. 2.21% > max_natr_pct 1.8%, like PUMP/TRUMP/CHIP)
    is rejected and excluded from candidates.
    """
    screener = QuantitativeScreener(ScreenerConfig(max_natr_pct=1.8, min_natr_pct=0.8))

    mock_gateway = MagicMock()
    mock_gateway._exchange = MagicMock()
    mock_gateway._market_info = {
        "PUMP/USDT:USDT": {"active": True, "quote": "USDT", "linear": True, "symbol": "PUMP/USDT:USDT"},
        "NEAR/USDT:USDT": {"active": True, "quote": "USDT", "linear": True, "symbol": "NEAR/USDT:USDT"},
    }

    async def mock_fetch_tickers():
        return {
            "PUMP/USDT:USDT": {"quoteVolume": 80_000_000.0, "last": 5.0, "bid": 4.999, "ask": 5.001, "percentage": 3.0},
            "NEAR/USDT:USDT": {"quoteVolume": 60_000_000.0, "last": 5.0, "bid": 4.999, "ask": 5.001, "percentage": 1.5},
        }

    async def mock_fetch_ohlcv(symbol, timeframe="1h", limit=100):
        if "PUMP" in symbol:
            # 2.2% NATR: high volatility candles (exceeds 1.8% ceiling)
            return [[1000 + i * 3600, 5.0, 5.15, 4.85, 5.0 + (i % 2) * 0.1, 1000.0] for i in range(50)]
        else:
            # 1.3% NATR: calm mean-reverting candles
            return [[1000 + i * 3600, 5.0, 5.035, 4.965, 5.0 + (i % 2) * 0.02, 1000.0] for i in range(50)]

    mock_gateway._exchange.fetch_tickers = mock_fetch_tickers
    mock_gateway._exchange.fetch_funding_rates = AsyncMock(return_value={})
    mock_gateway._exchange.fetch_ohlcv = mock_fetch_ohlcv

    ranked = await screener.scan_and_rank_all_pairs(mock_gateway)
    symbols = [r.symbol for r in ranked]

    assert "PUMP/USDT:USDT" not in symbols, "PUMP with NATR=2.2% MUST be rejected by max_natr_pct=1.8%"
    assert "NEAR/USDT:USDT" in symbols, "NEAR with NATR=1.3% must pass filter"


@pytest.mark.asyncio
async def test_screener_filters_trending_hurst():
    """
    CRITICAL FILTER TEST:
    Verify token with trending Hurst exponent (H = 0.52 >= max_hurst_exponent 0.46)
    is rejected and excluded from candidate list.
    """
    screener = QuantitativeScreener(ScreenerConfig(max_hurst_exponent=0.46))

    mock_gateway = MagicMock()
    mock_gateway._exchange = MagicMock()
    mock_gateway._market_info = {
        "TREND/USDT:USDT": {"active": True, "quote": "USDT", "linear": True, "symbol": "TREND/USDT:USDT"},
        "ENA/USDT:USDT": {"active": True, "quote": "USDT", "linear": True, "symbol": "ENA/USDT:USDT"},
    }

    async def mock_fetch_tickers():
        return {
            "TREND/USDT:USDT": {"quoteVolume": 50_000_000.0, "last": 100.0, "bid": 99.98, "ask": 100.02, "percentage": 2.0},
            "ENA/USDT:USDT": {"quoteVolume": 50_000_000.0, "last": 1.0, "bid": 0.9998, "ask": 1.0002, "percentage": 1.0},
        }

    async def mock_fetch_ohlcv(symbol, timeframe="1h", limit=100):
        if "TREND" in symbol:
            # Monotonic strong upward drift -> H > 0.50
            return [[1000 + i * 3600, 100.0 + i * 1.5, 101.0 + i * 1.5, 99.5 + i * 1.5, 100.5 + i * 1.5, 1000.0] for i in range(60)]
        else:
            # Mean-reverting sine wave with 1.3% NATR -> H < 0.45
            return [[1000 + i * 3600, 1.0, 1.006, 0.994, 1.0 + 0.004 * math.sin(i * 0.5), 1000.0] for i in range(60)]

    mock_gateway._exchange.fetch_tickers = mock_fetch_tickers
    mock_gateway._exchange.fetch_funding_rates = AsyncMock(return_value={})
    mock_gateway._exchange.fetch_ohlcv = mock_fetch_ohlcv

    ranked = await screener.scan_and_rank_all_pairs(mock_gateway)
    symbols = [r.symbol for r in ranked]

    assert "TREND/USDT:USDT" not in symbols, "Token with trending Hurst H>=0.46 MUST be rejected"
    assert "ENA/USDT:USDT" in symbols, "ENA with mean-reverting Hurst must pass filter"


@pytest.mark.asyncio
async def test_screener_filters_extreme_24h_pump():
    """
    CRITICAL FILTER TEST:
    Verify token with 24h price change = +8.0% (> max_24h_price_change_pct 6.0%)
    is rejected and excluded from candidate list.
    """
    screener = QuantitativeScreener(ScreenerConfig(max_24h_price_change_pct=6.0))

    mock_gateway = MagicMock()
    mock_gateway._exchange = MagicMock()
    mock_gateway._market_info = {
        "PUMP24/USDT:USDT": {"active": True, "quote": "USDT", "linear": True, "symbol": "PUMP24/USDT:USDT"},
        "CALM/USDT:USDT": {"active": True, "quote": "USDT", "linear": True, "symbol": "CALM/USDT:USDT"},
    }

    async def mock_fetch_tickers():
        return {
            "PUMP24/USDT:USDT": {"quoteVolume": 90_000_000.0, "last": 10.0, "bid": 9.998, "ask": 10.002, "percentage": 8.0},
            "CALM/USDT:USDT": {"quoteVolume": 40_000_000.0, "last": 10.0, "bid": 9.998, "ask": 10.002, "percentage": 2.5},
        }

    async def mock_fetch_ohlcv(symbol, timeframe="1h", limit=100):
        return [[1000 + i * 3600, 10.0, 10.06, 9.94, 10.0 + (i % 2) * 0.02, 1000.0] for i in range(50)]

    mock_gateway._exchange.fetch_tickers = mock_fetch_tickers
    mock_gateway._exchange.fetch_funding_rates = AsyncMock(return_value={})
    mock_gateway._exchange.fetch_ohlcv = mock_fetch_ohlcv

    ranked = await screener.scan_and_rank_all_pairs(mock_gateway)
    symbols = [r.symbol for r in ranked]

    assert "PUMP24/USDT:USDT" not in symbols, "Token with 24h change +8.0% MUST be rejected by max_24h_price_change_pct=6.0%"
    assert "CALM/USDT:USDT" in symbols, "Token with 24h change +2.5% must pass"


@pytest.mark.asyncio
async def test_screener_promotes_ideal_mean_reverting_pairs():
    """
    CRITICAL FILTER TEST:
    Verify ideal mean-reverting pairs (NEAR/ENA: H ~ 0.35, NATR ~ 1.4%, 24h change ~ 2%)
    receive top PMM Composite scores and rank #1-#5.
    """
    screener = QuantitativeScreener(ScreenerConfig())

    mock_gateway = MagicMock()
    mock_gateway._exchange = MagicMock()
    mock_gateway._market_info = {
        "NEAR/USDT:USDT": {"active": True, "quote": "USDT", "linear": True, "symbol": "NEAR/USDT:USDT"},
        "ENA/USDT:USDT": {"active": True, "quote": "USDT", "linear": True, "symbol": "ENA/USDT:USDT"},
    }

    async def mock_fetch_tickers():
        return {
            "NEAR/USDT:USDT": {"quoteVolume": 150_000_000.0, "last": 5.0, "bid": 4.999, "ask": 5.001, "percentage": 1.2},
            "ENA/USDT:USDT": {"quoteVolume": 100_000_000.0, "last": 1.0, "bid": 0.9998, "ask": 1.0002, "percentage": -0.8},
        }

    async def mock_fetch_ohlcv(symbol, timeframe="1h", limit=100):
        if "NEAR" in symbol:
            # 1.4% NATR on 5.0: high=5.035, low=4.965
            return [[1000 + i * 3600, 5.0, 5.035, 4.965, 5.0 + 0.02 * math.sin(i * 0.4), 1000.0] for i in range(60)]
        else:
            # 1.4% NATR on 1.0: high=1.007, low=0.993
            return [[1000 + i * 3600, 1.0, 1.007, 0.993, 1.0 + 0.005 * math.sin(i * 0.4), 1000.0] for i in range(60)]

    mock_gateway._exchange.fetch_tickers = mock_fetch_tickers
    mock_gateway._exchange.fetch_funding_rates = AsyncMock(return_value={
        "NEAR/USDT:USDT": {"fundingRate": 0.00003},
        "ENA/USDT:USDT": {"fundingRate": -0.00002},
    })
    mock_gateway._exchange.fetch_ohlcv = mock_fetch_ohlcv

    ranked = await screener.scan_and_rank_all_pairs(mock_gateway)
    assert len(ranked) == 2
    for r in ranked:
        assert r.pmm_score >= 85.0, f"Ideal pair {r.symbol} should have high PMM score, got {r.pmm_score}"
        assert r.rank in [1, 2]


# ─────────────────────────────────────────────────────────────
# 5. Test Hysteresis Prevents Unnecessary Churn
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hysteresis_prevents_unnecessary_churn():
    """Verify rank buffer (Top 7) and 10% score delta prevent unnecessary worker churn."""
    rebalancer = PairRebalancer(config=RebalancerConfig(
        max_active_pairs=5,
        rank_threshold=7,
        score_delta_threshold_pct=0.10
    ))

    # Mock Screener candidates
    # Incumbent pair AVAX is at Rank 6 with score 80.0.
    # Top challengers: SOL (90.0), BTC (88.0), ETH (86.0), DOGE (84.0), LINK (82.0).
    # Rank 5 (LINK) has score 82.0 vs AVAX 80.0 -> delta is (82-80)/80 = 2.5% < 10% -> DO NOT DRAIN.
    mock_candidates = [
        MarketMetric(symbol="SOL/USDT:USDT", pmm_score=90.0, rank=1),
        MarketMetric(symbol="BTC/USDT:USDT", pmm_score=88.0, rank=2),
        MarketMetric(symbol="ETH/USDT:USDT", pmm_score=86.0, rank=3),
        MarketMetric(symbol="DOGE/USDT:USDT", pmm_score=84.0, rank=4),
        MarketMetric(symbol="LINK/USDT:USDT", pmm_score=82.0, rank=5),
        MarketMetric(symbol="AVAX/USDT:USDT", pmm_score=80.0, rank=6),  # Active Incumbent
        MarketMetric(symbol="NEAR/USDT:USDT", pmm_score=70.0, rank=7),
        MarketMetric(symbol="ADA/USDT:USDT", pmm_score=50.0, rank=8),
    ]

    rebalancer.screener.scan_and_rank_all_pairs = AsyncMock(return_value=mock_candidates)

    mock_bot_mgr = MagicMock()
    mock_bot_mgr.gateway = MagicMock()
    mock_bot_mgr.gateway._is_connected = True

    # Setup running AVAX worker
    mock_avax_worker = MagicMock()
    mock_avax_worker.config.enabled = True
    mock_avax_worker.is_draining = False
    mock_avax_worker.is_flat = False
    mock_avax_worker.set_drain_mode = MagicMock()

    mock_bot_mgr.workers = {"AVAX/USDT:USDT": mock_avax_worker}
    mock_bot_mgr.register_dynamic_pair = AsyncMock(return_value=True)

    summary = await rebalancer.execute_rebalance_cycle(mock_bot_mgr)

    # AVAX should NOT be drained because it is within Rank Buffer 7 and delta < 10%
    assert "AVAX/USDT:USDT" in summary["retained_pairs"]
    assert "AVAX/USDT:USDT" not in summary["draining_pairs"]
    mock_avax_worker.set_drain_mode.assert_not_called()

    # Now simulate AVAX falling to Rank 8 (> rank_threshold 7)
    mock_candidates_demoted = [
        MarketMetric(symbol="SOL/USDT:USDT", pmm_score=90.0, rank=1),
        MarketMetric(symbol="BTC/USDT:USDT", pmm_score=88.0, rank=2),
        MarketMetric(symbol="ETH/USDT:USDT", pmm_score=86.0, rank=3),
        MarketMetric(symbol="DOGE/USDT:USDT", pmm_score=84.0, rank=4),
        MarketMetric(symbol="LINK/USDT:USDT", pmm_score=82.0, rank=5),
        MarketMetric(symbol="NEAR/USDT:USDT", pmm_score=75.0, rank=6),
        MarketMetric(symbol="ADA/USDT:USDT", pmm_score=70.0, rank=7),
        MarketMetric(symbol="AVAX/USDT:USDT", pmm_score=40.0, rank=8),  # Demoted
    ]
    rebalancer.screener.scan_and_rank_all_pairs = AsyncMock(return_value=mock_candidates_demoted)

    summary_demoted = await rebalancer.execute_rebalance_cycle(mock_bot_mgr)
    assert "AVAX/USDT:USDT" in summary_demoted["draining_pairs"]
    mock_avax_worker.set_drain_mode.assert_called_with(True)


# ─────────────────────────────────────────────────────────────
# 6. Test Drain & Retire Lifecycle
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_drain_and_retire_lifecycle():
    """Verify worker drain mode halts quotes, lets TP/SL close position, and retires once flat."""
    from app.core.worker import PMMWorker

    mock_gateway = MagicMock()
    mock_gateway.get_market_precision.return_value = (2, 2, 0.01, 0.01)
    mock_gateway.cancel_order = AsyncMock()

    cfg = PairConfig(symbol="SOL/USDT:USDT", enabled=True)
    worker = PMMWorker(cfg, mock_gateway)

    # Simulate active position
    worker.tracker.long_pos = SidePositionState(
        symbol="SOL/USDT:USDT",
        position_side=PositionSide.LONG,
        amount=10.0,
        entry_price=100.0,
        current_price=101.0,
        notional=1010.0,
        unrealized_pnl=10.0
    )
    worker.tracker.short_pos = SidePositionState(
        symbol="SOL/USDT:USDT",
        position_side=PositionSide.SHORT,
        amount=0.0,
        entry_price=0.0,
        current_price=101.0,
        notional=0.0,
        unrealized_pnl=0.0
    )

    assert worker.is_flat is False

    # 1. Activate Drain Mode
    worker.set_drain_mode(True)
    assert worker.is_draining is True

    # 2. Verify _check_should_requote returns False when draining
    worker._running = True
    should_req, reason = worker._check_should_requote()
    assert should_req is False
    assert "Draining" in reason

    # 3. Simulate Rebalance cycle while worker still has position
    rebalancer = PairRebalancer()
    mock_bot_mgr = MagicMock()
    mock_bot_mgr.gateway = MagicMock()
    mock_bot_mgr.gateway._is_connected = True
    mock_bot_mgr.workers = {"SOL/USDT:USDT": worker}
    mock_bot_mgr.stop_pair = AsyncMock()
    mock_bot_mgr.register_dynamic_pair = AsyncMock(return_value=True)
    rebalancer.screener.scan_and_rank_all_pairs = AsyncMock(return_value=[
        MarketMetric(symbol="BTC/USDT:USDT", pmm_score=90.0, rank=1)
    ])

    summary1 = await rebalancer.execute_rebalance_cycle(mock_bot_mgr)
    # Should NOT be retired yet because position is not flat
    assert "SOL/USDT:USDT" not in summary1["retired_pairs"]
    mock_bot_mgr.stop_pair.assert_not_called()

    # 4. Position closes (TP fill simulated -> amount = 0.0)
    worker.tracker.long_pos.amount = 0.0
    assert worker.is_flat is True

    # 5. Next Rebalance cycle sees flat position -> retires worker safely
    summary2 = await rebalancer.execute_rebalance_cycle(mock_bot_mgr)
    assert "SOL/USDT:USDT" in summary2["retired_pairs"]
    mock_bot_mgr.stop_pair.assert_called_with("SOL/USDT:USDT")
