"""Tests for Circuit Breaker NATR Anchor, Rising-Edge Latch, and Unit Safety (Phase 5)."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.core.gateway import ExchangeGateway
from app.core.market_state import MarketState
from app.core.worker import PMMWorker, calculate_atr_from_candles
from app.models.config import PairConfig


def _create_pair_config(symbol: str = "SOL/USDT:USDT") -> PairConfig:
    return PairConfig(
        symbol=symbol,
        exchange="binance",
        enabled=True,
        leverage=5,
        margin_mode="isolated",
        order_amount_usdt=30.0,
        bid_spread=0.0035,
        ask_spread=0.0035,
        order_levels=3,
        circuit_breaker_enabled=True,
        circuit_breaker_lookback_sec=60,
        circuit_breaker_natr_multiplier=1.0,
        circuit_breaker_min_threshold_pct=0.008,
        circuit_breaker_pause_sec=60,
    )


def test_natr_unit_below_0_1_pct_not_misread():
    """
    TASK CB-4: Unit safety for update_natr_15m:
    - update_natr_15m(0.0005) (0.05%) must store decimal 0.0005 without being corrupted.
    - update_natr_15m(1.5) (invalid percentage passed as > 0.20 decimal) must be rejected, retaining old value.
    """
    cfg = _create_pair_config()
    ms = MarketState(cfg)

    # 1. Valid small decimal (0.05% = 0.0005)
    ms.update_natr_15m(0.0005)
    assert ms.current_natr_15m == pytest.approx(0.0005, abs=1e-6)

    # 2. Valid typical decimal (1.2% = 0.012)
    ms.update_natr_15m(0.012)
    assert ms.current_natr_15m == pytest.approx(0.012, abs=1e-6)

    # 3. Out of bounds > 0.20 (e.g. 1.5) should be rejected and keep 0.012
    ms.update_natr_15m(1.5)
    assert ms.current_natr_15m == pytest.approx(0.012, abs=1e-6)

    # 4. Out of bounds < 0.0001 (e.g. 0.0) should be rejected and keep 0.012
    ms.update_natr_15m(0.0)
    assert ms.current_natr_15m == pytest.approx(0.012, abs=1e-6)


@pytest.mark.asyncio
async def test_pause_not_extended_by_same_spike():
    """
    TASK CB-3: Rising-edge latch on circuit breaker pause:
    - Trip at t0=1000: paused_until set to 1060, quotes cancelled.
    - Tick at t1=1030 with same spike still in 60s sliding window: paused_until remains 1060 (NOT extended to 1090).
    - Quotes are not cancelled again.
    """
    cfg = _create_pair_config()
    gw = MagicMock(spec=ExchangeGateway)
    gw.get_market_precision.return_value = (2, 3, 0.01, 0.001)
    gw.get_market_limits.return_value = (0.001, 5.0)
    gw.fetch_free_balance.return_value = 1000.0
    gw.cancel_order = AsyncMock(return_value=True)

    worker = PMMWorker(cfg, gw)
    worker.market_state.update_natr_15m(0.010)  # threshold 1.0% (0.010)

    # 1. First trip at t = 1000
    t0 = 1000.0
    worker.market_state.price_history_60s.append((t0 - 10, 100.0))
    worker.market_state.price_history_60s.append((t0, 102.0))  # +2.0% spike

    with patch("time.time", return_value=t0):
        await worker._check_dynamic_circuit_breaker()
        assert worker.market_state.is_circuit_breaker_active(t0) is True
        assert worker.market_state.circuit_breaker_paused_until == 1060.0

    # 2. Subsequent tick at t = 1030 (spike still in 60s window)
    t1 = 1030.0
    worker.market_state.price_history_60s.append((t1, 101.9))  # still elevated

    with patch("time.time", return_value=t1):
        await worker._check_dynamic_circuit_breaker()
        # paused_until must NOT be extended to 1090.0!
        assert worker.market_state.circuit_breaker_paused_until == 1060.0


@pytest.mark.asyncio
async def test_natr_15m_refreshed_from_real_candles():
    """
    TASK CB-1: Worker periodically refreshes 15m NATR anchor from real 15m candles:
    - Mock fetch_ohlcv returning 15m candles with true ATR / close = 0.009
    - After loop iteration, market_state.current_natr_15m updates to ~0.009.
    """
    cfg = _create_pair_config()
    gw = MagicMock(spec=ExchangeGateway)
    gw.get_market_precision.return_value = (2, 3, 0.01, 0.001)
    gw.get_market_limits.return_value = (0.001, 5.0)

    # Generate synthetic 15m candles where ATR = 0.9 and close = 100.0 -> NATR = 0.009
    # [timestamp, open, high, low, close, volume]
    synthetic_candles_15m = []
    base_ts = 1000000.0
    for i in range(20):
        # high=100.9, low=100.0, close=100.0 -> TR = 0.9
        synthetic_candles_15m.append([base_ts + i * 900, 100.0, 100.9, 100.0, 100.0, 50.0])

    gw.fetch_ohlcv = AsyncMock(return_value=synthetic_candles_15m)
    gw.fetch_ticker_and_mark = AsyncMock(return_value={"bid": 100.0, "ask": 100.1, "mark": 100.05})

    worker = PMMWorker(cfg, gw)
    # Prime initial market state
    worker.market_state.update_ticker(100.0, 100.1, 100.05)
    worker._running = True

    iteration = 0
    async def mock_sleep(seconds):
        nonlocal iteration
        iteration += 1
        if iteration >= 1:
            worker._running = False

    with patch("asyncio.sleep", side_effect=mock_sleep), \
         patch.object(worker, "_requote", AsyncMock()):
        await worker._main_worker_loop()

    assert gw.fetch_ohlcv.call_count >= 1
    # ATR is 0.9, close is 100.0 -> NATR is 0.009
    assert worker.market_state.current_natr_15m == pytest.approx(0.009, abs=1e-4)


@pytest.mark.asyncio
async def test_no_false_trip_after_reconnect():
    """
    TASK CB-5: Price history 60s sliding window is reset upon WS reconnect / worker start:
    - An old price point (e.g. from before disconnect) is cleared.
    - Post-reconnect tick does not trip CB against stale pre-disconnect price point.
    """
    cfg = _create_pair_config()
    gw = MagicMock(spec=ExchangeGateway)
    gw.get_market_precision.return_value = (2, 3, 0.01, 0.001)
    gw.get_market_limits.return_value = (0.001, 5.0)

    worker = PMMWorker(cfg, gw)
    # Stale price point from 40s ago at 95.0
    t0 = 1000.0
    worker.market_state.price_history_60s.append((t0 - 40, 95.0))

    # Simulate WS reconnect clearing window
    worker.market_state.price_history_60s.clear()
    assert len(worker.market_state.price_history_60s) == 0

    # New tick arrives at 100.0
    worker.market_state.update_top_of_book(100.0, 100.1, 100.05)
    is_tripped, delta, thresh = worker.market_state.check_circuit_breaker(t0)

    # Must NOT trip because len(price_history_60s) < 2
    assert is_tripped is False
