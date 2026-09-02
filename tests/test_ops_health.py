"""Tests for Operations, Live Health Metrics, and REST Weight Budget (Phase 4)."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.core.circuit_breaker import CircuitBreaker
from app.core.manager import BotManager
from app.core.ratelimit import rate_limiter
from app.core.worker import PMMWorker
from app.models.config import GlobalConfig, PairConfig
from app.models.state import PositionSide


@pytest.fixture
def global_config():
    return GlobalConfig(
        account_daily_loss_limit_usdt=100.0,
        min_margin_ratio=1.5,
        kill_margin_ratio=1.2,
        order_budget_per_min=900,
        weight_budget_per_min=2400,
    )


@pytest.fixture
def pair_config():
    return PairConfig(
        symbol="SOL/USDT:USDT",
        order_amount_usdt=30.0,
        bid_spread=0.0035,
        ask_spread=0.0035,
        order_levels=3,
        daily_loss_limit_usdt=25.0,
    )


@pytest.mark.asyncio
async def test_margin_ratio_trips_with_real_metrics(global_config):
    """
    TASK H-5: Health monitor uses live account balance & maintenance margin:
    - Mock exchange fetch_balance returning totalWalletBalance=100.0, totalMaintMargin=90.0
    - Margin ratio is 100/90 = 1.11 < 1.20 kill threshold
    - Circuit breaker trips and BotManager executes emergency_kill_all.
    """
    manager = BotManager()
    manager.global_config = global_config
    manager.circuit_breaker = CircuitBreaker(global_config)
    manager._is_running = True

    mock_gateway = MagicMock()
    mock_gateway._is_connected = True
    mock_exchange = AsyncMock()
    mock_exchange.fetch_balance = AsyncMock(
        return_value={
            "info": {
                "totalWalletBalance": "100.0",
                "totalMaintMargin": "90.0",
            },
            "USDT": {"total": 100.0},
        }
    )
    mock_gateway._exchange = mock_exchange
    manager.gateway = mock_gateway

    manager.emergency_kill_all = AsyncMock(return_value={"status": "COMPLETED"})

    # Run one cycle of _health_monitor_loop by patching asyncio.sleep to break loop
    async def mock_sleep(seconds):
        manager._is_running = False

    with patch("asyncio.sleep", side_effect=mock_sleep):
        await manager._health_monitor_loop()

    assert manager.circuit_breaker.is_tripped is True
    assert "Emergency Margin Ratio breached" in manager.circuit_breaker.trip_reason
    manager.emergency_kill_all.assert_called_once()


@pytest.mark.asyncio
async def test_rest_ticker_skipped_when_ws_fresh(pair_config):
    """
    TASK H-8: Worker only polls REST ticker when WS stream is stale (>5s):
    - When market_state.last_update_time is fresh (<5s), REST fetch_ticker_and_mark is skipped.
    - When market_state.last_update_time is stale (>5s), REST fetch_ticker_and_mark is invoked.
    """
    mock_gateway = AsyncMock()
    mock_gateway.get_market_precision = MagicMock(return_value=(2, 3, 0.01, 0.001))
    mock_gateway.get_market_limits = MagicMock(return_value=(0.001, 5.0))
    mock_gateway.fetch_ticker_and_mark = AsyncMock(return_value={"bid": 100.0, "ask": 100.2, "mark": 100.1})

    worker = PMMWorker(pair_config, mock_gateway)

    # 1. Fresh WS ticker state: set best_bid/ask and last_update_time to now
    worker.market_state.update_ticker(bid=100.0, ask=100.2, mark=100.1)
    assert time.time() - worker.market_state.last_update_time < 2.0

    worker._running = True
    iteration_count = 0

    async def mock_sleep_fresh(seconds):
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count >= 5:
            worker._running = False

    with patch("asyncio.sleep", side_effect=mock_sleep_fresh), \
         patch.object(worker, "_requote", AsyncMock()):
        await worker._main_worker_loop()

    # When WS stream is fresh, REST fetch_ticker_and_mark must NOT be called
    assert mock_gateway.fetch_ticker_and_mark.call_count == 0

    # 2. Stale WS ticker state (>5.0s ago)
    worker.market_state.last_update_time = time.time() - 10.0
    worker._running = True
    iteration_count = 0

    async def mock_sleep_stale(seconds):
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count >= 1:
            worker._running = False

    with patch("asyncio.sleep", side_effect=mock_sleep_stale), \
         patch.object(worker, "_requote", AsyncMock()):
        await worker._main_worker_loop()

    # When WS is stale, REST fetch_ticker_and_mark is called
    assert mock_gateway.fetch_ticker_and_mark.call_count >= 1


def test_rate_limiter_configure():
    """TASK H-8: GlobalRateLimiter configure method dynamically updates token bucket capacity."""
    rate_limiter.configure(orders_per_min=1200, weight_per_min=2400)
    assert rate_limiter.orders_bucket.capacity == 1200.0
    assert rate_limiter.orders_bucket.refill_rate == 20.0
    assert rate_limiter.weight_bucket.capacity == 2400.0
    assert rate_limiter.weight_bucket.refill_rate == 40.0
