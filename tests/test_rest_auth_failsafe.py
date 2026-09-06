"""Unit Tests for REST Health Circuit Breaker (TASK RO-2)."""
import time
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.core.gateway import ExchangeGateway
from app.core.worker import PMMWorker
from app.models.config import ExchangeCredentials, PairConfig
from app.models.state import OrderSide, PositionSide


@pytest.fixture
def mock_gateway():
    creds = ExchangeCredentials(
        exchange="binance",
        api_key="test_key",
        api_secret="test_secret",
        testnet=False,
    )
    gw = ExchangeGateway(creds)
    gw._exchange = MagicMock()
    return gw


@pytest.fixture
def mock_worker_with_gw(mock_gateway):
    cfg = PairConfig(
        symbol="DOT/USDT:USDT",
        requote_threshold_pct=0.002,
        order_refresh_time=30,
        order_levels=1,
    )
    worker = PMMWorker(cfg, mock_gateway)
    worker.market_state.best_bid = 0.88
    worker.market_state.best_ask = 0.89
    worker.market_state.smoothed_mid = 0.885
    worker._running = True
    worker._paused = False
    return worker


@pytest.mark.asyncio
async def test_auth_error_threshold_pauses_quoting(mock_gateway, mock_worker_with_gw):
    """3 consecutive auth errors (-2015) must trip is_auth_healthy and pause quoting."""
    assert mock_gateway.is_auth_healthy is True

    mock_gateway._exchange.cancel_order = AsyncMock(
        side_effect=Exception('binance {"code":-2015,"msg":"Invalid API-key, IP, or permissions for action"}')
    )

    # 1st failure
    await mock_gateway.cancel_order("DOT/USDT:USDT", "ord_1")
    assert mock_gateway._consecutive_auth_errors == 1
    assert mock_gateway.is_auth_healthy is True

    # 2nd failure
    await mock_gateway.cancel_order("DOT/USDT:USDT", "ord_2")
    assert mock_gateway._consecutive_auth_errors == 2
    assert mock_gateway.is_auth_healthy is True

    # 3rd failure -> Trip!
    await mock_gateway.cancel_order("DOT/USDT:USDT", "ord_3")
    assert mock_gateway._consecutive_auth_errors == 3
    assert mock_gateway.is_auth_healthy is False

    # Worker quoting check must immediately be blocked
    should_requote, reason = mock_worker_with_gw._check_should_requote()
    assert should_requote is False
    assert "REST auth unhealthy" in reason


@pytest.mark.asyncio
async def test_auth_recovery_resumes_quoting(mock_gateway, mock_worker_with_gw):
    """A successful REST call resets the error counter and restores is_auth_healthy."""
    # Force trip
    mock_gateway._consecutive_auth_errors = 3
    mock_gateway._auth_error_tripped_at = time.time()
    assert mock_gateway.is_auth_healthy is False

    # Mock a successful REST call (e.g. cancel_order succeeds)
    mock_gateway._exchange.cancel_order = AsyncMock(return_value={"id": "ord_ok"})
    ok = await mock_gateway.cancel_order("DOT/USDT:USDT", "ord_ok")
    assert ok is True

    # Counter and tripped_at must be reset
    assert mock_gateway._consecutive_auth_errors == 0
    assert mock_gateway.is_auth_healthy is True

    # Worker can quote again if normal conditions met
    should_requote, _ = mock_worker_with_gw._check_should_requote()
    assert should_requote is True


@pytest.mark.asyncio
async def test_other_errors_do_not_trip_auth_circuit_breaker(mock_gateway):
    """Non-auth errors (e.g. margin, timeout) must not increment auth error counter."""
    mock_gateway._exchange.cancel_order = AsyncMock(
        side_effect=Exception('binance {"code":-2019,"msg":"Margin is insufficient"}')
    )

    for _ in range(5):
        await mock_gateway.cancel_order("DOT/USDT:USDT", "ord_test")

    assert mock_gateway._consecutive_auth_errors == 0
    assert mock_gateway.is_auth_healthy is True
