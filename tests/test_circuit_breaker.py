"""Unit Tests for Account and Per-Pair Circuit Breaker."""
import time
import pytest
from unittest.mock import AsyncMock, patch

from app.core.circuit_breaker import CircuitBreaker
from app.models.config import GlobalConfig, PairConfig
from app.persistence.db import db


@pytest.mark.asyncio
async def test_circuit_breaker_healthy_on_fresh_account():
    """Verify Circuit Breaker is healthy (not tripped) when account is empty or no trades occurred today."""
    cfg = GlobalConfig(account_daily_loss_limit_usdt=100.0, kill_margin_ratio=1.2)
    cb = CircuitBreaker(cfg)

    with patch("app.persistence.db.db.get_pnl_summary", new_callable=AsyncMock) as mock_pnl:
        mock_pnl.return_value = {"total_net_pnl": 0.0}

        # Empty account with 0 maintenance margin and no loss
        tripped, reason = await cb.check_account_health(
            total_account_balance=1000.0,
            total_maintenance_margin=0.0,
        )

        assert tripped is False
        assert reason == "OK"
        assert cb.is_tripped is False


@pytest.mark.asyncio
async def test_circuit_breaker_trips_on_excessive_daily_loss():
    """Verify Circuit Breaker trips when net daily loss strictly exceeds configured limit."""
    cfg = GlobalConfig(account_daily_loss_limit_usdt=50.0)
    cb = CircuitBreaker(cfg)

    with patch("app.persistence.db.db.get_pnl_summary", new_callable=AsyncMock) as mock_pnl:
        mock_pnl.return_value = {"total_realized_pnl": -55.0, "total_fee": 1.0, "total_net_pnl": -56.0}

        tripped, reason = await cb.check_account_health(total_account_balance=1000.0, total_maintenance_margin=0.0)
        assert tripped is True
        assert "Account daily loss exceeded" in reason
        assert cb.is_tripped is True


@pytest.mark.asyncio
async def test_circuit_breaker_trips_on_dangerous_margin_ratio():
    """Verify Circuit Breaker trips when margin ratio falls below emergency kill threshold."""
    cfg = GlobalConfig(kill_margin_ratio=1.2)
    cb = CircuitBreaker(cfg)

    with patch("app.persistence.db.db.get_pnl_summary", new_callable=AsyncMock) as mock_pnl:
        mock_pnl.return_value = {"total_net_pnl": 0.0}

        # Balance = 100, Maintenance Margin = 90 -> Ratio = 1.11 (< 1.2)
        tripped, reason = await cb.check_account_health(
            total_account_balance=100.0,
            total_maintenance_margin=90.0,
        )
        assert tripped is True
        assert "Emergency Margin Ratio breached" in reason
        assert cb.is_tripped is True


@pytest.mark.asyncio
async def test_circuit_breaker_manual_reset_clears_tripped_state():
    """Verify circuit_breaker.reset() clears tripped state and sets reset window."""
    cfg = GlobalConfig(account_daily_loss_limit_usdt=50.0)
    cb = CircuitBreaker(cfg)
    cb._tripped = True
    cb._trip_reason = "Test trip"

    assert cb.is_tripped is True
    cb.reset()

    assert cb.is_tripped is False
    assert cb.trip_reason == ""
    assert cb._reset_timestamp > 0


@pytest.mark.asyncio
async def test_check_pair_loss_healthy():
    """Verify check_pair_loss returns healthy when pair PnL is within daily limit."""
    global_cfg = GlobalConfig()
    cb = CircuitBreaker(global_cfg)
    pair_cfg = PairConfig(symbol="SOL/USDT:USDT", daily_loss_limit_usdt=30.0)

    with patch("app.persistence.db.db.get_pnl_summary", new_callable=AsyncMock) as mock_pnl:
        mock_pnl.return_value = {"total_realized_pnl": -10.0, "total_fee": 1.0, "total_net_pnl": -11.0}

        tripped, reason = await cb.check_pair_loss(pair_cfg)
        assert tripped is False
        assert reason == "OK"
        mock_pnl.assert_awaited_once()
        assert mock_pnl.call_args.kwargs["symbol"] == "SOL/USDT:USDT"


@pytest.mark.asyncio
async def test_check_pair_loss_trips_on_excessive_loss():
    """Verify check_pair_loss trips when pair net loss exceeds daily_loss_limit_usdt / max_daily_loss_usdt."""
    global_cfg = GlobalConfig()
    cb = CircuitBreaker(global_cfg)
    pair_cfg = PairConfig(symbol="ETH/USDT:USDT", daily_loss_limit_usdt=30.0)

    with patch("app.persistence.db.db.get_pnl_summary", new_callable=AsyncMock) as mock_pnl:
        mock_pnl.return_value = {"total_realized_pnl": -32.0, "total_fee": 1.0, "total_net_pnl": -33.0}

        tripped, reason = await cb.check_pair_loss(pair_cfg)
        assert tripped is True
        assert "Pair ETH/USDT:USDT daily loss limit exceeded: -33.00 USDT >= 30.00 USDT" in reason
        mock_pnl.assert_awaited_once()
        assert mock_pnl.call_args.kwargs["symbol"] == "ETH/USDT:USDT"
