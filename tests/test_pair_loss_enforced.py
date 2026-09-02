"""Tests for Daily Loss Enforcement & PnL Restoration (TASK H-1)."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.core.circuit_breaker import CircuitBreaker, utc_day_start
from app.core.manager import BotManager
from app.core.worker import PMMWorker
from app.models.config import GlobalConfig, PairConfig
from app.models.state import PositionSide, SidePositionState


@pytest.fixture(autouse=True)
def mock_db_ops():
    with patch("app.core.worker.db.save_fill", new_callable=AsyncMock), \
         patch("app.core.worker.db.record_pnl", new_callable=AsyncMock), \
         patch("app.core.worker.db.save_order", new_callable=AsyncMock), \
         patch("app.core.executor.db.save_fill", new_callable=AsyncMock), \
         patch("app.core.executor.db.record_pnl", new_callable=AsyncMock):
        yield


def test_daily_window_is_utc():
    """Verify utc_day_start computes exact 00:00:00 UTC epoch boundaries."""
    t_now = 1717171717.0
    day_start = utc_day_start(t_now)
    assert day_start % 86400 == 0
    assert 0 <= (t_now - day_start) < 86400


@pytest.mark.asyncio
async def test_session_pnl_restored_from_db_on_restart():
    """Verify worker.start() restores daily realized PnL from DB summary since UTC 00:00."""
    cfg = PairConfig(symbol="SOL/USDT:USDT", leverage=5)
    mock_gw = MagicMock()
    mock_gw.get_market_precision.return_value = (2, 3, 0.01, 0.001)
    mock_gw.get_market_limits.return_value = (0.0, 5.0)
    mock_gw.setup_symbol = AsyncMock()
    mock_gw.fetch_positions = AsyncMock(return_value=[])
    mock_gw.fetch_positions_hedge = AsyncMock(
        return_value=(
            SidePositionState(symbol="SOL/USDT:USDT", position_side=PositionSide.LONG, amount=0.0, entry_price=0.0),
            SidePositionState(symbol="SOL/USDT:USDT", position_side=PositionSide.SHORT, amount=0.0, entry_price=0.0),
        )
    )
    mock_gw.fetch_ticker_and_mark = AsyncMock(return_value={"bid": 100.0, "ask": 100.1, "mark": 100.05})
    mock_gw.watch_public_ticker = AsyncMock()

    worker = PMMWorker(cfg, mock_gw)

    fake_summary = {"total_net_pnl": -12.50, "total_realized_pnl": -12.50}
    with patch("app.core.worker.db.get_pnl_summary", new_callable=AsyncMock, return_value=fake_summary):
        await worker.start()

    assert worker.session_realized_pnl == pytest.approx(-12.50, abs=1e-4)
    await worker.stop()


@pytest.mark.asyncio
async def test_health_loop_kills_worker_on_daily_loss():
    """
    Verify _health_monitor_loop evaluates per-pair daily loss limits:
    - Worker A (SOL) loses -35 USDT (limit 30 USDT) -> isolated kill triggered.
    - Worker B (DOGE) has +5 USDT -> remains running.
    """
    g_cfg = GlobalConfig()
    sol_cfg = PairConfig(symbol="SOL/USDT:USDT", max_daily_loss_usdt=30.0)
    doge_cfg = PairConfig(symbol="DOGE/USDT:USDT", max_daily_loss_usdt=30.0)

    mgr = BotManager()
    mgr.circuit_breaker = CircuitBreaker(g_cfg)

    mock_gw = MagicMock()
    mock_gw.get_market_precision.return_value = (2, 3, 0.01, 0.001)
    mock_gw.get_market_limits.return_value = (0.0, 5.0)

    worker_sol = PMMWorker(sol_cfg, mock_gw)
    worker_sol._running = True
    async def kill_sol(reason):
        worker_sol.is_locked_killed = True
    worker_sol._trigger_isolated_kill = AsyncMock(side_effect=kill_sol)

    worker_doge = PMMWorker(doge_cfg, mock_gw)
    worker_doge._running = True
    worker_doge._trigger_isolated_kill = AsyncMock()

    mgr.workers = {
        "SOL/USDT:USDT": worker_sol,
        "DOGE/USDT:USDT": worker_doge,
    }
    mgr._is_running = True

    async def fake_get_pnl_summary(symbol=None, since_timestamp=None):
        if symbol == "SOL/USDT:USDT":
            return {"total_net_pnl": -35.0, "total_realized_pnl": -35.0}
        elif symbol == "DOGE/USDT:USDT":
            return {"total_net_pnl": 5.0, "total_realized_pnl": 5.0}
        return {"total_net_pnl": -30.0}

    with patch("app.core.circuit_breaker.db.get_pnl_summary", side_effect=fake_get_pnl_summary), \
         patch("asyncio.sleep", side_effect=asyncio.CancelledError()):
        try:
            await mgr._health_monitor_loop()
        except asyncio.CancelledError:
            pass

    # SOL must have received isolated kill
    worker_sol._trigger_isolated_kill.assert_awaited_once()
    # DOGE must NOT be killed
    worker_doge._trigger_isolated_kill.assert_not_awaited()
