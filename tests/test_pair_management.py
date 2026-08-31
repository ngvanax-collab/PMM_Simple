"""Unit tests for Pair Management (Add, Edit, Delete pair and API endpoint)."""
from unittest.mock import AsyncMock, MagicMock
from fastapi import FastAPI
import pytest
from httpx import AsyncClient, ASGITransport

from app.api.router import router as api_router
from app.core.manager import BotManager
from app.models.config import PairConfig
from app.persistence.store import ConfigStore


@pytest.fixture
def clean_config_store(tmp_path):
    cs = ConfigStore()
    cs.pairs_dir = tmp_path / "pairs"
    cs.pairs_dir.mkdir(parents=True, exist_ok=True)
    return cs


@pytest.mark.asyncio
async def test_bot_manager_delete_pair(clean_config_store, monkeypatch):
    """Verify delete_pair stops worker and unlinks config file from disk."""
    import app.core.manager as manager_mod
    monkeypatch.setattr(manager_mod, "config_store", clean_config_store)

    mgr = BotManager()
    mgr.gateway = MagicMock()

    cfg = PairConfig(symbol="AVAX/USDT:USDT", enabled=True)
    clean_config_store.save_pair_config(cfg)
    assert clean_config_store.load_pair_config("AVAX/USDT:USDT") is not None

    mock_worker = MagicMock()
    mock_worker.stop = AsyncMock()
    mock_worker.config = cfg
    mgr.workers["AVAX/USDT:USDT"] = mock_worker

    # Delete pair
    success = await mgr.delete_pair("AVAX/USDT:USDT")
    assert success is True
    assert "AVAX/USDT:USDT" not in mgr.workers
    mock_worker.stop.assert_called_once()
    assert clean_config_store.load_pair_config("AVAX/USDT:USDT") is None


@pytest.mark.asyncio
async def test_api_delete_pair(monkeypatch):
    """Verify DELETE /api/pairs/{symbol} endpoint."""
    from app.core.manager import bot_manager
    bot_manager.delete_pair = AsyncMock(return_value=True)

    app = FastAPI()
    app.include_router(api_router)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/api/pairs/SOL/USDT:USDT")
        assert resp.status_code == 200
        assert resp.json() == {"status": "success", "symbol": "SOL/USDT:USDT", "deleted": True}
