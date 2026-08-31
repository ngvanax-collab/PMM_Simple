"""FastAPI REST API Router for Bot Control, Configuration & Telemetry."""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.manager import bot_manager
from app.models.config import ExchangeCredentials, PairConfig
from app.persistence.db import db
from app.persistence.store import config_store, credential_store

router = APIRouter(prefix="/api", tags=["PMM Engine"])


class StatusResponse(BaseModel):
    is_connected: bool
    circuit_breaker_tripped: bool
    circuit_breaker_reason: str
    total_workers: int
    active_workers: int


@router.get("/status", response_model=StatusResponse)
async def get_system_status():
    active_cnt = sum(1 for w in bot_manager.workers.values() if w.is_running and not w.is_paused)
    return StatusResponse(
        is_connected=(bot_manager.gateway is not None and bot_manager.gateway._is_connected),
        circuit_breaker_tripped=bot_manager.circuit_breaker.is_tripped,
        circuit_breaker_reason=bot_manager.circuit_breaker.trip_reason,
        total_workers=len(bot_manager.workers),
        active_workers=active_cnt,
    )


@router.get("/pairs")
async def get_pairs():
    result = []
    for symbol, worker in bot_manager.workers.items():
        long_s = worker.tracker.long_pos
        short_s = worker.tracker.short_pos
        result.append({
            "symbol": symbol,
            "enabled": worker.config.enabled,
            "is_running": worker.is_running,
            "is_paused": worker.is_paused,
            "long": {
                "amount": long_s.amount,
                "entry_price": long_s.entry_price,
                "current_price": long_s.current_price,
                "notional": long_s.notional,
                "upnl": long_s.unrealized_pnl,
                "in_cooldown": worker.tracker.is_in_cooldown(long_s.position_side),
            },
            "short": {
                "amount": short_s.amount,
                "entry_price": short_s.entry_price,
                "current_price": short_s.current_price,
                "notional": short_s.notional,
                "upnl": short_s.unrealized_pnl,
                "in_cooldown": worker.tracker.is_in_cooldown(short_s.position_side),
            },
            "gross_exposure": worker.tracker.gross_exposure_usdt,
        })
    return result


@router.post("/pairs/{symbol:path}/start")
async def start_pair(symbol: str):
    success = await bot_manager.start_pair(symbol)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to start pair {symbol}")
    return {"status": "success", "symbol": symbol}


@router.post("/pairs/{symbol:path}/stop")
async def stop_pair(symbol: str):
    success = await bot_manager.stop_pair(symbol)
    return {"status": "success", "symbol": symbol, "stopped": success}


@router.delete("/pairs/{symbol:path}")
async def delete_pair(symbol: str):
    success = await bot_manager.delete_pair(symbol)
    if not success:
        raise HTTPException(status_code=404, detail=f"Pair {symbol} not found or failed to delete")
    return {"status": "success", "symbol": symbol, "deleted": True}


@router.post("/pairs/{symbol:path}/pause")
async def pause_pair(symbol: str):
    success = await bot_manager.pause_pair(symbol)
    return {"status": "success", "symbol": symbol, "paused": success}


@router.post("/pairs/{symbol:path}/resume")
async def resume_pair(symbol: str):
    success = await bot_manager.resume_pair(symbol)
    return {"status": "success", "symbol": symbol, "resumed": success}


@router.post("/credentials")
async def save_credentials(creds: ExchangeCredentials):
    await credential_store.save_credentials(creds)
    success = await bot_manager.start_gateway(creds)
    if not success:
        return {"status": "error", "message": "Saved, but failed to connect or verify Hedge Mode"}
    return {"status": "success", "message": "Connected & Hedge Mode Verified"}


@router.post("/kill-all")
async def execute_emergency_kill():
    report = await bot_manager.emergency_kill_all()
    return {"status": "success", "report": report}


@router.get("/fills")
async def get_recent_fills(limit: int = 50, symbol: Optional[str] = None):
    fills = await db.get_recent_fills(limit=limit, symbol=symbol)
    return fills


@router.get("/pnl")
async def get_pnl_summary():
    summary = await db.get_pnl_summary()
    return summary


# ── Rebalancer & Screener Endpoints ──

@router.get("/rebalancer/status")
async def get_rebalancer_status():
    from app.core.pair_rebalancer import rebalancer_service
    from app.core.screener import screener_engine

    active_slots = sum(
        1 for w in bot_manager.workers.values()
        if w.config.enabled and (not getattr(w, "is_draining", False) or not w.is_flat)
    )
    return {
        "enabled": rebalancer_service.config.enabled,
        "is_running": rebalancer_service.is_running,
        "is_scanning": screener_engine.is_scanning,
        "last_scan_time": screener_engine.last_scan_time,
        "last_rebalance_time": rebalancer_service.last_rebalance_time,
        "active_slots": active_slots,
        "max_active_pairs": rebalancer_service.config.max_active_pairs,
        "total_candidates": len(screener_engine.last_metrics),
        "config": rebalancer_service.config.model_dump(),
    }


@router.post("/rebalancer/scan")
async def trigger_rebalance_scan():
    from app.core.pair_rebalancer import rebalancer_service
    summary = await rebalancer_service.execute_rebalance_cycle(bot_manager)
    return {"status": "success", "summary": summary}


@router.get("/rebalancer/candidates")
async def get_rebalancer_candidates():
    from app.core.screener import screener_engine
    sorted_metrics = sorted(screener_engine.last_metrics.values(), key=lambda m: m.rank)
    return [m.to_dict() for m in sorted_metrics]


@router.get("/rebalancer/events")
async def get_rebalancer_events(limit: int = 50):
    from app.core.pair_rebalancer import rebalancer_service
    return rebalancer_service.get_recent_events(limit=limit)

