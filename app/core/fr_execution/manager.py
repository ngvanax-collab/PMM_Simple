"""Funding Rate Arbitrage Manager: Coordination, Background Polling & Telemetry Provider."""
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger

from app.config import CONFIG_DIR
from app.core.fr_execution.engine import FRExecutionEngine
from app.core.fr_execution.gateway import MultiExchangeGateway
from app.core.fr_execution.kill_switch import ThreeTierKillSwitch
from app.core.fr_execution.models import DualLegPosition, FRPolicy, FRRiskConfig, FRSummaryMetrics
from app.core.fr_execution.position_tracker import FRPositionTracker
from app.persistence.store import credential_store


class FRManager:
    """Singleton orchestrator for Funding Rate Arbitrage execution and UI telemetry."""

    def __init__(self):
        self.config_path = CONFIG_DIR / "fr_arbitrage_config.json"
        self.risk_config: FRRiskConfig = self._load_risk_config()
        self.tracker = FRPositionTracker()
        self.kill_switch = ThreeTierKillSwitch()
        self.gateway = MultiExchangeGateway()
        self.engine = FRExecutionEngine(self.gateway, self.tracker, self.kill_switch, self.risk_config)

        self._poll_task: Optional[asyncio.Task] = None
        self._reconcile_task: Optional[asyncio.Task] = None
        self._is_running = False

    def _load_risk_config(self) -> FRRiskConfig:
        """Load FR risk configuration from disk or defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return FRRiskConfig(**data)
            except Exception as e:
                logger.error(f"Failed to load fr_arbitrage_config.json: {e}")
        cfg = FRRiskConfig()
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(cfg.model_dump(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving default fr_arbitrage_config.json: {e}")
        return cfg

    def save_risk_config(self, config: FRRiskConfig) -> None:
        """Save FR risk configuration to disk."""
        self.risk_config = config
        if hasattr(self, "engine") and self.engine is not None:
            self.engine.risk_config = config
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config.model_dump(), f, indent=2)
            logger.info("FR Arbitrage configuration saved.")
        except Exception as e:
            logger.error(f"Error saving fr_arbitrage_config.json: {e}")

    async def initialize(self) -> bool:
        """Initialize exchange gateway, position tracker, and start background loops."""
        logger.info("Initializing Funding Rate Arbitrage Engine...")

        # Load credentials for Binance and Bybit
        binance_creds = await credential_store.load_credentials("binance")
        bybit_creds = await credential_store.load_credentials("bybit")

        self.gateway.creds["binance"] = binance_creds
        self.gateway.creds["bybit"] = bybit_creds

        if not binance_creds or not bybit_creds:
            logger.warning("Binance or Bybit credentials not fully configured in DB. FR Engine running in standby.")
        else:
            await self.gateway.initialize()

        self._is_running = True
        self._poll_task = asyncio.create_task(self._policy_polling_loop())
        self._reconcile_task = asyncio.create_task(self._reconciliation_loop())

        logger.info("Funding Rate Arbitrage Manager started successfully.")
        return True

    async def _policy_polling_loop(self) -> None:
        """Background loop to periodically poll Decision Layer for policies."""
        while self._is_running:
            try:
                if self.risk_config.auto_execution_enabled and not self.kill_switch.is_global_tripped:
                    await self.engine.fetch_and_execute_policies()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in policy polling loop: {e}")

            await asyncio.sleep(self.risk_config.poll_interval_sec)

    async def _reconciliation_loop(self) -> None:
        """Background loop to reconcile open positions with REST every 60 seconds."""
        while self._is_running:
            try:
                if self.gateway.is_connected("binance") or self.gateway.is_connected("bybit"):
                    binance_pos = await self.gateway.fetch_positions("binance")
                    bybit_pos = await self.gateway.fetch_positions("bybit")
                    self.tracker.reconcile_with_exchange_positions(binance_pos, bybit_pos)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in FR position reconciliation loop: {e}")

            await asyncio.sleep(60.0)

    async def manual_close_pair(self, symbol: str) -> Dict[str, Any]:
        """Manually exit and flat an active dual-leg position."""
        sym = symbol.strip().upper()
        dual_pos = self.tracker.get_position(sym)
        if not dual_pos:
            return {"status": "POSITION_NOT_FOUND", "symbol": sym}

        from app.core.fr_execution.models import FRAction, FRPolicy
        dummy_policy = FRPolicy(
            policy_id=f"manual_exit_{int(time.time())}",
            symbol=sym,
            exchange_long=dual_pos.long_leg.exchange,
            exchange_short=dual_pos.short_leg.exchange,
            action=FRAction.EXIT,
            target_notional_usdt=0.0,
            expected_net_funding_bps=0.0,
            expected_fee_bps=0.0,
            expected_slippage_bps=0.0,
            expected_net_edge_bps=0.0,
        )
        return await self.engine._handle_exit(dummy_policy, dual_pos)

    def toggle_pause_symbol(self, symbol: str) -> bool:
        """Toggle pause state for a symbol."""
        sym = symbol.strip().upper()
        if self.kill_switch.is_symbol_paused(sym):
            self.kill_switch.resume_symbol(sym)
            pos = self.tracker.get_position(sym)
            if pos:
                pos.is_paused = False
            return False  # Now active
        else:
            self.kill_switch.pause_symbol(sym, "User UI pause")
            pos = self.tracker.get_position(sym)
            if pos:
                pos.is_paused = True
            return True  # Now paused

    async def trigger_emergency_kill_all(self) -> Dict[str, Any]:
        """Trigger system-wide 6-phase Emergency Kill-All."""
        return await self.kill_switch.emergency_kill_all(self.gateway, self.tracker)

    async def get_summary_metrics(self) -> FRSummaryMetrics:
        """Fetch free margins and compute current metrics."""
        binance_free = await self.gateway.get_free_margin("binance") if self.gateway.is_connected("binance") else 0.0
        bybit_free = await self.gateway.get_free_margin("bybit") if self.gateway.is_connected("bybit") else 0.0
        return self.tracker.get_summary_metrics(binance_free, bybit_free)

    def get_recent_policies(self) -> List[FRPolicy]:
        """Return cached policies from Decision Layer."""
        return self.engine.recent_policies

    def get_active_positions(self) -> List[DualLegPosition]:
        """Return active dual positions."""
        return self.tracker.get_active_positions()

    async def shutdown(self) -> None:
        """Graceful shutdown of loops and gateways."""
        self._is_running = False
        if self._poll_task:
            self._poll_task.cancel()
        if self._reconcile_task:
            self._reconcile_task.cancel()
        await self.gateway.close()
        logger.info("Funding Rate Arbitrage Manager shut down.")


# Global instance
fr_manager = FRManager()
