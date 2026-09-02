"""Bot Manager: Multi-Pair Coordination, Health Guard & 6-Phase Emergency Kill-All."""
import asyncio
import time
from typing import Any, Dict, List, Optional
from loguru import logger

from app.core.circuit_breaker import CircuitBreaker
from app.core.gateway import ExchangeGateway
from app.core.pair_rebalancer import rebalancer_service
from app.core.ratelimit import rate_limiter
from app.core.worker import PMMWorker
from app.models.config import ExchangeCredentials, GlobalConfig, PairConfig
from app.models.state import FillRecord, OrderPurpose, OrderRecord, OrderSide, OrderType, PositionSide
from app.persistence.db import db
from app.persistence.store import config_store, credential_store


class BotManager:
    """Orchestrates multi-pair workers, system health checks, and 6-phase emergency kill-all."""

    def __init__(self):
        self.global_config: GlobalConfig = config_store.load_global_config()
        rate_limiter.configure(
            orders_per_min=self.global_config.order_budget_per_min,
            weight_per_min=self.global_config.weight_budget_per_min,
        )
        self.circuit_breaker = CircuitBreaker(self.global_config)
        self.gateway: Optional[ExchangeGateway] = None
        self.workers: Dict[str, PMMWorker] = {}  # symbol -> PMMWorker
        self._health_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._kill_all_lock = asyncio.Lock()

    async def initialize(self) -> bool:
        """Initialize DB, load credentials, setup gateway and workers."""
        await db.connect()

        # Load credentials
        creds = await credential_store.load_credentials()
        if not creds or not creds.api_key or not creds.api_secret:
            logger.warning("No exchange credentials found. BotManager waiting for credentials setup in UI.")
            return False

        return await self.start_gateway(creds)

    async def start_gateway(self, creds: ExchangeCredentials) -> bool:
        """Initialize Exchange Gateway and load pair workers."""
        if self.gateway:
            await self.gateway.close()

        self.gateway = ExchangeGateway(creds)
        success = await self.gateway.initialize()
        if not success:
            logger.error("Failed to connect to exchange or verify Hedge Mode.")
            return False

        # Set Private WS Callbacks
        self.gateway.on_order_update = self._handle_order_update
        self.gateway.on_fill_update = self._handle_fill_update
        self.gateway.on_position_update = self._handle_position_update

        await self.gateway.start_private_watchers()

        # Load and instantiate workers for all saved pairs
        pair_configs = config_store.load_all_pair_configs()
        self.workers.clear()
        for symbol, p_cfg in pair_configs.items():
            worker = PMMWorker(p_cfg, self.gateway)
            self.workers[symbol] = worker

        self._is_running = True
        self._health_task = asyncio.create_task(self._health_monitor_loop())
        logger.info(f"BotManager initialized with {len(self.workers)} workers ready.")
        await self.start_all()
        # Start dynamic rebalancer loop
        await rebalancer_service.start_background_loop(self)
        return True

    def _handle_order_update(self, order: OrderRecord) -> None:
        """Dispatch order update to corresponding pair worker with error isolation."""
        if order.symbol in self.workers:
            async def _safe_order_cb():
                try:
                    await self.workers[order.symbol].on_order_update(order)
                except Exception as e:
                    logger.error(f"[{order.symbol}] Exception in on_order_update: {e}")
            asyncio.create_task(_safe_order_cb())

    def _handle_fill_update(self, fill: FillRecord) -> None:
        """Dispatch fill update to corresponding pair worker with error isolation."""
        if fill.symbol in self.workers:
            async def _safe_fill_cb():
                try:
                    await self.workers[fill.symbol].on_fill(fill)
                except Exception as e:
                    logger.error(f"[{fill.symbol}] Exception in on_fill: {e}")
            asyncio.create_task(_safe_fill_cb())

    def _handle_position_update(self, symbol: str, pos_side: PositionSide, amount: float, entry_p: float, upnl: float) -> None:
        """Dispatch position update with error isolation and async lock safety."""
        if symbol in self.workers:
            async def _safe_pos_cb():
                try:
                    worker = self.workers[symbol]
                    async with worker._lock:
                        state = worker.tracker.get_state(pos_side)
                        state.amount = amount
                        state.entry_price = entry_p
                        state.unrealized_pnl = upnl
                        state.last_fill_time = max(state.last_fill_time, time.time())
                except Exception as e:
                    logger.error(f"[{symbol}] Exception in _handle_position_update: {e}")
            asyncio.create_task(_safe_pos_cb())

    # ── Worker Management ──

    async def start_pair(self, symbol: str) -> bool:
        """Start worker for a single pair."""
        if self.circuit_breaker.is_tripped:
            logger.error(
                f"Cannot start {symbol}: Circuit breaker is TRIPPED! Reason: {self.circuit_breaker.trip_reason}. Use 'RESET CIRCUIT BREAKER' in UI to unblock."
            )
            return False

        if not self.gateway or not self.gateway._is_connected:
            logger.error(
                f"Cannot start {symbol}: Gateway is not connected. Please verify API credentials and Hedge Mode in API Settings."
            )
            return False

        worker = self.workers.get(symbol)
        if not worker:
            p_cfg = config_store.load_pair_config(symbol)
            if not p_cfg:
                return False
            worker = PMMWorker(p_cfg, self.gateway)
            self.workers[symbol] = worker

        worker.config.enabled = True
        config_store.save_pair_config(worker.config)
        await worker.start()
        return True

    def get_active_worker_count(self) -> int:
        """Return count of actively running workers that are not paused and not draining."""
        return sum(1 for w in self.workers.values() if w.is_running and not w.is_paused and not getattr(w, "is_draining", False))

    async def register_dynamic_pair(self, pair_config: PairConfig) -> bool:
        """Dynamically register, save and start a new pair worker."""
        symbol = pair_config.symbol
        config_store.save_pair_config(pair_config)

        if not self.gateway or not self.gateway._is_connected:
            logger.warning(f"[{symbol}] Gateway not connected. Pair config saved, worker not started.")
            return False

        worker = self.workers.get(symbol)
        if not worker:
            worker = PMMWorker(pair_config, self.gateway)
            self.workers[symbol] = worker
        else:
            worker.config = pair_config
            worker.set_drain_mode(False)

        if pair_config.enabled:
            await worker.start()
            logger.info(f"[{symbol}] Dynamically registered and started worker.")
            return True
        return True

    async def drain_and_retire_pair(self, symbol: str) -> None:
        """Initiate Graceful Drain for a pair worker. Stop if already flat."""
        worker = self.workers.get(symbol)
        if not worker:
            return

        worker.set_drain_mode(True)
        if worker.is_flat:
            logger.info(f"[{symbol}] Worker is already flat during drain request. Stopping worker.")
            await self.stop_pair(symbol)

    async def stop_pair(self, symbol: str) -> bool:
        """Stop worker for a single pair."""
        worker = self.workers.get(symbol)
        if worker:
            worker.config.enabled = False
            config_store.save_pair_config(worker.config)
            await worker.stop()
            return True
        return False

    async def delete_pair(self, symbol: str) -> bool:
        """Completely stop, remove worker and delete pair configuration from disk."""
        worker = self.workers.pop(symbol, None)
        if worker:
            try:
                await worker.stop()
            except Exception as e:
                logger.error(f"[{symbol}] Error stopping worker during deletion: {e}")

        deleted = config_store.delete_pair_config(symbol)
        logger.info(f"[{symbol}] Pair deleted from bot and disk (success={deleted}).")
        return deleted

    async def pause_pair(self, symbol: str) -> bool:
        """Pause worker for a single pair."""
        worker = self.workers.get(symbol)
        if worker:
            worker.pause()
            return True
        return False

    async def resume_pair(self, symbol: str) -> bool:
        """Resume worker for a single pair."""
        worker = self.workers.get(symbol)
        if worker:
            worker.resume()
            return True
        return False

    async def unlock_pair(self, symbol: str) -> bool:
        """Unlock and restore worker after isolated kill switch."""
        worker = self.workers.get(symbol)
        if worker:
            await worker.unlock()
            worker.config.enabled = True
            config_store.save_pair_config(worker.config)
            await worker.start()
            return True
        return False

    async def start_all(self) -> None:
        """Start all configured and enabled pairs."""
        if self.circuit_breaker.is_tripped:
            logger.error("Cannot start all pairs: Circuit Breaker is TRIPPED!")
            return

        for symbol, worker in list(self.workers.items()):
            if worker.config.enabled:
                try:
                    await worker.start()
                except Exception as e:
                    logger.error(f"[{symbol}] Failed to start worker: {e}")

    async def stop_all(self) -> None:
        """Stop all workers."""
        for worker in self.workers.values():
            await worker.stop()

    # ── 6-Phase Emergency Kill-All ──

    async def emergency_kill_all(self) -> Dict[str, Any]:
        """
        Execute 6-Phase Emergency Kill-All Procedure (Hedge Mode Native):
        1. Stop all workers quoting loops immediately.
        2. Cancel ALL open orders across all configured symbols on exchange.
        3. Fetch REAL exchange positions with positionAmt != 0.
        4. Place MARKET exit orders with correct positionSide (No reduceOnly).
        5. Re-fetch positions to confirm 100% position amount = 0; close residual if any.
        6. Generate comprehensive audit report.
        """
        async with self._kill_all_lock:
            logger.critical("================ EMERGENCY KILL-ALL INITIATED ================")
            report: Dict[str, Any] = {
                "timestamp": time.time(),
                "phase1_stopped_workers": 0,
                "phase2_cancelled_orders": 0,
                "phase3_positions_found": [],
                "phase4_market_exits_placed": [],
                "phase5_confirmation": {},
                "status": "IN_PROGRESS",
            }

            # ── Phase 1: Stop all workers ──
            logger.critical("Phase 1: Halting all active worker quoting loops...")
            stopped_count = 0
            for symbol, worker in self.workers.items():
                await worker.stop()
                stopped_count += 1
            report["phase1_stopped_workers"] = stopped_count

            if not self.gateway or not self.gateway._is_connected:
                report["status"] = "ERROR_NO_GATEWAY"
                logger.error("Emergency Kill-All: Gateway not connected!")
                return report

            # ── Phase 2: Cancel ALL open orders ──
            logger.critical("Phase 2: Cancelling ALL open orders across symbols on exchange...")
            for symbol in list(self.workers.keys()):
                await self.gateway.cancel_all_symbol_orders(symbol)
                report["phase2_cancelled_orders"] += 1

            # ── Phase 3: Fetch REAL exchange positions ──
            logger.critical("Phase 3: Fetching real exchange positions for both LONG and SHORT sides...")
            open_positions = []
            for symbol in list(self.workers.keys()):
                long_p, short_p = await self.gateway.fetch_positions_hedge(symbol)
                if long_p and long_p.amount > 1e-6:
                    open_positions.append((symbol, PositionSide.LONG, long_p.amount))
                if short_p and short_p.amount > 1e-6:
                    open_positions.append((symbol, PositionSide.SHORT, short_p.amount))

            report["phase3_positions_found"] = [
                {"symbol": s, "side": ps.value, "amount": amt} for s, ps, amt in open_positions
            ]
            logger.critical(f"Found {len(open_positions)} open positions to close.")

            # ── Phase 4: Place MARKET Exit Orders (Hedge Mode Native) ──
            logger.critical("Phase 4: Placing MARKET exit orders with correct positionSide (No reduceOnly)...")
            for symbol, pos_side, amount in open_positions:
                exit_trade_side = OrderSide.SELL if pos_side == PositionSide.LONG else OrderSide.BUY
                logger.critical(
                    f"Executing Kill-All Market Close: {symbol} {exit_trade_side.value} {pos_side.value} qty={amount:.4f}"
                )
                resp = await self.gateway.create_exit_order(
                    symbol=symbol,
                    side=exit_trade_side,
                    position_side=pos_side,
                    order_type=OrderType.MARKET,
                    amount=amount,
                    client_order_id=f"kill_{pos_side.value.lower()}_{int(time.time()*1000)}",
                    purpose=OrderPurpose.KILL_ALL_EXIT,
                )
                report["phase4_market_exits_placed"].append({
                    "symbol": symbol,
                    "side": pos_side.value,
                    "amount": amount,
                    "success": resp is not None,
                })

            # Wait 500ms for fills to settle
            await asyncio.sleep(0.5)

            # ── Phase 5: Re-fetch positions to confirm zero ──
            logger.critical("Phase 5: Re-fetching positions to confirm 100% flat (amount = 0)...")
            residual_positions = []
            for symbol in list(self.workers.keys()):
                long_p, short_p = await self.gateway.fetch_positions_hedge(symbol)
                if long_p and long_p.amount > 1e-6:
                    residual_positions.append((symbol, PositionSide.LONG, long_p.amount))
                if short_p and short_p.amount > 1e-6:
                    residual_positions.append((symbol, PositionSide.SHORT, short_p.amount))

            if residual_positions:
                logger.warning(f"Residual positions detected ({len(residual_positions)}). Executing second-pass market close...")
                for symbol, pos_side, amount in residual_positions:
                    exit_trade_side = OrderSide.SELL if pos_side == PositionSide.LONG else OrderSide.BUY
                    await self.gateway.create_exit_order(
                        symbol=symbol,
                        side=exit_trade_side,
                        position_side=pos_side,
                        order_type=OrderType.MARKET,
                        amount=amount,
                        purpose=OrderPurpose.KILL_ALL_EXIT,
                    )
                # Second confirmation
                await asyncio.sleep(0.5)

            # Final check
            final_residuals = 0
            for symbol in list(self.workers.keys()):
                long_p, short_p = await self.gateway.fetch_positions_hedge(symbol)
                if (long_p and long_p.amount > 1e-6) or (short_p and short_p.amount > 1e-6):
                    final_residuals += 1

            report["phase5_confirmation"] = {
                "all_flat": (final_residuals == 0),
                "unclosed_count": final_residuals,
            }
            report["status"] = "COMPLETED" if final_residuals == 0 else "PARTIAL_WARNING"

            # ── Phase 6: Report and Alert ──
            logger.critical(f"Emergency Kill-All Complete! Final Status: {report['status']}")
            return report

    # ── Background Health Loop ──

    async def _health_monitor_loop(self) -> None:
        """Periodic background task to check account circuit breaker and per-pair loss limits."""
        while self._is_running:
            try:
                balance = None
                maint_margin = None
                if self.gateway and getattr(self.gateway, "_is_connected", False) and hasattr(self.gateway, "_exchange") and self.gateway._exchange:
                    try:
                        await rate_limiter.acquire_weight(2)
                        raw = await self.gateway._exchange.fetch_balance()
                        info = raw.get("info", {}) if isinstance(raw, dict) else {}
                        raw_bal = info.get("totalWalletBalance") or (raw.get("USDT", {}).get("total") if isinstance(raw.get("USDT"), dict) else 0.0) or 0.0
                        raw_maint = info.get("totalMaintMargin") or 0.0
                        balance = float(raw_bal) if float(raw_bal or 0.0) > 0 else None
                        maint_margin = float(raw_maint) if float(raw_maint or 0.0) > 0 else (0.0 if balance is not None else None)
                    except Exception as e:
                        logger.warning(f"Health monitor balance fetch failed: {e}")

                tripped, reason = await self.circuit_breaker.check_account_health(
                    total_account_balance=balance,
                    total_maintenance_margin=maint_margin,
                )
                if tripped:
                    logger.critical(f"Circuit Breaker activated! Executing Emergency Kill-All: {reason}")
                    await self.emergency_kill_all()
                    break

                # ── Check per-worker pair daily loss (TASK H-1) ──
                for worker in list(self.workers.values()):
                    if worker.is_running and not worker.is_locked_killed:
                        try:
                            pair_tripped, pair_reason = await self.circuit_breaker.check_pair_loss(worker.config)
                            if pair_tripped:
                                logger.critical(f"[{worker.symbol}] Isolated Pair Loss Circuit Breaker tripped: {pair_reason}")
                                await worker._trigger_isolated_kill(pair_reason)
                        except Exception as pe:
                            logger.error(f"[{worker.symbol}] Error checking pair loss circuit breaker: {pe}")

                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor exception: {e}")
                await asyncio.sleep(5.0)

    async def shutdown(self) -> None:
        """Graceful system shutdown."""
        self._is_running = False
        await rebalancer_service.stop_background_loop()
        if self._health_task:
            self._health_task.cancel()
        await self.stop_all()
        if self.gateway:
            await self.gateway.close()
        await db.close()
        logger.info("BotManager shutdown complete.")


# Global singleton instance
bot_manager = BotManager()
