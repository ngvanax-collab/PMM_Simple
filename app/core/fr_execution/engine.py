"""Funding Rate Arbitrage Execution Engine (Native Dual-Exchange Hedge Mode & Legging Risk Protection)."""
import asyncio
import json
import time
from typing import Any, Dict, List, Optional
import httpx
from loguru import logger

from app.core.fr_execution.gateway import MultiExchangeGateway
from app.core.fr_execution.kill_switch import ThreeTierKillSwitch
from app.core.fr_execution.models import DualLegPosition, FRAction, FRPolicy, FRRiskConfig
from app.core.fr_execution.position_tracker import FRPositionTracker


class FRExecutionEngine:
    """
    Core Async Execution Engine for Funding Rate Arbitrage.
    Complies with SIGNAL_CONTRACT_V2:
    - Pure consumer of Decision Layer policies (OPEN, HOLD, REDUCE, EXIT, PAUSE).
    - Dual-Exchange Hedge Invariant: Parallel 2-leg execution without reduceOnly.
    - Legging Risk Protection: Automatic 5s emergency rollback on partial fill/failure.
    """

    def __init__(
        self,
        gateway: MultiExchangeGateway,
        tracker: FRPositionTracker,
        kill_switch: ThreeTierKillSwitch,
        risk_config: Optional[FRRiskConfig] = None,
    ):
        self.gateway = gateway
        self.tracker = tracker
        self.kill_switch = kill_switch
        self.risk_config = risk_config or FRRiskConfig()
        self.recent_policies: List[FRPolicy] = []
        self._execution_locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, symbol: str) -> asyncio.Lock:
        """Get per-symbol concurrency lock."""
        sym = symbol.strip().upper()
        if sym not in self._execution_locks:
            self._execution_locks[sym] = asyncio.Lock()
        return self._execution_locks[sym]

    async def execute_policy(self, policy: FRPolicy) -> Dict[str, Any]:
        """Dispatch policy to appropriate action handler."""
        sym = policy.symbol.strip().upper()
        allowed, reason = self.kill_switch.is_execution_allowed(symbol=sym)
        if not allowed:
            logger.warning(f"[{sym}] Policy {policy.action.value} blocked by kill switch: {reason}")
            return {"status": "BLOCKED", "reason": reason, "policy_id": policy.policy_id}

        async with self._get_lock(sym):
            logger.info(
                f"[{sym}] Processing Policy {policy.policy_id}: Action={policy.action.value} "
                f"Long={policy.exchange_long.upper()} Short={policy.exchange_short.upper()} "
                f"TargetNotional=${policy.target_notional_usdt} Edge={policy.expected_net_edge_bps}bps"
            )

            dual_pos = self.tracker.get_or_create_position(sym, policy.exchange_long, policy.exchange_short)
            dual_pos.last_policy_action = policy.action.value

            if policy.action == FRAction.OPEN:
                return await self._handle_open(policy, dual_pos)
            elif policy.action == FRAction.HOLD:
                return await self._handle_hold(policy, dual_pos)
            elif policy.action == FRAction.REDUCE:
                return await self._handle_reduce(policy, dual_pos)
            elif policy.action == FRAction.EXIT:
                return await self._handle_exit(policy, dual_pos)
            elif policy.action == FRAction.PAUSE:
                return await self._handle_pause(policy, dual_pos)
            else:
                logger.error(f"[{sym}] Unknown policy action: {policy.action}")
                return {"status": "UNKNOWN_ACTION"}

    # ── Action Handlers ──

    async def _actual_leg_size(self, exchange: str, symbol: str, position_side: str) -> float:
        """
        Query actual live position size from exchange for a specific leg side.
        Parses contracts | positionAmt | size (absolute value).
        """
        try:
            raw_positions = await self.gateway.fetch_positions(exchange, symbols=[symbol])
            pos_side_upper = position_side.upper()
            for raw in raw_positions:
                r_side = str(raw.get("side", raw.get("positionSide", "BOTH"))).upper()
                pos_idx = raw.get("positionIdx")
                if pos_idx is not None:
                    if pos_idx == 1:
                        r_side = "LONG"
                    elif pos_idx == 2:
                        r_side = "SHORT"
                if r_side == pos_side_upper or r_side == "BOTH":
                    amt = abs(float(raw.get("contracts", raw.get("positionAmt", raw.get("size", 0.0))) or 0.0))
                    if amt > 0:
                        return amt
            return 0.0
        except Exception as e:
            logger.error(f"[{exchange.upper()}][{symbol}] Error fetching actual leg size: {e}")
            return 0.0

    async def _handle_open(self, policy: FRPolicy, dual_pos: DualLegPosition) -> Dict[str, Any]:
        """
        Execute OPEN Action:
        1. Check margin on both exchanges.
        2. Fetch current prices and quantize size.
        3. Fire dual legs concurrently (BUY Long on ex_long, SELL Short on ex_short) with 5s timeout tasks.
        4. Legging Risk Protection: Cancel on timeout, check actual filled size on exchange, and rollback.
        """
        sym = policy.symbol.strip().upper()
        ex_long = policy.exchange_long.lower()
        ex_short = policy.exchange_short.lower()

        # Check if already open
        if dual_pos.status == "OPEN" and dual_pos.long_leg.size > 0 and dual_pos.short_leg.size > 0:
            logger.info(f"[{sym}] Position already OPEN. Skipping duplicate OPEN.")
            return {"status": "ALREADY_OPEN", "symbol": sym}

        # Check Free Margin on both exchanges
        long_free = await self.gateway.get_free_margin(ex_long)
        short_free = await self.gateway.get_free_margin(ex_short)
        required_margin = (policy.target_notional_usdt / max(1, policy.max_leverage))

        if long_free < required_margin:
            err = f"Insufficient margin on {ex_long.upper()} (free ${long_free:.2f} < req ${required_margin:.2f})"
            logger.error(f"[{sym}] {err}")
            return {"status": "INSUFFICIENT_MARGIN", "exchange": ex_long, "detail": err}

        if short_free < required_margin:
            err = f"Insufficient margin on {ex_short.upper()} (free ${short_free:.2f} < req ${required_margin:.2f})"
            logger.error(f"[{sym}] {err}")
            return {"status": "INSUFFICIENT_MARGIN", "exchange": ex_short, "detail": err}

        # Fetch Reference Prices
        p_long = await self.gateway.fetch_ticker_price(ex_long, sym)
        p_short = await self.gateway.fetch_ticker_price(ex_short, sym)
        ref_price = max(p_long, p_short)
        if ref_price <= 0:
            logger.error(f"[{sym}] Invalid reference price (long={p_long}, short={p_short})")
            return {"status": "INVALID_PRICE"}

        # Calculate & Quantize Target Qty
        raw_qty = policy.target_notional_usdt / ref_price
        qty_long = self.gateway.amount_to_precision(ex_long, sym, raw_qty)
        qty_short = self.gateway.amount_to_precision(ex_short, sym, raw_qty)
        target_qty = min(qty_long, qty_short)

        if target_qty <= 0:
            logger.error(f"[{sym}] Target quantity {target_qty} is zero after precision quantization.")
            return {"status": "ZERO_QUANTITY"}

        dual_pos.status = "OPENING"
        ts = int(time.time() * 1000)
        cid_long = f"fr_open_l_{sym.replace('/', '_')}_{ts}"
        cid_short = f"fr_open_s_{sym.replace('/', '_')}_{ts}"

        logger.info(f"[{sym}] Firing DUAL OPEN orders: qty={target_qty} (Long on {ex_long.upper()}, Short on {ex_short.upper()})")

        # Concurrently fire both legs with 5s timeout using separate tasks
        long_task = asyncio.create_task(
            self.gateway.create_hedge_order(
                exchange=ex_long,
                symbol=sym,
                side="buy",
                position_side="LONG",
                amount=target_qty,
                order_type="market",
                client_order_id=cid_long,
            )
        )
        short_task = asyncio.create_task(
            self.gateway.create_hedge_order(
                exchange=ex_short,
                symbol=sym,
                side="sell",
                position_side="SHORT",
                amount=target_qty,
                order_type="market",
                client_order_id=cid_short,
            )
        )

        open_timeout = float(getattr(self.risk_config, "order_timeout_sec", 5.0))
        done, pending = await asyncio.wait(
            {long_task, short_task},
            timeout=open_timeout,
            return_when=asyncio.ALL_COMPLETED,
        )

        for t in pending:
            t.cancel()
            t_cid = cid_long if t == long_task else cid_short
            t_ex = ex_long if t == long_task else ex_short
            logger.critical(f"[{sym}] Dual open leg timeout (>5.0s) on {t_ex.upper()}! Cancelled task (cid={t_cid}).")

        res_long = None
        if long_task in done and not long_task.cancelled():
            try:
                res_long = long_task.result()
            except Exception as e:
                logger.critical(f"[{sym}] Long leg exception: {e}")
                res_long = None

        res_short = None
        if short_task in done and not short_task.cancelled():
            try:
                res_short = short_task.result()
            except Exception as e:
                logger.critical(f"[{sym}] Short leg exception: {e}")
                res_short = None

        long_ok = (isinstance(res_long, dict) and not isinstance(res_long, Exception))
        short_ok = (isinstance(res_short, dict) and not isinstance(res_short, Exception))

        # ── LEGGING RISK PROTECTION ROLLBACK ──
        if long_ok and not short_ok:
            actual_from_exchange = await self._actual_leg_size(ex_long, sym, "LONG")
            qty_confirmed = float(res_long.get("filled", res_long.get("amount", target_qty)) or target_qty) if long_ok else 0.0
            actual_qty = max(qty_confirmed, actual_from_exchange)
            if actual_qty > 0:
                logger.critical(f"[{sym}] LEGGING RISK: Long leg filled ({actual_qty}) on {ex_long.upper()} but Short leg failed on {ex_short.upper()}! Emergency rollback...")
                rollback_res = await self.gateway.emergency_market_close(ex_long, sym, "LONG", actual_qty)
                residual = await self._actual_leg_size(ex_long, sym, "LONG")
                if residual > 1e-5:
                    logger.critical(f"[{sym}] Residual size after rollback ({residual})! Tripping exchange kill switch.")
                    self.kill_switch.trip_exchange(ex_long, f"legging rollback residual on {sym}: {residual}")
                dual_pos.status = "FLAT"
                return {"status": "LEGGING_ROLLBACK", "detail": "Long leg rolled back due to Short leg failure", "rollback": rollback_res is not None, "actual_qty": actual_qty}
            else:
                logger.warning(f"[{sym}] Long leg not filled on exchange. Skipping rollback.")
                dual_pos.status = "FLAT"
                return {"status": "BOTH_LEGS_FAILED"}

        if short_ok and not long_ok:
            actual_from_exchange = await self._actual_leg_size(ex_short, sym, "SHORT")
            qty_confirmed = float(res_short.get("filled", res_short.get("amount", target_qty)) or target_qty) if short_ok else 0.0
            actual_qty = max(qty_confirmed, actual_from_exchange)
            if actual_qty > 0:
                logger.critical(f"[{sym}] LEGGING RISK: Short leg filled ({actual_qty}) on {ex_short.upper()} but Long leg failed on {ex_long.upper()}! Emergency rollback...")
                rollback_res = await self.gateway.emergency_market_close(ex_short, sym, "SHORT", actual_qty)
                residual = await self._actual_leg_size(ex_short, sym, "SHORT")
                if residual > 1e-5:
                    logger.critical(f"[{sym}] Residual size after rollback ({residual})! Tripping exchange kill switch.")
                    self.kill_switch.trip_exchange(ex_short, f"legging rollback residual on {sym}: {residual}")
                dual_pos.status = "FLAT"
                return {"status": "LEGGING_ROLLBACK", "detail": "Short leg rolled back due to Long leg failure", "rollback": rollback_res is not None, "actual_qty": actual_qty}
            else:
                logger.warning(f"[{sym}] Short leg not filled on exchange. Skipping rollback.")
                dual_pos.status = "FLAT"
                return {"status": "BOTH_LEGS_FAILED"}

        if not long_ok and not short_ok:
            # Check if ghost-fills occurred despite task timeouts/errors
            actual_long = await self._actual_leg_size(ex_long, sym, "LONG")
            actual_short = await self._actual_leg_size(ex_short, sym, "SHORT")
            if actual_long > 0:
                await self.gateway.emergency_market_close(ex_long, sym, "LONG", actual_long)
                residual = await self._actual_leg_size(ex_long, sym, "LONG")
                if residual > 1e-5:
                    self.kill_switch.trip_exchange(ex_long, f"legging rollback residual on {sym}: {residual}")
            if actual_short > 0:
                await self.gateway.emergency_market_close(ex_short, sym, "SHORT", actual_short)
                residual = await self._actual_leg_size(ex_short, sym, "SHORT")
                if residual > 1e-5:
                    self.kill_switch.trip_exchange(ex_short, f"legging rollback residual on {sym}: {residual}")
            logger.error(f"[{sym}] Both legs failed to open.")
            dual_pos.status = "FLAT"
            return {"status": "BOTH_LEGS_FAILED"}

        # Both legs succeeded!
        dual_pos.status = "OPEN"
        now = time.time()
        dual_pos.created_at = now
        dual_pos.updated_at = now
        
        # Update local legs
        self.tracker.update_leg(sym, ex_long, "LONG", target_qty, p_long or ref_price, p_long or ref_price, 0.0, leverage=policy.max_leverage)
        self.tracker.update_leg(sym, ex_short, "SHORT", target_qty, p_short or ref_price, p_short or ref_price, 0.0, leverage=policy.max_leverage)

        logger.info(f"[{sym}] Dual-Leg Position successfully OPENED on {ex_long.upper()} & {ex_short.upper()}.")
        return {"status": "SUCCESS", "action": "OPEN", "symbol": sym, "size": target_qty}

    async def _handle_hold(self, policy: FRPolicy, dual_pos: DualLegPosition) -> Dict[str, Any]:
        """
        Execute HOLD Action:
        - Re-check mark prices, update uPnL, inspect stop loss.
        """
        sym = policy.symbol.strip().upper()
        p_long = await self.gateway.fetch_ticker_price(dual_pos.long_leg.exchange, sym)
        p_short = await self.gateway.fetch_ticker_price(dual_pos.short_leg.exchange, sym)

        if p_long > 0:
            dual_pos.long_leg.mark_price = p_long
            if dual_pos.long_leg.entry_price > 0:
                dual_pos.long_leg.unrealized_pnl = (p_long - dual_pos.long_leg.entry_price) * dual_pos.long_leg.size

        if p_short > 0:
            dual_pos.short_leg.mark_price = p_short
            if dual_pos.short_leg.entry_price > 0:
                dual_pos.short_leg.unrealized_pnl = (dual_pos.short_leg.entry_price - p_short) * dual_pos.short_leg.size

        dual_pos.recalculate()

        # Stop-loss check
        if dual_pos.net_pnl < -abs(self.risk_config.max_loss_usd):
            logger.warning(
                f"[{sym}] STOP-LOSS TRIGGERED! Net PnL ${dual_pos.net_pnl:.2f} < -${self.risk_config.max_loss_usd:.2f}. Initiating emergency exit..."
            )
            return await self._handle_exit(policy, dual_pos)

        return {"status": "HOLDING", "symbol": sym, "net_pnl": dual_pos.net_pnl}

    async def _handle_reduce(self, policy: FRPolicy, dual_pos: DualLegPosition) -> Dict[str, Any]:
        """
        Execute REDUCE Action:
        - Proportionally reduce volume on both legs to reduce_to_notional_usdt.
        """
        sym = policy.symbol.strip().upper()
        target_notional = policy.reduce_to_notional_usdt or (policy.target_notional_usdt * 0.5)
        current_size = min(dual_pos.long_leg.size, dual_pos.short_leg.size)
        ref_price = max(dual_pos.long_leg.mark_price, dual_pos.short_leg.mark_price)

        if ref_price <= 0 or current_size <= 0:
            return {"status": "NO_POSITION_TO_REDUCE"}

        new_target_qty = target_notional / ref_price
        reduce_qty = max(0.0, current_size - new_target_qty)

        if reduce_qty <= 0:
            logger.info(f"[{sym}] Current size already <= target reduce size.")
            return {"status": "ALREADY_REDUCED"}

        ex_long = dual_pos.long_leg.exchange
        ex_short = dual_pos.short_leg.exchange
        fmt_reduce_long = self.gateway.amount_to_precision(ex_long, sym, reduce_qty)
        fmt_reduce_short = self.gateway.amount_to_precision(ex_short, sym, reduce_qty)
        fmt_qty = min(fmt_reduce_long, fmt_reduce_short)

        if fmt_qty <= 0:
            return {"status": "ZERO_REDUCE_QTY"}

        logger.info(f"[{sym}] Executing REDUCE by {fmt_qty} on both legs...")
        # Concurrently close delta amount
        long_close = self.gateway.create_hedge_order(ex_long, sym, "sell", "LONG", fmt_qty, order_type="market")
        short_close = self.gateway.create_hedge_order(ex_short, sym, "buy", "SHORT", fmt_qty, order_type="market")

        results = await asyncio.gather(long_close, short_close, return_exceptions=True)
        res_long, res_short = results[0], results[1]

        long_reduced = isinstance(res_long, dict) and not isinstance(res_long, Exception)
        short_reduced = isinstance(res_short, dict) and not isinstance(res_short, Exception)

        if long_reduced:
            dual_pos.long_leg.size = max(0.0, dual_pos.long_leg.size - fmt_qty)
        if short_reduced:
            dual_pos.short_leg.size = max(0.0, dual_pos.short_leg.size - fmt_qty)

        dual_pos.recalculate()

        if long_reduced and short_reduced:
            return {"status": "REDUCED", "symbol": sym, "reduced_qty": fmt_qty}

        logger.critical(
            f"[{sym}] PARTIAL REDUCE: long_reduced={long_reduced}, short_reduced={short_reduced}. Reconcile will handle residual."
        )
        return {
            "status": "REDUCE_PARTIAL",
            "symbol": sym,
            "reduced_qty": fmt_qty,
            "long_reduced": long_reduced,
            "short_reduced": short_reduced,
        }

    async def _handle_exit(self, policy: FRPolicy, dual_pos: DualLegPosition) -> Dict[str, Any]:
        """
        Execute EXIT Action:
        - Close both legs concurrently with MARKET orders in Hedge Mode (NO reduceOnly).
        - Close Long: SELL with positionSide='LONG'
        - Close Short: BUY with positionSide='SHORT'
        """
        sym = policy.symbol.strip().upper()
        ex_long = dual_pos.long_leg.exchange
        ex_short = dual_pos.short_leg.exchange
        size_long = dual_pos.long_leg.size
        size_short = dual_pos.short_leg.size

        if size_long <= 0 and size_short <= 0:
            dual_pos.status = "FLAT"
            return {"status": "ALREADY_FLAT", "symbol": sym}

        dual_pos.status = "CLOSING"
        logger.info(f"[{sym}] Executing EXIT: Closing Long ({size_long} on {ex_long.upper()}) and Short ({size_short} on {ex_short.upper()})")

        tasks = []
        if size_long > 0:
            tasks.append(self.gateway.emergency_market_close(ex_long, sym, "LONG", size_long))
        else:
            tasks.append(asyncio.sleep(0, result={"skipped": True}))

        if size_short > 0:
            tasks.append(self.gateway.emergency_market_close(ex_short, sym, "SHORT", size_short))
        else:
            tasks.append(asyncio.sleep(0, result={"skipped": True}))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        res_long, res_short = results[0], results[1]

        long_closed = isinstance(res_long, dict) and not isinstance(res_long, Exception)
        short_closed = isinstance(res_short, dict) and not isinstance(res_short, Exception)

        if long_closed:
            dual_pos.long_leg.size = 0.0
        if short_closed:
            dual_pos.short_leg.size = 0.0

        dual_pos.recalculate()

        if long_closed and short_closed:
            dual_pos.status = "FLAT"
            logger.info(f"[{sym}] Dual-Leg Position EXIT completed. Account is 100% FLAT.")
            return {"status": "EXIT_COMPLETED", "symbol": sym}

        dual_pos.status = "CLOSING"
        logger.critical(
            f"[{sym}] PARTIAL EXIT: long_closed={long_closed}, short_closed={short_closed}. Reconcile will handle residual."
        )
        return {
            "status": "EXIT_PARTIAL",
            "symbol": sym,
            "long_closed": long_closed,
            "short_closed": short_closed,
        }

    async def _handle_pause(self, policy: FRPolicy, dual_pos: DualLegPosition) -> Dict[str, Any]:
        """
        Execute PAUSE Action:
        - Cancel open orders for symbol on the actual leg exchanges and pause in kill switch.
        """
        sym = policy.symbol.strip().upper()
        self.kill_switch.pause_symbol(sym, reason="Policy PAUSE action")
        dual_pos.is_paused = True
        exchanges_to_cancel = {dual_pos.long_leg.exchange.lower(), dual_pos.short_leg.exchange.lower()}
        for ex in exchanges_to_cancel:
            if ex:
                await self.gateway.cancel_all_orders(ex, sym)
        logger.info(f"[{sym}] Symbol paused and open orders cancelled on {exchanges_to_cancel}.")
        return {"status": "PAUSED", "symbol": sym}

    # ── Policy Polling from Decision Layer ──

    async def fetch_and_execute_policies(self, client: Optional[httpx.AsyncClient] = None) -> List[Dict[str, Any]]:
        """Poll GET /api/v1/fr/policies from Decision Layer and dispatch executions."""
        if not self.risk_config.auto_execution_enabled:
            return []

        base_url = self.risk_config.decision_layer_url.rstrip("/")
        url = f"{base_url}/api/v1/fr/policies"

        # Build current positions payload to pass to Decision Layer
        active_positions = self.tracker.get_active_positions()
        pos_payload = []
        for p in active_positions:
            pos_payload.append({
                "symbol": p.symbol,
                "exchange_long": p.long_leg.exchange,
                "exchange_short": p.short_leg.exchange,
                "status": p.status,
                "current_notional_usdt": p.long_leg.notional + p.short_leg.notional,
            })

        params = {}
        if pos_payload:
            params["current_positions"] = json.dumps(pos_payload)

        try:
            should_close_client = False
            if client is None:
                client = httpx.AsyncClient(timeout=8.0)
                should_close_client = True

            try:
                resp = await client.get(url, params=params)
                if resp.status_code != 200:
                    logger.warning(f"Decision Layer returned status {resp.status_code}: {resp.text}")
                    return []

                data = resp.json()
                items = data.get("items", [])
                self.recent_policies.clear()
                results = []

                for item in items:
                    try:
                        policy = FRPolicy(**item)
                        self.recent_policies.append(policy)
                        res = await self.execute_policy(policy)
                        results.append(res)
                    except Exception as e:
                        logger.error(f"Error parsing/executing policy item {item}: {e}")

                return results
            finally:
                if should_close_client:
                    await client.aclose()

        except Exception as e:
            logger.warning(f"Failed to poll Decision Layer at {url}: {e}")
            return []
