"""Three-Tier Kill Switch and 6-Phase Emergency Kill-All for Funding Rate Arbitrage."""
import asyncio
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from loguru import logger

from app.core.fr_execution.gateway import MultiExchangeGateway
from app.core.fr_execution.position_tracker import FRPositionTracker


class ThreeTierKillSwitch:
    """
    3-Tier Risk Hierarchy:
    - Tier 1: Symbol Kill Switch (isolate / halt specific symbols)
    - Tier 2: Exchange Kill Switch (isolate single exchange)
    - Tier 3: Global Kill Switch (system-wide immediate halt)
    """

    def __init__(self):
        self.paused_symbols: Set[str] = set()
        self.tripped_exchanges: Dict[str, str] = {}  # exchange -> trip reason
        self.is_global_tripped: bool = False
        self.global_trip_reason: Optional[str] = None
        self._lock = asyncio.Lock()

    # ── Tier 1: Symbol Kill Switch ──

    def pause_symbol(self, symbol: str, reason: str = "Manual pause") -> None:
        """Pause trading and execution for specific symbol."""
        sym = symbol.strip().upper()
        self.paused_symbols.add(sym)
        logger.warning(f"[KILL-SWITCH][TIER 1] Symbol {sym} PAUSED. Reason: {reason}")

    def resume_symbol(self, symbol: str) -> None:
        """Resume trading for specific symbol."""
        sym = symbol.strip().upper()
        self.paused_symbols.discard(sym)
        logger.info(f"[KILL-SWITCH][TIER 1] Symbol {sym} RESUMED.")

    def is_symbol_paused(self, symbol: str) -> bool:
        """Check if symbol is explicitly paused."""
        return symbol.strip().upper() in self.paused_symbols

    # ── Tier 2: Exchange Kill Switch ──

    def trip_exchange(self, exchange: str, reason: str) -> None:
        """Trip kill switch for specific exchange (Binance or Bybit)."""
        ex = exchange.lower()
        self.tripped_exchanges[ex] = reason
        logger.critical(f"[KILL-SWITCH][TIER 2] Exchange {ex.upper()} TRIPPED. Reason: {reason}")

    def reset_exchange(self, exchange: str) -> None:
        """Reset kill switch for specific exchange."""
        ex = exchange.lower()
        self.tripped_exchanges.pop(ex, None)
        logger.info(f"[KILL-SWITCH][TIER 2] Exchange {ex.upper()} RESET.")

    def is_exchange_tripped(self, exchange: str) -> bool:
        """Check if specific exchange is tripped."""
        return exchange.lower() in self.tripped_exchanges

    # ── Tier 3: Global Kill Switch ──

    def trip_global(self, reason: str) -> None:
        """Trip system-wide global kill switch."""
        self.is_global_tripped = True
        self.global_trip_reason = reason
        logger.critical(f"[KILL-SWITCH][TIER 3] GLOBAL KILL SWITCH TRIPPED! Reason: {reason}")

    def reset_global(self) -> None:
        """Reset system-wide global kill switch."""
        self.is_global_tripped = False
        self.global_trip_reason = None
        logger.info("[KILL-SWITCH][TIER 3] Global Kill Switch RESET.")

    def is_execution_allowed(self, symbol: Optional[str] = None, exchange: Optional[str] = None) -> Tuple[bool, str]:
        """Check all 3 tiers before allowing an order or position open."""
        if self.is_global_tripped:
            return False, f"Global Kill Switch active: {self.global_trip_reason}"

        if exchange and self.is_exchange_tripped(exchange):
            return False, f"Exchange {exchange.upper()} Kill Switch active: {self.tripped_exchanges.get(exchange.lower())}"

        if symbol and self.is_symbol_paused(symbol):
            return False, f"Symbol {symbol.upper()} is paused."

        return True, "OK"

    # ── 6-Phase Emergency Kill-All Procedure ──

    async def emergency_kill_all(
        self,
        gateway: MultiExchangeGateway,
        tracker: FRPositionTracker,
    ) -> Dict[str, Any]:
        """
        Execute 6-Phase Native Hedge Emergency Kill-All:
        Phase 1: Trip global kill switch to block incoming requests.
        Phase 2: Cancel all open orders on Binance Futures.
        Phase 3: Cancel all open orders on Bybit Linear.
        Phase 4: Market-close all open positions on Binance (LONG/SHORT) in Hedge Mode without reduceOnly.
        Phase 5: Market-close all open positions on Bybit (LONG/SHORT) in Hedge Mode without reduceOnly.
        Phase 6: Re-fetch positions on both exchanges to confirm net open position == 0.
        """
        async with self._lock:
            start_time = time.time()
            self.trip_global("EMERGENCY KILL-ALL INITIATED")
            logger.critical("🚨 STARTING 6-PHASE ARBITRAGE EMERGENCY KILL-ALL PROCEDURE...")

            report: Dict[str, Any] = {
                "timestamp": start_time,
                "phases": {},
                "status": "IN_PROGRESS",
            }

            # Phase 1: Global Switch Tripped
            report["phases"]["phase_1_global_lock"] = "COMPLETED"

            # Phase 2: Cancel all Binance orders
            try:
                binance_cancel_ok = await gateway.cancel_all_orders("binance")
                report["phases"]["phase_2_cancel_binance_orders"] = "SUCCESS" if binance_cancel_ok else "FAILED"
            except Exception as e:
                report["phases"]["phase_2_cancel_binance_orders"] = f"ERROR: {e}"

            # Phase 3: Cancel all Bybit orders
            try:
                bybit_cancel_ok = await gateway.cancel_all_orders("bybit")
                report["phases"]["phase_3_cancel_bybit_orders"] = "SUCCESS" if bybit_cancel_ok else "FAILED"
            except Exception as e:
                report["phases"]["phase_3_cancel_bybit_orders"] = f"ERROR: {e}"

            # Phase 4: Market close Binance positions
            binance_close_results = []
            try:
                raw_binance_pos = await gateway.fetch_positions("binance")
                for p in raw_binance_pos:
                    sym = p.get("symbol", "").replace("/", "").replace(":USDT", "").upper()
                    size = abs(float(p.get("contracts", p.get("positionAmt", 0.0)) or 0.0))
                    side = str(p.get("side", p.get("positionSide", "LONG"))).upper()
                    if size > 0 and sym:
                        res = await gateway.emergency_market_close("binance", sym, side, size)
                        binance_close_results.append({"symbol": sym, "side": side, "size": size, "res": res is not None})
                report["phases"]["phase_4_close_binance_positions"] = binance_close_results
            except Exception as e:
                report["phases"]["phase_4_close_binance_positions"] = f"ERROR: {e}"

            # Phase 5: Market close Bybit positions
            bybit_close_results = []
            try:
                raw_bybit_pos = await gateway.fetch_positions("bybit")
                for p in raw_bybit_pos:
                    sym = p.get("symbol", "").replace("/", "").replace(":USDT", "").upper()
                    size = abs(float(p.get("size", p.get("contracts", 0.0)) or 0.0))
                    pos_idx = int(p.get("positionIdx", 0) or 0)
                    side = "LONG" if pos_idx == 1 else ("SHORT" if pos_idx == 2 else "LONG")
                    if size > 0 and sym:
                        res = await gateway.emergency_market_close("bybit", sym, side, size)
                        bybit_close_results.append({"symbol": sym, "side": side, "size": size, "res": res is not None})
                report["phases"]["phase_5_close_bybit_positions"] = bybit_close_results
            except Exception as e:
                report["phases"]["phase_5_close_bybit_positions"] = f"ERROR: {e}"

            # Phase 6: Verification & Flat Audit
            await asyncio.sleep(1.0)
            remaining_binance = await gateway.fetch_positions("binance")
            remaining_bybit = await gateway.fetch_positions("bybit")

            binance_open_cnt = sum(1 for p in remaining_binance if abs(float(p.get("contracts", p.get("positionAmt", 0.0)) or 0.0)) > 0)
            bybit_open_cnt = sum(1 for p in remaining_bybit if abs(float(p.get("size", p.get("contracts", 0.0)) or 0.0)) > 0)

            report["phases"]["phase_6_reconcile"] = {
                "binance_remaining_positions": binance_open_cnt,
                "bybit_remaining_positions": bybit_open_cnt,
                "is_flat": (binance_open_cnt == 0 and bybit_open_cnt == 0),
            }

            # Update tracker
            tracker.reconcile_with_exchange_positions(remaining_binance, remaining_bybit)

            report["status"] = "COMPLETED" if (binance_open_cnt == 0 and bybit_open_cnt == 0) else "PARTIAL_REMAINING"
            report["duration_seconds"] = round(time.time() - start_time, 2)
            logger.info(f"🚨 6-Phase Emergency Kill-All finished with status: {report['status']} in {report['duration_seconds']}s")
            return report
