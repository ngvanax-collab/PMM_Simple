"""Dual-Leg Position Tracker for Funding Rate Arbitrage with Realtime PnL & Funding Accrual."""
import time
from typing import Any, Dict, List, Optional
from loguru import logger

from app.core.fr_execution.models import DualLegPosition, FRSummaryMetrics, LegPositionState


class FRPositionTracker:
    """Manages dual-leg arbitrage position states with 60s REST reconciliation and WS events."""

    def __init__(self):
        self.positions: Dict[str, DualLegPosition] = {}  # symbol -> DualLegPosition
        self.realized_funding_pnl: float = 0.0
        self.last_reconciled_at: float = 0.0
        self.processed_funding_ids: set = set()  # Deduplicate real funding transactions (TASK L-2)

    @staticmethod
    def _canon(symbol: str) -> str:
        """
        Unify any symbol format into standard canonical format 'BASE/USDT:USDT'.
        Handles: 'SOLUSDT', 'SOL/USDT', 'SOL/USDT:USDT', '1000PEPEUSDT', etc.
        """
        if not symbol:
            return ""
        s = str(symbol).strip().upper()
        # Remove trailing :USDT or :USDC if already present
        if ":" in s:
            s = s.split(":")[0]
        # Remove slash if present to get clean pair
        if "/" in s:
            parts = s.split("/")
            base = parts[0]
            quote = parts[1] if len(parts) > 1 else "USDT"
            return f"{base}/{quote}:{quote}"
        # If no slash, e.g. SOLUSDT, 1000PEPEUSDT
        if s.endswith("USDT"):
            base = s[:-4]
            return f"{base}/USDT:USDT"
        elif s.endswith("USDC"):
            base = s[:-4]
            return f"{base}/USDC:USDC"
        return f"{s}/USDT:USDT"

    def get_or_create_position(self, symbol: str, ex_long: str = "binance", ex_short: str = "bybit") -> DualLegPosition:
        """Retrieve existing dual-leg position or initialize a new flat one."""
        sym = self._canon(symbol)
        if sym not in self.positions:
            now = time.time()
            long_leg = LegPositionState(
                exchange=ex_long.lower(),
                symbol=sym,
                position_side="LONG",
                entry_time=now,
                last_updated=now,
            )
            short_leg = LegPositionState(
                exchange=ex_short.lower(),
                symbol=sym,
                position_side="SHORT",
                entry_time=now,
                last_updated=now,
            )
            dual_pos = DualLegPosition(
                symbol=sym,
                long_leg=long_leg,
                short_leg=short_leg,
                status="FLAT",
                created_at=now,
                updated_at=now,
            )
            self.positions[sym] = dual_pos
        return self.positions[sym]

    def get_position(self, symbol: str) -> Optional[DualLegPosition]:
        """Get dual position for symbol."""
        return self.positions.get(self._canon(symbol))

    def get_all_positions(self) -> List[DualLegPosition]:
        """Return list of all tracked dual positions."""
        for p in self.positions.values():
            p.recalculate()
        return list(self.positions.values())

    def get_active_positions(self) -> List[DualLegPosition]:
        """Return list of active non-flat positions."""
        active = []
        for p in self.positions.values():
            p.recalculate()
            if p.long_leg.size > 0 or p.short_leg.size > 0 or p.status in ("OPEN", "OPENING", "CLOSING"):
                active.append(p)
        return active

    def update_leg(
        self,
        symbol: str,
        exchange: str,
        position_side: str,
        size: float,
        entry_price: float,
        mark_price: float,
        upnl: float,
        funding_accrued: float = 0.0,
        leverage: int = 1,
    ) -> DualLegPosition:
        """Update individual leg state and recalculate pair aggregates."""
        sym = self._canon(symbol)
        dual_pos = self.get_or_create_position(sym)
        pos_side = position_side.upper()
        now = time.time()

        leg = dual_pos.long_leg if pos_side == "LONG" else dual_pos.short_leg
        leg.exchange = exchange.lower()
        leg.size = abs(size)
        leg.entry_price = entry_price
        leg.mark_price = mark_price
        leg.notional = round(leg.size * mark_price, 2) if mark_price > 0 else 0.0
        leg.unrealized_pnl = upnl
        if funding_accrued != 0.0:
            leg.funding_accrued += funding_accrued
        leg.leverage = leverage
        leg.last_updated = now

        # Update overall status
        if dual_pos.long_leg.size > 0 and dual_pos.short_leg.size > 0:
            dual_pos.status = "OPEN"
        elif dual_pos.long_leg.size == 0 and dual_pos.short_leg.size == 0:
            dual_pos.status = "FLAT"
        else:
            dual_pos.status = "OPENING"  # Partial / Legging state

        dual_pos.updated_at = now
        dual_pos.recalculate()
        return dual_pos

    def reconcile_with_exchange_positions(
        self,
        binance_raw_positions: List[Dict[str, Any]],
        bybit_raw_positions: List[Dict[str, Any]],
    ) -> None:
        """Reconcile internal position tracker against REST snapshot from both exchanges."""
        self.last_reconciled_at = time.time()

        # Parse Binance Positions
        for raw in binance_raw_positions:
            raw_sym = raw.get("symbol", "")
            sym = self._canon(raw_sym)
            pos_side = str(raw.get("side", raw.get("positionSide", "BOTH"))).upper()
            if pos_side not in ("LONG", "SHORT"):
                # One-way mode fallback if any
                contracts = float(raw.get("contracts", raw.get("positionAmt", 0.0)) or 0.0)
                pos_side = "LONG" if contracts >= 0 else "SHORT"

            size = abs(float(raw.get("contracts", raw.get("positionAmt", 0.0)) or 0.0))
            entry_p = float(raw.get("entryPrice", 0.0) or 0.0)
            mark_p = float(raw.get("markPrice", 0.0) or 0.0)
            upnl = float(raw.get("unrealizedPnl", raw.get("unRealizedProfit", 0.0)) or 0.0)
            leverage = int(raw.get("leverage", 1) or 1)

            if sym:
                if sym not in self.positions:
                    logger.warning(f"[{sym}] Unmanaged exchange position detected during Binance reconcile.")
                dual_pos = self.get_or_create_position(sym, ex_long="binance", ex_short="bybit")
                if pos_side == "LONG" and dual_pos.long_leg.exchange == "binance":
                    self.update_leg(sym, "binance", "LONG", size, entry_p, mark_p, upnl, leverage=leverage)
                elif pos_side == "SHORT" and dual_pos.short_leg.exchange == "binance":
                    self.update_leg(sym, "binance", "SHORT", size, entry_p, mark_p, upnl, leverage=leverage)

        # Parse Bybit Positions
        for raw in bybit_raw_positions:
            raw_sym = raw.get("symbol", "")
            sym = self._canon(raw_sym)
            pos_idx = int(raw.get("positionIdx", 0) or 0)
            pos_side = "LONG" if pos_idx == 1 else ("SHORT" if pos_idx == 2 else "BOTH")
            
            if pos_side not in ("LONG", "SHORT"):
                side_str = str(raw.get("side", "")).upper()
                pos_side = "LONG" if side_str == "BUY" else "SHORT"

            size = abs(float(raw.get("size", raw.get("contracts", 0.0)) or 0.0))
            entry_p = float(raw.get("avgPrice", raw.get("entryPrice", 0.0)) or 0.0)
            mark_p = float(raw.get("markPrice", 0.0) or 0.0)
            upnl = float(raw.get("unrealisedPnl", raw.get("unrealizedPnl", 0.0)) or 0.0)
            leverage = int(raw.get("leverage", 1) or 1)

            if sym:
                if sym not in self.positions:
                    logger.warning(f"[{sym}] Unmanaged exchange position detected during Bybit reconcile.")
                dual_pos = self.get_or_create_position(sym, ex_long="binance", ex_short="bybit")
                if pos_side == "LONG" and dual_pos.long_leg.exchange == "bybit":
                    self.update_leg(sym, "bybit", "LONG", size, entry_p, mark_p, upnl, leverage=leverage)
                elif pos_side == "SHORT" and dual_pos.short_leg.exchange == "bybit":
                    self.update_leg(sym, "bybit", "SHORT", size, entry_p, mark_p, upnl, leverage=leverage)

        for p in self.positions.values():
            p.recalculate()

    def get_summary_metrics(self, binance_free_margin: float = 0.0, bybit_free_margin: float = 0.0) -> FRSummaryMetrics:
        """Compute aggregated portfolio summary metrics for UI display."""
        active_positions = self.get_active_positions()
        total_pnl = self.realized_funding_pnl
        total_notional = 0.0

        for pos in active_positions:
            total_pnl += pos.total_funding_accrued + pos.net_upnl
            total_notional += (pos.long_leg.notional + pos.short_leg.notional)

        # Compute APR based on real net PnL and active capital (TASK L-1)
        net_apr = 0.0
        if total_notional > 0 and total_pnl > 0:
            net_apr = round((total_pnl / total_notional) * 365.0 * 100.0, 2)

        return FRSummaryMetrics(
            total_realized_funding_pnl=round(self.realized_funding_pnl, 4),
            net_arbitrage_apr=net_apr,
            active_arb_pairs=len(active_positions),
            binance_free_margin=round(binance_free_margin, 2),
            bybit_free_margin=round(bybit_free_margin, 2),
            total_equity_usdt=round(binance_free_margin + bybit_free_margin, 2),
            last_reconciled_at=self.last_reconciled_at,
        )

    def record_funding_payment(
        self,
        symbol: str,
        exchange: str,
        position_side: str = "LONG",
        payment_amount: float = 0.0,
        amount: Optional[float] = None,
        funding_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Record real funding payment deduplicated by transaction ID (TASK L-2).
        Supports both direct amount tracking and transaction deduplication.
        """
        eff_amount = amount if amount is not None else payment_amount
        ts = timestamp if timestamp is not None else time.time()
        fid = funding_id or f"funding_{exchange}_{symbol}_{position_side}_{ts}_{eff_amount}"

        if fid in self.processed_funding_ids:
            return False
        self.processed_funding_ids.add(fid)
        if len(self.processed_funding_ids) > 2000:
            self.processed_funding_ids = set(list(self.processed_funding_ids)[-1000:])

        sym = self._canon(symbol)
        if sym in self.positions:
            pos = self.positions[sym]
            ex = exchange.lower()
            side_upper = str(position_side).upper()
            if pos.long_leg.exchange == ex and (side_upper == "LONG" or pos.short_leg.exchange != ex):
                pos.long_leg.funding_accrued += eff_amount
                pos.long_leg.last_updated = ts
            elif pos.short_leg.exchange == ex and (side_upper == "SHORT" or pos.long_leg.exchange != ex):
                pos.short_leg.funding_accrued += eff_amount
                pos.short_leg.last_updated = ts
            else:
                pos.total_funding_accrued += eff_amount
            pos.recalculate()

        self.realized_funding_pnl += eff_amount
        logger.info(f"[{exchange.upper()}][{sym}] Recorded funding payment: ${eff_amount:+.4f} (ID: {fid})")
        return True
