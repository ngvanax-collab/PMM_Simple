"""Triple Barrier & Dynamic Trailing TP Executor per positionSide with Virtual Local SL & Passive Exit."""
import asyncio
import time
from typing import Any, Dict, List, Optional
from loguru import logger

from app.core.gateway import ExchangeGateway
from app.core.position_tracker import PositionTracker
from app.core.quoter import PMMQuoter
from app.models.config import PairConfig
from app.models.state import (
    ExecutorBarrierState,
    FillRecord,
    OrderPurpose,
    OrderRecord,
    OrderSide,
    OrderStatus,
    OrderType,
    PnLRecord,
    PositionSide,
)
from app.persistence.db import db


class TripleBarrierExecutor:
    """Manages Take Profit, Virtual Local Stop Loss, Dynamic Trailing TP, and Passive Maker Time-Limit Exit."""

    def __init__(
        self,
        config: PairConfig,
        position_side: PositionSide,
        gateway: ExchangeGateway,
        position_tracker: PositionTracker,
        quoter: PMMQuoter,
    ):
        self.config = config
        self.position_side = position_side
        self.symbol = config.symbol
        self.gateway = gateway
        self.tracker = position_tracker
        self.quoter = quoter

        self.state = ExecutorBarrierState(
            symbol=self.symbol,
            position_side=self.position_side,
            active=False,
        )
        self._lock = asyncio.Lock()
        self._is_exiting = False

    @property
    def exit_side(self) -> OrderSide:
        """Exit order trade side: Closing LONG = SELL, Closing SHORT = BUY."""
        return OrderSide.SELL if self.position_side == PositionSide.LONG else OrderSide.BUY

    async def on_entry_fill(self, fill: FillRecord) -> None:
        """
        Triggered when an entry quote order fills (increasing position size).
        Arms Virtual Local SL, resets Trailing TP/Passive Exit tracking, and places TP orders.
        NOTE: Does NOT place STOP_MARKET orders on exchange (eliminates Binance -4045).
        """
        async with self._lock:
            pos_state = self.tracker.get_state(self.position_side)
            if pos_state.amount <= 1e-7:
                return

            self.state.active = True
            self._is_exiting = False
            self.state.entry_price = pos_state.entry_price
            self.state.total_qty = pos_state.amount
            self.state.remaining_qty = pos_state.amount
            if self.state.pyramid_filled_count == 0 or self.state.initial_entry_price <= 0:
                self.state.initial_entry_price = pos_state.entry_price
                self.state.initial_qty = pos_state.amount
            self.state.entry_timestamp = time.time()
            self.state.last_update_time = time.time()

            # Reset Trailing TP & Passive Exit tracking
            self.state.trailing_active = False
            self.state.trailing_high_watermark = 0.0
            self.state.trailing_tp_active = False
            self.state.peak_price = 0.0
            self.state.trough_price = float('inf')
            self.state.passive_exit_active = False
            self.state.passive_exit_order_id = None
            self.state.passive_exit_start_time = 0.0

            # Calculate Virtual Local Stop Loss trigger price
            if not self.state.is_guaranteed_sl_locked:
                if self.position_side == PositionSide.LONG:
                    raw_sl_price = self.state.entry_price * (1.0 - self.config.stop_loss)
                else:
                    raw_sl_price = self.state.entry_price * (1.0 + self.config.stop_loss)
                self.state.sl_price = self.quoter.quantize_price(raw_sl_price)
            self.state.sl_qty = self.quoter.quantize_amount(self.state.remaining_qty)

            logger.info(
                f"[{self.symbol}][{self.position_side.value}][VIRTUAL_SL] Barrier initialized: "
                f"size={self.state.remaining_qty:.4f} @ entry={self.state.entry_price:.4f} | "
                f"Virtual SL trigger={self.state.sl_price:.4f}"
            )

            await self._place_take_profit_orders()

    async def reconcile_barrier(self, current_mark_price: float = 0.0) -> None:
        """
        Reconcile and restore barrier protection on bot startup or periodic exchange reconcile.
        1. If position == 0: Clean up any stale barrier orders and deactivate barrier.
        2. If position > 0:
           - Arm barrier (active = True, entry_price, total_qty, remaining_qty).
           - Check overflow / breached state against mark price:
             * If Virtual SL breached: Execute emergency MARKET exit with purpose=STOP_LOSS.
             * If TP breached: Execute immediate MARKET exit with purpose=TAKE_PROFIT.
           - If within normal bounds: Synchronize Take Profit (LIMIT_MAKER) orders on exchange.
        """
        async with self._lock:
            pos_state = self.tracker.get_state(self.position_side)
            if pos_state.amount <= 1e-7:
                if self.state.active or self.state.tp_orders or self.state.sl_order_id or self.state.passive_exit_active:
                    logger.info(f"[{self.symbol}][{self.position_side.value}] Reconcile: No open position. Cleaning up barrier orders.")
                    await self._cleanup_all_barrier_orders()
                    self.state.active = False
                    self._is_exiting = False
                return

            # Position exists on exchange -> Arm barrier state
            self.state.active = True
            self._is_exiting = False
            self.state.entry_price = pos_state.entry_price
            self.state.total_qty = pos_state.amount
            self.state.remaining_qty = pos_state.amount
            if self.state.entry_timestamp == 0.0:
                self.state.entry_timestamp = time.time()
            self.state.last_update_time = time.time()

            # Calculate Virtual SL trigger price
            if self.position_side == PositionSide.LONG:
                raw_sl_price = self.state.entry_price * (1.0 - self.config.stop_loss)
            else:
                raw_sl_price = self.state.entry_price * (1.0 + self.config.stop_loss)
            self.state.sl_price = self.quoter.quantize_price(raw_sl_price)
            self.state.sl_qty = self.quoter.quantize_amount(self.state.remaining_qty)

            logger.info(
                f"[{self.symbol}][{self.position_side.value}][VIRTUAL_SL] Reconcile Barrier Auto-Arming: "
                f"size={self.state.remaining_qty:.4f} @ entry={self.state.entry_price:.4f} | "
                f"Virtual SL trigger={self.state.sl_price:.4f} | Mark={current_mark_price:.4f}"
            )

            # Check overflow / breached conditions against current mark price
            if current_mark_price > 0 and self.state.entry_price > 0:
                # 1. Virtual Stop Loss breach check (Emergency market close)
                if self.position_side == PositionSide.LONG:
                    if current_mark_price <= self.state.sl_price:
                        logger.critical(
                            f"[{self.symbol}][LONG][VIRTUAL_SL] Reconcile detected Mark Price {current_mark_price:.4f} <= SL trigger {self.state.sl_price:.4f}. "
                            f"Executing emergency market exit for STOP_LOSS."
                        )
                        await self._execute_market_exit(purpose=OrderPurpose.STOP_LOSS)
                        return
                else:  # SHORT
                    if current_mark_price >= self.state.sl_price:
                        logger.critical(
                            f"[{self.symbol}][SHORT][VIRTUAL_SL] Reconcile detected Mark Price {current_mark_price:.4f} >= SL trigger {self.state.sl_price:.4f}. "
                            f"Executing emergency market exit for STOP_LOSS."
                        )
                        await self._execute_market_exit(purpose=OrderPurpose.STOP_LOSS)
                        return

                # 2. Take Profit breach check (Immediate profit-taking market close on offline startup overflow)
                tp_pct = self.config.tp_levels[0][0] if self.config.tp_levels else self.config.take_profit
                has_active_tp_orders = any(tp.get("status") == "OPEN" for tp in self.state.tp_orders)
                if not has_active_tp_orders:
                    if self.position_side == PositionSide.LONG:
                        tp_price = self.state.entry_price * (1.0 + tp_pct)
                        if current_mark_price >= tp_price:
                            logger.info(
                                f"[{self.symbol}][LONG] Reconcile detected Mark Price {current_mark_price:.4f} >= TP price {tp_price:.4f} with no active TP orders. "
                                f"Executing immediate market exit for TAKE_PROFIT."
                            )
                            await self._execute_market_exit(purpose=OrderPurpose.TAKE_PROFIT)
                            return
                    else:  # SHORT
                        tp_price = self.state.entry_price * (1.0 - tp_pct)
                        if current_mark_price <= tp_price:
                            logger.info(
                                f"[{self.symbol}][SHORT] Reconcile detected Mark Price {current_mark_price:.4f} <= TP price {tp_price:.4f} with no active TP orders. "
                                f"Executing immediate market exit for TAKE_PROFIT."
                            )
                            await self._execute_market_exit(purpose=OrderPurpose.TAKE_PROFIT)
                            return

            # Normal in-range position: place/sync TP orders
            await self._place_take_profit_orders()

    async def on_exit_fill(self, fill: FillRecord) -> None:
        """
        Triggered when an exit order fills (partial/full TP, SL, or Passive Exit).
        Updates remaining position size, records PnL, and manages progressive cooldown.
        """
        async with self._lock:
            self.state.remaining_qty = max(0.0, self.state.remaining_qty - fill.amount)
            self.state.last_update_time = time.time()
            self.state.sl_qty = self.quoter.quantize_amount(self.state.remaining_qty)

            # Record PnL in database
            pnl_rec = PnLRecord(
                symbol=self.symbol,
                position_side=self.position_side,
                realized_pnl=fill.realized_pnl,
                fee=fill.fee,
                net_pnl=fill.realized_pnl - fill.fee,
                timestamp=fill.timestamp,
                note=f"Exit Fill: {fill.amount:.4f} @ {fill.price:.2f}",
            )
            try:
                await db.record_pnl(pnl_rec)
            except Exception as e:
                logger.warning(f"[{self.symbol}][{self.position_side.value}] Non-fatal DB record_pnl error: {e}")

            # Check if this exit fill was from the Stop Loss or Passive Exit
            cl_id = (fill.client_order_id or "").lower()
            is_sl_fill = (
                (self.state.sl_order_id is not None and fill.order_id == self.state.sl_order_id) or
                "sl_" in cl_id or "exit_stop_loss" in cl_id
            )
            is_passive_exit_fill = (
                self.state.passive_exit_order_id is not None and fill.order_id == self.state.passive_exit_order_id
            ) or "pe_" in cl_id or "pexit_" in cl_id

            if is_passive_exit_fill:
                logger.info(
                    f"[{self.symbol}][{self.position_side.value}][PASSIVE_EXIT] Passive Maker Exit filled! "
                    f"Net Realized PnL: ${fill.realized_pnl - fill.fee:+.4f}"
                )

            if is_sl_fill:
                self.tracker.set_sl_cooldown(self.position_side)
            elif fill.realized_pnl > 0 or "tp_" in cl_id or "trailing" in cl_id:
                self.tracker.reset_sl_cooldown(self.position_side)

            # Update status of filled TP order if applicable
            for tp_info in self.state.tp_orders:
                if tp_info.get("order_id") == fill.order_id or tp_info.get("client_order_id") == fill.client_order_id:
                    tp_info["status"] = "FILLED"

            if self.state.remaining_qty <= 1e-7:
                # Position fully closed -> Reset all barrier state
                logger.info(f"[{self.symbol}][{self.position_side.value}] Position completely closed. Resetting barrier.")
                await self._cleanup_all_barrier_orders()
                self.state.active = False
                self._is_exiting = False
                self.state.trailing_tp_active = False
                self.state.peak_price = 0.0
                self.state.trough_price = float('inf')
                self.state.passive_exit_active = False
                self.state.passive_exit_order_id = None
                self.state.passive_exit_start_time = 0.0
                self.state.pyramid_filled_count = 0
                self.state.is_guaranteed_sl_locked = False
                self.state.initial_entry_price = 0.0
                self.state.initial_qty = 0.0
                pos_state = self.tracker.get_state(self.position_side)
                pos_state.pyramid_filled_count = 0
                pos_state.is_guaranteed_sl_locked = False
                return

            # Partial exit: Re-place TP orders for remaining quantity to protect position
            logger.info(
                f"[{self.symbol}][{self.position_side.value}] Partial exit fill: remaining={self.state.remaining_qty:.4f}. Re-placing TP orders."
            )
            if self.state.active and not self._is_exiting:
                await self._place_take_profit_orders()

    async def _place_take_profit_orders(self) -> None:
        """Place multi-level Take Profit orders with inventory skew boost."""
        # Cancel any previous TP orders before placing new ones
        for tp_info in self.state.tp_orders:
            order_id = tp_info.get("order_id")
            if order_id and tp_info.get("status") != "FILLED":
                await self.gateway.cancel_order(self.symbol, order_id)
        self.state.tp_orders.clear()

        entry_p = self.state.entry_price
        total_qty = self.state.remaining_qty
        if entry_p <= 0 or total_qty <= 0:
            return

        # Calculate TP price levels
        tp_levels = self.config.tp_levels if self.config.tp_levels else [[self.config.take_profit, 1.0]]

        # Apply tp_skew_boost based on inventory ratio rho
        pos_val = total_qty * entry_p
        max_pos = self.config.max_long_usdt if self.position_side == PositionSide.LONG else self.config.max_short_usdt
        rho = min(1.0, max(0.0, pos_val / max(1.0, max_pos))) if self.config.inventory_skew_enabled else 0.0
        skew_discount = 1.0 - (self.config.tp_skew_boost * rho)
        skew_discount = max(0.5, min(1.0, skew_discount))

        spread_floor = self.quoter.calculate_spread_floor(entry_p)
        min_eff_tp = max(self.config.minimum_spread, spread_floor + 0.0005)

        for idx, (tp_pct, fraction) in enumerate(tp_levels):
            eff_tp_pct = max(min_eff_tp, tp_pct * skew_discount)
            qty = self.quoter.quantize_amount(total_qty * fraction)
            if qty <= 0:
                continue

            if self.position_side == PositionSide.LONG:
                raw_tp_price = entry_p * (1.0 + eff_tp_pct)
            else:
                raw_tp_price = entry_p * (1.0 - eff_tp_pct)

            # TP is an exit order — quantize in the safe direction for Post-Only maker
            is_tp_ask = (self.position_side == PositionSide.LONG)
            tp_price = self.quoter.quantize_price(raw_tp_price, is_bid=not is_tp_ask)
            client_id = f"tp_{self.position_side.value.lower()}_{idx}_{int(time.time()*1000)}"

            order_resp = await self.gateway.create_exit_order(
                symbol=self.symbol,
                side=self.exit_side,
                position_side=self.position_side,
                order_type=OrderType.LIMIT_MAKER if self.config.take_profit_order_type == "LIMIT_MAKER" else OrderType.LIMIT,
                amount=qty,
                price=tp_price,
                client_order_id=client_id,
                purpose=OrderPurpose.TAKE_PROFIT,
            )

            if order_resp:
                order_id = str(order_resp.get("id") or client_id)
                self.state.tp_orders.append({
                    "order_id": order_id,
                    "client_order_id": client_id,
                    "price": tp_price,
                    "qty": qty,
                    "level": idx,
                    "status": "OPEN",
                })
                logger.info(
                    f"[{self.symbol}][{self.position_side.value}] TP #{idx} placed: {qty:.4f} @ {tp_price:.4f} (order_id={order_id})"
                )

    async def check_favorable_momentum_pyramid(
        self,
        current_price: float,
        market_state: Optional[Any] = None,
    ) -> bool:
        """
        Check & Execute Favorable Momentum Pyramiding + Guaranteed Profit SL:
        1. Conditions:
           - favorable_pyramiding_enabled is True.
           - Position active and remaining_qty > 0.
           - pyramid_filled_count == 0.
           - uPnL >= trailing_tp_activation_pct.
           - 60s momentum in favorable direction >= pyramiding_trigger_natr_mult * NATR_15m.
        2. Actions:
           - Send MARKET entry order for 50% initial size.
           - Calculate new average entry price P_avg_new.
           - Move Virtual SL to Guaranteed Profit (Lock Breakeven/Profit).
           - Tighten trailing_tp_callback_pct to 20% of TP_base.
           - Synchronize Take Profit orders.
        """
        if not getattr(self.config, "favorable_pyramiding_enabled", True):
            return False

        if not self.state.active or self.state.remaining_qty <= 1e-7 or self._is_exiting or self.state.pyramid_filled_count > 0:
            return False

        if self.state.entry_price <= 0 or current_price <= 0:
            return False

        act_pct = getattr(self.config, "trailing_tp_activation_pct", 0.008)
        if self.position_side == PositionSide.LONG:
            upnl_pct = (current_price - self.state.entry_price) / self.state.entry_price
        else:
            upnl_pct = (self.state.entry_price - current_price) / self.state.entry_price

        if upnl_pct < act_pct:
            return False

        # Check 60s micro-momentum in favorable direction
        natr_15m = 0.012
        surge_up, plunge_down = 0.0, 0.0
        if market_state is not None:
            natr_15m = getattr(market_state, "current_natr_15m", 0.012)
            if hasattr(market_state, "get_favorable_momentum_60s"):
                surge_up, plunge_down = market_state.get_favorable_momentum_60s()

        mult = getattr(self.config, "pyramiding_trigger_natr_mult", 0.65)
        threshold = mult * natr_15m

        favorable_momentum = surge_up if self.position_side == PositionSide.LONG else plunge_down
        if favorable_momentum < threshold:
            return False

        pyr_pct = getattr(self.config, "pyramiding_size_pct", 0.50)
        init_qty = self.state.initial_qty if self.state.initial_qty > 0 else self.state.total_qty
        init_p = self.state.initial_entry_price if self.state.initial_entry_price > 0 else self.state.entry_price
        if init_qty <= 0:
            init_qty = self.state.remaining_qty
        if init_p <= 0:
            init_p = self.state.entry_price

        pyr_qty = self.quoter.quantize_amount(init_qty * pyr_pct)
        if pyr_qty <= 0:
            return False

        entry_side = OrderSide.BUY if self.position_side == PositionSide.LONG else OrderSide.SELL
        client_id = f"q_pyr_{self.position_side.value.lower()}_{int(time.time()*1000)}"

        logger.info(
            f"[{self.symbol}][{self.position_side.value}][MOMENTUM_PYRAMID] Favorable Momentum Pyramid triggered: "
            f"uPnL={upnl_pct*100:.2f}% >= {act_pct*100:.2f}%, Momentum={favorable_momentum*100:.2f}% >= {threshold*100:.2f}%. "
            f"Adding {pyr_qty:.4f} {self.position_side.value} via Market entry..."
        )

        resp = await self.gateway.create_entry_market_order(
            symbol=self.symbol,
            side=entry_side,
            position_side=self.position_side,
            amount=pyr_qty,
            client_order_id=client_id,
        )

        if not resp:
            logger.error(f"[{self.symbol}][{self.position_side.value}][MOMENTUM_PYRAMID] Failed to place pyramid entry order!")
            return False

        # Update position size and average price
        self.state.pyramid_filled_count = 1
        old_qty = self.state.remaining_qty
        old_entry = self.state.entry_price
        new_total_qty = self.quoter.quantize_amount(old_qty + pyr_qty)
        new_avg_price = (old_qty * old_entry + pyr_qty * current_price) / new_total_qty

        self.state.entry_price = new_avg_price
        self.state.total_qty = new_total_qty
        self.state.remaining_qty = new_total_qty
        self.state.sl_qty = new_total_qty

        pos_state = self.tracker.get_state(self.position_side)
        pos_state.amount = new_total_qty
        pos_state.entry_price = new_avg_price
        pos_state.pyramid_filled_count = 1

        # Move Stop Loss to Guaranteed Profit Stop Loss
        s_floor = self.quoter.calculate_spread_floor(new_avg_price)
        if self.position_side == PositionSide.LONG:
            raw_sl = max(new_avg_price * (1.0 + s_floor), init_p * 1.0035)
        else:
            raw_sl = min(new_avg_price * (1.0 - s_floor), init_p * 0.9965)

        self.state.sl_price = self.quoter.quantize_price(raw_sl)
        self.state.is_guaranteed_sl_locked = True
        pos_state.is_guaranteed_sl_locked = True

        # Tighten Trailing TP Callback
        tp_base = getattr(self.config, "take_profit", 0.008)
        tight_cb = max(0.0020, 0.20 * tp_base)
        self.config.trailing_tp_callback_pct = tight_cb

        logger.info(
            f"[{self.symbol}][{self.position_side.value}][MOMENTUM_PYRAMID] Pyramid executed: +{pyr_qty:.4f} -> total={new_total_qty:.4f} @ new_avg={new_avg_price:.4f}"
        )
        logger.info(
            f"[{self.symbol}][{self.position_side.value}][GUARANTEED_SL] Stop Loss locked at {self.state.sl_price:.4f} "
            f"(Initial Entry={init_p:.4f}, New Avg={new_avg_price:.4f}, s_floor={s_floor*100:.3f}%). Trailing callback tightened to {tight_cb*100:.2f}%."
        )

        # Synchronize / re-place Take Profit orders with updated quantity
        await self._place_take_profit_orders()
        return True

    async def check_runtime_barriers(
        self,
        current_price: float,
        best_bid: float = 0.0,
        best_ask: float = 0.0,
        market_state: Optional[Any] = None,
    ) -> None:
        """
        Check Virtual Local Stop Loss, Guaranteed Profit SL, Minimum Holding Lock,
        Favorable Momentum Pyramiding, Dynamic Trailing TP, and Passive Exit on every price tick.
        """
        if not self.state.active or self.state.remaining_qty <= 1e-7 or self._is_exiting or current_price <= 0:
            return

        async with self._lock:
            if not self.state.active or self.state.remaining_qty <= 1e-7 or self._is_exiting:
                return

            now = time.time()
            elapsed = now - self.state.entry_timestamp

            # ── 1. Virtual Local Stop Loss / Guaranteed Profit SL (Highest Priority) ──
            if self.position_side == PositionSide.LONG:
                if self.state.is_guaranteed_sl_locked and self.state.sl_price > 0:
                    sl_trigger = self.state.sl_price
                else:
                    sl_trigger = self.state.sl_price if self.state.sl_price > 0 else self.state.entry_price * (1.0 - self.config.stop_loss)

                if current_price <= sl_trigger:
                    tag = "GUARANTEED_SL" if self.state.is_guaranteed_sl_locked else "VIRTUAL_SL"
                    logger.critical(
                        f"[{self.symbol}][LONG][{tag}] Mark price {current_price:.4f} <= SL trigger {sl_trigger:.4f}. "
                        f"Executing emergency market exit."
                    )
                    await self._execute_market_exit(purpose=OrderPurpose.STOP_LOSS)
                    return
            else:  # SHORT
                if self.state.is_guaranteed_sl_locked and self.state.sl_price > 0:
                    sl_trigger = self.state.sl_price
                else:
                    sl_trigger = self.state.sl_price if self.state.sl_price > 0 else self.state.entry_price * (1.0 + self.config.stop_loss)

                if current_price >= sl_trigger:
                    tag = "GUARANTEED_SL" if self.state.is_guaranteed_sl_locked else "VIRTUAL_SL"
                    logger.critical(
                        f"[{self.symbol}][SHORT][{tag}] Mark price {current_price:.4f} >= SL trigger {sl_trigger:.4f}. "
                        f"Executing emergency market exit."
                    )
                    await self._execute_market_exit(purpose=OrderPurpose.STOP_LOSS)
                    return

            # ── 2. Minimum Holding Lock (Bypass non-SL exits if within lock window) ──
            min_hold = getattr(self.config, "min_holding_sec", 3.0)
            if elapsed < min_hold:
                return

            # ── 3. Favorable Momentum Pyramiding Check ──
            await self.check_favorable_momentum_pyramid(current_price, market_state=market_state)

            # ── 4. Dynamic Trailing Take Profit (Floating Profit Capture) ──
            trailing_tp_enabled = getattr(self.config, "trailing_tp_enabled", True)
            if trailing_tp_enabled and self.state.entry_price > 0:
                act_pct = getattr(self.config, "trailing_tp_activation_pct", 0.008)
                cb_pct = getattr(self.config, "trailing_tp_callback_pct", 0.003)

                if self.position_side == PositionSide.LONG:
                    act_price = self.state.entry_price * (1.0 + act_pct)
                    if not self.state.trailing_tp_active and current_price >= act_price:
                        self.state.trailing_tp_active = True
                        self.state.peak_price = current_price
                        logger.info(
                            f"[{self.symbol}][LONG][TRAILING_TP_ACTIVE] Trailing TP activated at price {current_price:.4f} (>= {act_price:.4f})"
                        )

                    if self.state.trailing_tp_active:
                        self.state.peak_price = max(self.state.peak_price, current_price)
                        trail_tp_trigger = self.state.peak_price * (1.0 - cb_pct)
                        if current_price <= trail_tp_trigger:
                            logger.info(
                                f"[{self.symbol}][LONG][TRAILING_TP_ACTIVE] Trailing TP triggered! "
                                f"Peak={self.state.peak_price:.4f}, Current={current_price:.4f} <= Trigger={trail_tp_trigger:.4f}. "
                                f"Executing Take Profit exit."
                            )
                            await self._execute_market_exit(purpose=OrderPurpose.TRAILING_TAKE_PROFIT)
                            return

                else:  # SHORT
                    act_price = self.state.entry_price * (1.0 - act_pct)
                    if not self.state.trailing_tp_active and current_price <= act_price:
                        self.state.trailing_tp_active = True
                        self.state.trough_price = current_price
                        logger.info(
                            f"[{self.symbol}][SHORT][TRAILING_TP_ACTIVE] Trailing TP activated at price {current_price:.4f} (<= {act_price:.4f})"
                        )

                    if self.state.trailing_tp_active:
                        if self.state.trough_price == float('inf') or self.state.trough_price <= 0 or current_price < self.state.trough_price:
                            self.state.trough_price = current_price
                        trail_tp_trigger = self.state.trough_price * (1.0 + cb_pct)
                        if current_price >= trail_tp_trigger:
                            logger.info(
                                f"[{self.symbol}][SHORT][TRAILING_TP_ACTIVE] Trailing TP triggered! "
                                f"Trough={self.state.trough_price:.4f}, Current={current_price:.4f} >= Trigger={trail_tp_trigger:.4f}. "
                                f"Executing Take Profit exit."
                            )
                            await self._execute_market_exit(purpose=OrderPurpose.TRAILING_TAKE_PROFIT)
                            return

            # ── 4. Legacy Trailing Stop Barrier (if enabled separately) ──
            trailing_cfg = self.config.trailing_stop
            if trailing_cfg and self.state.entry_price > 0:
                if self.position_side == PositionSide.LONG:
                    act_price = self.state.entry_price * (1.0 + trailing_cfg.activation_price)
                    if not self.state.trailing_active and current_price >= act_price:
                        self.state.trailing_active = True
                        self.state.trailing_high_watermark = current_price
                        logger.info(
                            f"[{self.symbol}][LONG] Trailing stop activated at price {current_price:.4f} (>= {act_price:.4f})"
                        )

                    if self.state.trailing_active:
                        if current_price > self.state.trailing_high_watermark:
                            self.state.trailing_high_watermark = current_price
                        trail_stop_price = self.state.trailing_high_watermark * (1.0 - trailing_cfg.trailing_delta)
                        if current_price <= trail_stop_price:
                            logger.info(
                                f"[{self.symbol}][LONG] Trailing stop triggered! High={self.state.trailing_high_watermark:.4f}, Current={current_price:.4f} <= {trail_stop_price:.4f}"
                            )
                            await self._execute_market_exit(purpose=OrderPurpose.TRAILING_STOP)
                            return

                elif self.position_side == PositionSide.SHORT:
                    act_price = self.state.entry_price * (1.0 - trailing_cfg.activation_price)
                    if not self.state.trailing_active and current_price <= act_price:
                        self.state.trailing_active = True
                        self.state.trailing_high_watermark = current_price
                        logger.info(
                            f"[{self.symbol}][SHORT] Trailing stop activated at price {current_price:.4f} (<= {act_price:.4f})"
                        )

                    if self.state.trailing_active:
                        if self.state.trailing_high_watermark <= 0 or current_price < self.state.trailing_high_watermark:
                            self.state.trailing_high_watermark = current_price
                        trail_stop_price = self.state.trailing_high_watermark * (1.0 + trailing_cfg.trailing_delta)
                        if current_price >= trail_stop_price:
                            logger.info(
                                f"[{self.symbol}][SHORT] Trailing stop triggered! Low={self.state.trailing_high_watermark:.4f}, Current={current_price:.4f} >= {trail_stop_price:.4f}"
                            )
                            await self._execute_market_exit(purpose=OrderPurpose.TRAILING_STOP)
                            return

            # ── 5. Passive Maker Time-Limit Exit ──
            if self.config.time_limit > 0 and elapsed >= self.config.time_limit:
                await self._handle_passive_time_limit_exit(current_price, best_bid, best_ask)

    async def _handle_passive_time_limit_exit(
        self,
        current_price: float,
        best_bid: float = 0.0,
        best_ask: float = 0.0,
    ) -> None:
        """
        Handle time-limit exit via Post-Only Passive Maker order near breakeven/market.
        Avoids paying expensive Taker fees on aged positions.
        """
        now = time.time()
        timeout_sec = getattr(self.config, "passive_exit_timeout_sec", 120.0)
        offset_pct = getattr(self.config, "passive_exit_spread_pct", 0.0006)
        qty = self.quoter.quantize_amount(self.state.remaining_qty)

        if qty <= 0 or self.state.entry_price <= 0:
            return

        if self.state.passive_exit_active:
            # Order is already placed; check if timeout expired to refresh price
            if now - self.state.passive_exit_start_time >= timeout_sec:
                if self.state.passive_exit_order_id:
                    await self.gateway.cancel_order(self.symbol, self.state.passive_exit_order_id)
                    self.state.passive_exit_order_id = None
                self.state.passive_exit_active = False
            else:
                return

        # Cancel open Take Profit orders before placing passive exit
        for tp_info in self.state.tp_orders:
            order_id = tp_info.get("order_id")
            if order_id and tp_info.get("status") != "FILLED":
                await self.gateway.cancel_order(self.symbol, order_id)
        self.state.tp_orders.clear()

        # Calculate near-market Post-Only exit price
        if self.position_side == PositionSide.LONG:
            # LONG exit = SELL: target >= entry * (1 + offset_pct) or best_ask
            target_p = max(self.state.entry_price * (1.0 + offset_pct), best_ask if best_ask > 0 else current_price)
            exit_price = self.quoter.quantize_price(target_p, is_bid=False)
        else:
            # SHORT exit = BUY: target <= entry * (1 - offset_pct) or best_bid
            target_p = min(self.state.entry_price * (1.0 - offset_pct), best_bid if best_bid > 0 else current_price)
            exit_price = self.quoter.quantize_price(target_p, is_bid=True)

        client_id = f"pe_{self.position_side.value.lower()}_{int(now*1000)}"
        resp = await self.gateway.create_exit_order(
            symbol=self.symbol,
            side=self.exit_side,
            position_side=self.position_side,
            order_type=OrderType.LIMIT_MAKER,
            amount=qty,
            price=exit_price,
            client_order_id=client_id,
            purpose=OrderPurpose.PASSIVE_TIME_LIMIT_EXIT,
        )

        if resp:
            self.state.passive_exit_active = True
            self.state.passive_exit_order_id = str(resp.get("id") or client_id)
            self.state.passive_exit_start_time = now
            logger.info(
                f"[{self.symbol}][{self.position_side.value}][PASSIVE_EXIT] Passive Maker Exit placed: "
                f"{qty:.4f} @ {exit_price:.4f} (order_id={self.state.passive_exit_order_id})"
            )
        else:
            logger.warning(
                f"[{self.symbol}][{self.position_side.value}][PASSIVE_EXIT] Failed to place passive exit order. "
                f"Will retry on next tick."
            )

    async def _execute_market_exit(self, purpose: OrderPurpose) -> None:
        """Emergency or conditional market exit: cancels existing TP/Passive orders and executes MARKET exit."""
        pos_qty = self.quoter.quantize_amount(self.state.remaining_qty)
        if pos_qty <= 0:
            return

        # Mark exiting and deactivate barrier immediately to prevent duplicate firing across price ticks
        self._is_exiting = True
        self.state.active = False

        # Cancel TP, SL, and Passive exit orders
        await self._cleanup_all_barrier_orders()

        if purpose == OrderPurpose.STOP_LOSS:
            client_id = f"sl_{self.position_side.value.lower()}_{int(time.time()*1000)}"
        elif purpose in (OrderPurpose.TAKE_PROFIT, OrderPurpose.TRAILING_TAKE_PROFIT):
            client_id = f"tp_{self.position_side.value.lower()}_{int(time.time()*1000)}"
        else:
            client_id = f"exit_{purpose.value.lower()}_{self.position_side.value.lower()}_{int(time.time()*1000)}"

        resp = await self.gateway.create_exit_order(
            symbol=self.symbol,
            side=self.exit_side,
            position_side=self.position_side,
            order_type=OrderType.MARKET,
            amount=pos_qty,
            client_order_id=client_id,
            purpose=purpose,
        )

        if resp:
            logger.info(
                f"[{self.symbol}][{self.position_side.value}] Market exit executed ({purpose.value}): qty={pos_qty:.4f}"
            )
            if purpose == OrderPurpose.STOP_LOSS:
                self.tracker.set_sl_cooldown(self.position_side)
        else:
            logger.critical(
                f"[{self.symbol}][{self.position_side.value}] Failed to execute market exit for {purpose.value}!"
            )
            self._is_exiting = False
            self.state.active = True

    async def _cleanup_all_barrier_orders(self) -> None:
        """Cancel all active TP orders, passive exit order, and any legacy SL order."""
        for tp_info in self.state.tp_orders:
            order_id = tp_info.get("order_id")
            if order_id and tp_info.get("status") != "FILLED":
                await self.gateway.cancel_order(self.symbol, order_id)
        self.state.tp_orders.clear()

        if self.state.passive_exit_order_id:
            await self.gateway.cancel_order(self.symbol, self.state.passive_exit_order_id)
            self.state.passive_exit_order_id = None
            self.state.passive_exit_active = False

        if self.state.sl_order_id:
            await self.gateway.cancel_order(self.symbol, self.state.sl_order_id)
            self.state.sl_order_id = None
