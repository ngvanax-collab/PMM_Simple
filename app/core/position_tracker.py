import asyncio
from collections import deque
import time
from typing import Deque, Optional, Set, Tuple
from loguru import logger

from app.core.gateway import ExchangeGateway
from app.models.config import PairConfig
from app.models.state import FillRecord, OrderSide, PositionSide, SidePositionState


class PositionTracker:
    """Tracks independent LONG and SHORT position states for a single symbol."""

    def __init__(self, config: PairConfig, gateway: ExchangeGateway):
        self.config = config
        self.symbol = config.symbol
        self.gateway = gateway

        self.long_pos = SidePositionState(
            symbol=self.symbol,
            position_side=PositionSide.LONG,
            amount=0.0,
            entry_price=0.0,
            leverage=config.leverage,
        )
        self.short_pos = SidePositionState(
            symbol=self.symbol,
            position_side=PositionSide.SHORT,
            amount=0.0,
            entry_price=0.0,
            leverage=config.leverage,
        )
        self._lock = asyncio.Lock()
        self.last_reconcile_time = 0.0
        self._processed_fill_ids: Set[str] = set()
        self._processed_fill_order: Deque[str] = deque(maxlen=10000)

    @property
    def gross_exposure_usdt(self) -> float:
        """Total gross exposure in USDT = LONG notional + SHORT notional."""
        return self.long_pos.notional + self.short_pos.notional

    def get_state(self, position_side: PositionSide) -> SidePositionState:
        """Get state for specified position side."""
        return self.long_pos if position_side == PositionSide.LONG else self.short_pos

    def is_in_cooldown(self, position_side: PositionSide) -> bool:
        """Check if specified position side is currently in post-SL progressive cooldown."""
        state = self.get_state(position_side)
        now = time.time()
        in_cd = (now < state.cooldown_until)
        state.in_cooldown = in_cd
        return in_cd

    def get_cooldown_info(self, position_side: PositionSide) -> Tuple[bool, int, int]:
        """Get (is_in_cooldown, consecutive_sl_count, remaining_seconds) for position side."""
        state = self.get_state(position_side)
        now = time.time()
        remaining = max(0, int(state.cooldown_until - now)) if now < state.cooldown_until else 0
        is_cd = remaining > 0
        state.in_cooldown = is_cd
        return (is_cd, state.consecutive_sl_count, remaining)

    def set_sl_cooldown(self, position_side: PositionSide) -> None:
        """Mark position side as having triggered a stop loss with progressive exponential cooldown."""
        state = self.get_state(position_side)
        state.consecutive_sl_count += 1

        base = getattr(self.config, "base_cooldown_sec", self.config.cooldown_time)
        mult = getattr(self.config, "cooldown_multiplier", 2.0)
        max_cd = getattr(self.config, "max_cooldown_sec", 86400)

        duration = min(base * (mult ** (state.consecutive_sl_count - 1)), max_cd)
        now = time.time()
        state.last_sl_time = now
        state.cooldown_until = now + duration
        state.in_cooldown = True
        logger.warning(
            f"[{self.symbol}][{position_side.value}][SIDE_COOLDOWN] SL triggered (#{state.consecutive_sl_count}). "
            f"Entering progressive cooldown for {duration:.0f}s (until {state.cooldown_until:.0f}). Other side remains unaffected."
        )

    def reset_sl_cooldown(self, position_side: Optional[PositionSide] = None) -> None:
        """Reset progressive SL counter and active cooldown upon profitable Take Profit."""
        sides = [position_side] if position_side else [PositionSide.LONG, PositionSide.SHORT]
        for side in sides:
            st = self.get_state(side)
            if st.consecutive_sl_count > 0 or st.cooldown_until > 0:
                st.consecutive_sl_count = 0
                st.cooldown_until = 0.0
                st.in_cooldown = False
                logger.info(f"[{self.symbol}][{side.value}][SIDE_COOLDOWN] Take Profit executed with profit. Progressive Cooldown reset to base.")

    async def on_fill(self, fill: FillRecord) -> None:
        """Process incoming trade fill to update position state."""
        async with self._lock:
            # Check fill idempotency (exactly-once processing)
            if fill.id:
                if fill.id in self._processed_fill_ids:
                    logger.warning(
                        f"[{self.symbol}][IDEMPOTENCY] Duplicate fill event detected and ignored: id={fill.id}, cid={fill.client_order_id}"
                    )
                    return
                if len(self._processed_fill_order) == self._processed_fill_order.maxlen:
                    oldest_id = self._processed_fill_order.popleft()
                    self._processed_fill_ids.discard(oldest_id)
                self._processed_fill_ids.add(fill.id)
                self._processed_fill_order.append(fill.id)
            else:
                logger.debug(f"[{self.symbol}][IDEMPOTENCY] Fill with empty id received, proceeding: cid={fill.client_order_id}")

            # Defensive normalization: extract position_side from client_order_id if present
            cid = str(fill.client_order_id or "").lower()
            if "short" in cid or "q_sell" in cid:
                fill.position_side = PositionSide.SHORT
            elif "long" in cid or "q_buy" in cid:
                fill.position_side = PositionSide.LONG

            state = self.get_state(fill.position_side)
            is_entry = (
                (fill.position_side == PositionSide.LONG and fill.side == OrderSide.BUY) or
                (fill.position_side == PositionSide.SHORT and fill.side == OrderSide.SELL)
            )

            if is_entry:
                # Adding to position
                new_total_qty = state.amount + fill.amount
                if new_total_qty > 0:
                    new_entry_price = (state.amount * state.entry_price + fill.amount * fill.price) / new_total_qty
                    state.entry_price = new_entry_price
                    state.amount = new_total_qty
                state.last_fill_time = fill.timestamp
                state.filled_levels_count += 1
                state.last_level_fill_time = fill.timestamp
                level_cd = getattr(self.config, "level_cooldown_sec", 1800)
                state.next_allowed_level_time = fill.timestamp + level_cd
                logger.info(
                    f"[{self.symbol}][{fill.position_side.value}][LEVEL_COOLDOWN_ACTIVE] "
                    f"Level #{state.filled_levels_count - 1} filled (total {state.filled_levels_count} levels). "
                    f"Next level cooldown active for {level_cd}s (until {state.next_allowed_level_time:.0f})."
                )
                logger.info(f"[{self.symbol}] {fill.position_side.value} INCREASE: +{fill.amount:.4f} @ {fill.price:.2f} -> Total: {state.amount:.4f} (Avg: {state.entry_price:.2f})")
            else:
                # Exiting / reducing position
                closed_amount = min(state.amount, fill.amount) if state.amount > 0 else fill.amount
                entry_price_before_exit = state.entry_price

                # If exchange did not supply non-zero realized_pnl, calculate standard realized PnL
                if abs(fill.realized_pnl) < 1e-8 and entry_price_before_exit > 0:
                    if fill.position_side == PositionSide.LONG:
                        calc_pnl = (fill.price - entry_price_before_exit) * closed_amount
                    else:
                        calc_pnl = (entry_price_before_exit - fill.price) * closed_amount
                    fill.realized_pnl = round(calc_pnl, 4)

                old_qty = state.amount
                state.amount = max(0.0, state.amount - fill.amount)
                if state.amount <= 1e-7:
                    state.amount = 0.0
                    state.entry_price = 0.0
                    state.filled_levels_count = 0
                    state.last_level_fill_time = 0.0
                    state.next_allowed_level_time = 0.0
                    logger.info(f"[{self.symbol}][{fill.position_side.value}] Position completely flat (amount=0). Level state reset.")
                state.last_fill_time = fill.timestamp
                logger.info(
                    f"[{self.symbol}] {fill.position_side.value} DECREASE: -{fill.amount:.4f} @ {fill.price:.2f} "
                    f"-> Remaining: {state.amount:.4f} | Realized PnL: ${fill.realized_pnl:+.4f}"
                )

            self._update_notional_and_upnl(state, state.current_price or fill.price)

    def update_mark_price(self, mark_price: float) -> None:
        """Update unrealized PnL and notional from latest mark price."""
        if mark_price <= 0:
            return
        self.long_pos.current_price = mark_price
        self._update_notional_and_upnl(self.long_pos, mark_price)

        self.short_pos.current_price = mark_price
        self._update_notional_and_upnl(self.short_pos, mark_price)

    def _update_notional_and_upnl(self, state: SidePositionState, price: float) -> None:
        """Calculate notional and uPnL."""
        state.notional = state.amount * price
        if state.amount > 0 and state.entry_price > 0 and price > 0:
            if state.position_side == PositionSide.LONG:
                state.unrealized_pnl = (price - state.entry_price) * state.amount
            else:
                state.unrealized_pnl = (state.entry_price - price) * state.amount
        else:
            state.unrealized_pnl = 0.0

        state.initial_margin = state.notional / state.leverage if state.leverage > 0 else 0.0

    async def reconcile_with_exchange(self) -> bool:
        """
        Query real exchange positions and synchronize local cache (Source of Truth).
        """
        async with self._lock:
            real_long, real_short = await self.gateway.fetch_positions_hedge(self.symbol)
            if real_long is None or real_short is None:
                logger.warning(f"[{self.symbol}] Reconcile failed: exchange returned None")
                return False

            # Check for desync on LONG
            if abs(self.long_pos.amount - real_long.amount) > 1e-5:
                logger.warning(
                    f"[{self.symbol}] LONG position desync detected! Local={self.long_pos.amount:.4f}, Exchange={real_long.amount:.4f}. Syncing to exchange."
                )
                self.long_pos.amount = real_long.amount
                self.long_pos.entry_price = real_long.entry_price
                if real_long.amount <= 1e-5:
                    self.long_pos.filled_levels_count = 0
                    self.long_pos.last_level_fill_time = 0.0
                    self.long_pos.next_allowed_level_time = 0.0
                elif self.long_pos.filled_levels_count == 0:
                    self.long_pos.filled_levels_count = 1

            # Check for desync on SHORT
            if abs(self.short_pos.amount - real_short.amount) > 1e-5:
                logger.warning(
                    f"[{self.symbol}] SHORT position desync detected! Local={self.short_pos.amount:.4f}, Exchange={real_short.amount:.4f}. Syncing to exchange."
                )
                self.short_pos.amount = real_short.amount
                self.short_pos.entry_price = real_short.entry_price
                if real_short.amount <= 1e-5:
                    self.short_pos.filled_levels_count = 0
                    self.short_pos.last_level_fill_time = 0.0
                    self.short_pos.next_allowed_level_time = 0.0
                elif self.short_pos.filled_levels_count == 0:
                    self.short_pos.filled_levels_count = 1

            # Update uPnL and notional
            if real_long.current_price > 0:
                self.long_pos.current_price = real_long.current_price
                self._update_notional_and_upnl(self.long_pos, real_long.current_price)

            if real_short.current_price > 0:
                self.short_pos.current_price = real_short.current_price
                self._update_notional_and_upnl(self.short_pos, real_short.current_price)

            self.last_reconcile_time = time.time()
            return True
