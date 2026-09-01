"""PMM Quoter: Order Level Calculation, Independent Per-Side Skew & Post-Only Clamping."""
import math
import time
from typing import Any, List, NamedTuple, Optional, Tuple
from loguru import logger

from app.models.config import PairConfig
from app.models.state import OrderSide, PositionSide


class QuoteLevel(NamedTuple):
    """Calculated quote level parameters."""
    side: OrderSide
    position_side: PositionSide
    level: int
    price: float
    amount: float
    notional: float


class PMMQuoter:
    """Core Market Making Quoter with Independent Per-Side Inventory Skew."""

    def __init__(
        self,
        config: PairConfig,
        price_precision: int = 2,
        amount_precision: int = 3,
        tick_size: float = 0.01,
        step_size: float = 0.001,
        maker_fee_pct: float = 0.0002,  # 0.02% maker fee
        min_amount: float = 0.0,
        min_notional: float = 5.0,
    ):
        self.config = config
        self.price_precision = price_precision
        self.amount_precision = amount_precision
        self.tick_size = tick_size
        self.step_size = step_size
        self.maker_fee_pct = maker_fee_pct
        self.min_amount = min_amount
        self.min_notional = max(min_notional, 5.0)

    def quantize_price(self, price: float, is_bid: Optional[bool] = None) -> float:
        """
        Quantize price according to tick size:
        - Bid (is_bid=True): Floor to nearest tick size (round DOWN).
        - Ask (is_bid=False): Ceil to nearest tick size (round UP).
        - Default (is_bid=None): Nearest round.
        """
        if self.tick_size <= 0:
            return round(price, self.price_precision)
        if is_bid is True:
            ticks = math.floor(price / self.tick_size + 1e-9)
        elif is_bid is False:
            ticks = math.ceil(price / self.tick_size - 1e-9)
        else:
            ticks = round(price / self.tick_size)
        quantized = ticks * self.tick_size
        return round(quantized, self.price_precision)

    def quantize_amount(self, amount: float) -> float:
        """Floor amount to the nearest valid step size."""
        if self.step_size <= 0:
            return round(amount, self.amount_precision)
        steps = math.floor(amount / self.step_size + 1e-9)
        quantized = steps * self.step_size
        return round(quantized, self.amount_precision)

    def calculate_spread_floor(self, mid_price: float) -> float:
        """
        Fee-Aware Spread Floor:
        Ensures spread_floor >= (2 * maker_fee_pct) + min(0.0015, minimum_spread)
        Clamped at minimum 0.0025 (0.25%) so all maker fills are net positive after fees.
        """
        tick_pct = (self.tick_size / mid_price) if mid_price > 0 else 0.0001
        fee_buffer = max(0.0015, self.config.minimum_spread)
        floor = 2.0 * tick_pct + (2.0 * self.maker_fee_pct) + fee_buffer
        return max(0.0025, floor)

    def clamp_quote(
        self,
        raw_price: float,
        is_bid: bool,
        mid_price: float,
        spread_floor: float,
        best_bid: Optional[float] = None,
        best_ask: Optional[float] = None,
    ) -> Optional[float]:
        """
        Enforce strict safety rules on quote prices:
        1. Bid is floored, Ask is ceiled to tick size.
        2. Bid MUST be strictly below mid_price by at least max(spread_floor / 2 * mid, tick_size).
           If best_ask is provided, Bid MUST be <= best_ask - tick_size.
           If best_bid is provided, Bid <= best_bid.
        3. Ask MUST be strictly above mid_price by at least max(spread_floor / 2 * mid, tick_size).
           If best_bid is provided, Ask MUST be >= best_bid + tick_size.
           If best_ask is provided, Ask >= best_ask.
        4. Apply price_floor and price_ceiling bounds.
        """
        quantized = self.quantize_price(raw_price, is_bid=is_bid)
        min_half_spread = max((spread_floor / 2.0) * mid_price, self.tick_size)

        if is_bid:
            max_allowed_bid = self.quantize_price(mid_price - min_half_spread, is_bid=True)
            if max_allowed_bid >= mid_price:
                max_allowed_bid = self.quantize_price(mid_price - self.tick_size, is_bid=True)
            if best_ask is not None and best_ask > 0:
                max_allowed_bid = min(max_allowed_bid, self.quantize_price(best_ask - self.tick_size, is_bid=True))
            if best_bid is not None and best_bid > 0:
                max_allowed_bid = min(max_allowed_bid, self.quantize_price(best_bid, is_bid=True))

            if self.config.price_floor > 0 and self.config.price_floor > max_allowed_bid:
                logger.warning(
                    f"[QUOTER_WARNING] price_floor ({self.config.price_floor}) > max_allowed_bid ({max_allowed_bid}) "
                    f"for {self.config.symbol}. Clamping to max_allowed_bid and skipping quote."
                )
                return None

            clamped = min(quantized, max_allowed_bid)
            if self.config.price_floor > 0:
                # price_floor must not breach max_allowed_bid
                clamped = max(self.config.price_floor, clamped)
                clamped = min(clamped, max_allowed_bid)
            return self.quantize_price(clamped, is_bid=True)
        else:
            min_allowed_ask = self.quantize_price(mid_price + min_half_spread, is_bid=False)
            if min_allowed_ask <= mid_price:
                min_allowed_ask = self.quantize_price(mid_price + self.tick_size, is_bid=False)
            if best_bid is not None and best_bid > 0:
                min_allowed_ask = max(min_allowed_ask, self.quantize_price(best_bid + self.tick_size, is_bid=False))
            if best_ask is not None and best_ask > 0:
                min_allowed_ask = max(min_allowed_ask, self.quantize_price(best_ask, is_bid=False))

            if self.config.price_ceiling > 0 and self.config.price_ceiling < min_allowed_ask:
                logger.warning(
                    f"[QUOTER_WARNING] price_ceiling ({self.config.price_ceiling}) < min_allowed_ask ({min_allowed_ask}) "
                    f"for {self.config.symbol}. Clamping to min_allowed_ask and skipping quote."
                )
                return None

            clamped = max(quantized, min_allowed_ask)
            if self.config.price_ceiling > 0:
                # price_ceiling must not breach min_allowed_ask
                clamped = min(self.config.price_ceiling, clamped)
                clamped = max(clamped, min_allowed_ask)
            return self.quantize_price(clamped, is_bid=False)

    def estimate_minimum_margin_required(self, active_long: bool = True, active_short: bool = True) -> float:
        """
        Estimate minimum margin required in USDT to place at least one level 0 quote order.
        Margin required per active side = min_notional / leverage.
        """
        active_count = (1 if active_long else 0) + (1 if active_short else 0)
        if active_count == 0:
            return 0.0
        lev = max(1, self.config.leverage)
        min_notional_per_order = max(self.min_notional, 5.0)
        return (min_notional_per_order * active_count) / lev

    def calculate_quotes(
        self,
        smoothed_mid: float,
        long_value_usdt: float,
        short_value_usdt: float,
        vol_mult: float = 1.0,
        pause_long_entry: bool = False,
        pause_short_entry: bool = False,
        best_bid: Optional[float] = None,
        best_ask: Optional[float] = None,
        available_margin: Optional[float] = None,
        long_state: Optional[Any] = None,
        short_state: Optional[Any] = None,
        current_time: Optional[float] = None,
        mark_price: Optional[float] = None,
    ) -> Tuple[List[QuoteLevel], List[QuoteLevel]]:
        """
        Generate bid (LONG entry) and ask (SHORT entry) quote levels with independent skew and ladder throttling.
        If long_state/short_state provided, enforces sequential quoting (Level 0 when flat, Level i after 30m cooldown if price condition met).
        If available_margin is provided, auto-scales order amounts and limits levels to available margin.
        Returns: (bids, asks)
        """
        if smoothed_mid <= 0:
            return [], []

        # ── 1. Calculate per-side inventory skew ──
        rho_long = min(1.0, max(0.0, long_value_usdt / max(1.0, self.config.max_long_usdt)))
        rho_short = min(1.0, max(0.0, short_value_usdt / max(1.0, self.config.max_short_usdt)))

        max_side_usdt = max(self.config.max_long_usdt, self.config.max_short_usdt, 1.0)
        rho_net = (long_value_usdt - short_value_usdt) / max_side_usdt
        rho_net = max(-1.0, min(1.0, rho_net))

        # Reference center price shifted slightly by net inventory
        ref_mid = smoothed_mid * (1.0 - self.config.skew_gamma_net * rho_net)

        # Per-side widening skew:
        # Holding LONG -> push bid further down (do not want to buy more)
        # Holding SHORT -> push ask further up (do not want to sell more)
        skew_bid = self.config.skew_kappa * self.config.bid_spread * rho_long if self.config.inventory_skew_enabled else 0.0
        skew_ask = self.config.skew_kappa * self.config.ask_spread * rho_short if self.config.inventory_skew_enabled else 0.0

        spread_floor = self.calculate_spread_floor(smoothed_mid)
        s_bid_eff = max(self.config.bid_spread * vol_mult + skew_bid, spread_floor)
        s_ask_eff = max(self.config.ask_spread * vol_mult + skew_ask, spread_floor)

        bids: List[QuoteLevel] = []
        asks: List[QuoteLevel] = []

        # Check gross exposure cap
        gross_exposure = long_value_usdt + short_value_usdt
        gross_exceeded = gross_exposure >= self.config.gross_exposure_cap_usdt

        # Check if per-side caps are reached
        long_cap_reached = (long_value_usdt >= self.config.max_long_usdt) or gross_exceeded or pause_long_entry
        short_cap_reached = (short_value_usdt >= self.config.max_short_usdt) or gross_exceeded or pause_short_entry

        active_sides = (0 if long_cap_reached else 1) + (0 if short_cap_reached else 1)
        if active_sides == 0:
            return [], []

        # Margin and Notional budgeting per active side
        lev = max(1, self.config.leverage)
        rem_bid_budget: Optional[float] = None
        rem_ask_budget: Optional[float] = None

        if available_margin is not None:
            if available_margin <= 0:
                return [], []
            side_margin_budget = available_margin / float(active_sides)
            side_notional_budget = side_margin_budget * lev
            rem_bid_budget = side_notional_budget if not long_cap_reached else 0.0
            rem_ask_budget = side_notional_budget if not short_cap_reached else 0.0

        # ── 2. Generate Bid Levels (BUY for positionSide=LONG) ──
        if not long_cap_reached:
            levels_to_quote_bid = list(range(self.config.order_levels))
            if long_state is not None:
                filled_cnt = getattr(long_state, "filled_levels_count", 0)
                pos_amt = getattr(long_state, "amount", 0.0)
                now = current_time if current_time is not None else time.time()
                next_cd_time = getattr(long_state, "next_allowed_level_time", 0.0)
                entry_p = getattr(long_state, "entry_price", 0.0)
                chk_price = mark_price or smoothed_mid

                if pos_amt <= 1e-7 or filled_cnt == 0:
                    levels_to_quote_bid = [0]
                else:
                    target_lvl = filled_cnt
                    if target_lvl >= self.config.order_levels:
                        levels_to_quote_bid = []
                    elif now < next_cd_time:
                        rem_cd = int(next_cd_time - now)
                        logger.info(
                            f"[{self.config.symbol}][LONG][LEVEL_COOLDOWN_ACTIVE] Level cooldown active (remaining {rem_cd}s). "
                            f"Suppressing next Level #{target_lvl} quote."
                        )
                        levels_to_quote_bid = []
                    else:
                        # Cooldown expired: check conditional price threshold
                        # Slot LONG: mark_price <= entry_price * (1 - order_level_spread)
                        threshold = entry_p * (1.0 - self.config.order_level_spread)
                        if chk_price <= threshold and threshold > 0:
                            logger.info(
                                f"[{self.config.symbol}][LONG][LEVEL_ORDER_PLACED] Mark price {chk_price:.4f} <= threshold {threshold:.4f}. "
                                f"Placing Level #{target_lvl} entry quote."
                            )
                            levels_to_quote_bid = [target_lvl]
                        else:
                            logger.debug(
                                f"[{self.config.symbol}][LONG] Mark price {chk_price:.4f} > threshold {threshold:.4f}. "
                                f"Waiting for price to reach Level #{target_lvl}."
                            )
                            levels_to_quote_bid = []

            for i in levels_to_quote_bid:
                if rem_bid_budget is not None and rem_bid_budget < self.min_notional:
                    break  # Margin budget exhausted for further levels

                s_bid_i = s_bid_eff + i * self.config.order_level_spread
                raw_bid_price = ref_mid * (1.0 - s_bid_i)
                bid_price = self.clamp_quote(
                    raw_bid_price,
                    is_bid=True,
                    mid_price=smoothed_mid,
                    spread_floor=spread_floor,
                    best_bid=best_bid,
                    best_ask=best_ask,
                )

                if bid_price is None or bid_price <= 0:
                    continue

                amount_usdt = self.config.order_amount_usdt + i * self.config.order_level_amount
                # Ensure we don't exceed max_long_usdt with this level
                if long_value_usdt + amount_usdt > self.config.max_long_usdt:
                    amount_usdt = max(self.min_notional, self.config.max_long_usdt - long_value_usdt)

                # Auto-scale down amount_usdt if remaining margin budget is limited
                if rem_bid_budget is not None and amount_usdt > rem_bid_budget:
                    amount_usdt = rem_bid_budget

                # Base quantity calculation
                raw_qty = amount_usdt / bid_price

                # Min quantity filter
                if self.min_amount > 0 and raw_qty < self.min_amount:
                    raw_qty = self.min_amount

                # Min notional filter with 1.05 safety buffer
                min_req_notional = self.min_notional * 1.05
                if raw_qty * bid_price < min_req_notional:
                    raw_qty = min_req_notional / bid_price

                qty = self.quantize_amount(raw_qty)
                if (qty * bid_price) < self.min_notional and self.step_size > 0:
                    qty = self.quantize_amount(qty + self.step_size)

                level_notional = round(qty * bid_price, 2)
                if rem_bid_budget is not None and level_notional > (rem_bid_budget * 1.05) and rem_bid_budget < self.min_notional:
                    break

                if qty > 0 and level_notional >= self.min_notional:
                    bids.append(QuoteLevel(
                        side=OrderSide.BUY,
                        position_side=PositionSide.LONG,
                        level=i,
                        price=bid_price,
                        amount=qty,
                        notional=level_notional,
                    ))
                    if rem_bid_budget is not None:
                        rem_bid_budget = max(0.0, rem_bid_budget - level_notional)

        # ── 3. Generate Ask Levels (SELL for positionSide=SHORT) ──
        if not short_cap_reached:
            levels_to_quote_ask = list(range(self.config.order_levels))
            if short_state is not None:
                filled_cnt = getattr(short_state, "filled_levels_count", 0)
                pos_amt = getattr(short_state, "amount", 0.0)
                now = current_time if current_time is not None else time.time()
                next_cd_time = getattr(short_state, "next_allowed_level_time", 0.0)
                entry_p = getattr(short_state, "entry_price", 0.0)
                chk_price = mark_price or smoothed_mid

                if pos_amt <= 1e-7 or filled_cnt == 0:
                    levels_to_quote_ask = [0]
                else:
                    target_lvl = filled_cnt
                    if target_lvl >= self.config.order_levels:
                        levels_to_quote_ask = []
                    elif now < next_cd_time:
                        rem_cd = int(next_cd_time - now)
                        logger.info(
                            f"[{self.config.symbol}][SHORT][LEVEL_COOLDOWN_ACTIVE] Level cooldown active (remaining {rem_cd}s). "
                            f"Suppressing next Level #{target_lvl} quote."
                        )
                        levels_to_quote_ask = []
                    else:
                        # Cooldown expired: check conditional price threshold
                        # Slot SHORT: mark_price >= entry_price * (1 + order_level_spread)
                        threshold = entry_p * (1.0 + self.config.order_level_spread)
                        if chk_price >= threshold and threshold > 0:
                            logger.info(
                                f"[{self.config.symbol}][SHORT][LEVEL_ORDER_PLACED] Mark price {chk_price:.4f} >= threshold {threshold:.4f}. "
                                f"Placing Level #{target_lvl} entry quote."
                            )
                            levels_to_quote_ask = [target_lvl]
                        else:
                            logger.debug(
                                f"[{self.config.symbol}][SHORT] Mark price {chk_price:.4f} < threshold {threshold:.4f}. "
                                f"Waiting for price to reach Level #{target_lvl}."
                            )
                            levels_to_quote_ask = []

            for i in levels_to_quote_ask:
                if rem_ask_budget is not None and rem_ask_budget < self.min_notional:
                    break  # Margin budget exhausted for further levels

                s_ask_i = s_ask_eff + i * self.config.order_level_spread
                raw_ask_price = ref_mid * (1.0 + s_ask_i)
                ask_price = self.clamp_quote(
                    raw_ask_price,
                    is_bid=False,
                    mid_price=smoothed_mid,
                    spread_floor=spread_floor,
                    best_bid=best_bid,
                    best_ask=best_ask,
                )

                if ask_price is None or ask_price <= 0:
                    continue

                amount_usdt = self.config.order_amount_usdt + i * self.config.order_level_amount
                if short_value_usdt + amount_usdt > self.config.max_short_usdt:
                    amount_usdt = max(self.min_notional, self.config.max_short_usdt - short_value_usdt)

                # Auto-scale down amount_usdt if remaining margin budget is limited
                if rem_ask_budget is not None and amount_usdt > rem_ask_budget:
                    amount_usdt = rem_ask_budget

                raw_qty = amount_usdt / ask_price

                # Min quantity filter
                if self.min_amount > 0 and raw_qty < self.min_amount:
                    raw_qty = self.min_amount

                # Min notional filter with 1.05 safety buffer
                min_req_notional = self.min_notional * 1.05
                if raw_qty * ask_price < min_req_notional:
                    raw_qty = min_req_notional / ask_price

                qty = self.quantize_amount(raw_qty)
                if (qty * ask_price) < self.min_notional and self.step_size > 0:
                    qty = self.quantize_amount(qty + self.step_size)

                level_notional = round(qty * ask_price, 2)
                if rem_ask_budget is not None and level_notional > (rem_ask_budget * 1.05) and rem_ask_budget < self.min_notional:
                    break

                if qty > 0 and level_notional >= self.min_notional:
                    asks.append(QuoteLevel(
                        side=OrderSide.SELL,
                        position_side=PositionSide.SHORT,
                        level=i,
                        price=ask_price,
                        amount=qty,
                        notional=level_notional,
                    ))
                    if rem_ask_budget is not None:
                        rem_ask_budget = max(0.0, rem_ask_budget - level_notional)

        return bids, asks
