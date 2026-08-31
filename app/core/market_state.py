"""Market State Tracking, EMA Smoothed Mid, Volatility & Basis Estimation."""
import math
import time
from collections import deque
from typing import Deque, Optional, Tuple
from loguru import logger

from app.models.config import PairConfig


class MarketState:
    """Tracks orderbook top, smoothed mid, volatility, and anomaly filters per symbol."""

    def __init__(self, config: PairConfig):
        self.config = config
        self.symbol = config.symbol

        self.best_bid: float = 0.0
        self.best_ask: float = 0.0
        self.raw_mid: float = 0.0
        self.mark_price: float = 0.0
        self.smoothed_mid: float = 0.0

        self.ema_alpha: float = 0.3  # Smoothing factor for mid price
        self.last_update_time: float = 0.0

        # Price history for rolling volatility calculation: deque of (timestamp, price)
        self.price_history: Deque[Tuple[float, float]] = deque(maxlen=600)
        self.baseline_vol: float = 0.005  # 0.5% default baseline volatility

        # ── Dynamic NATR-Anchored Volatility Circuit Breaker ──
        self.price_history_60s: Deque[Tuple[float, float]] = deque()
        self.circuit_breaker_paused_until: float = 0.0
        self.current_natr_15m: float = 0.012  # Default 1.2% (0.012 in decimal)

    def update_natr_15m(self, natr: float) -> None:
        """Update 15m NATR for dynamic circuit breaker thresholding."""
        # Normalize if passed as percentage (e.g. 1.4%) vs decimal (0.014)
        self.current_natr_15m = natr / 100.0 if natr > 0.1 else natr
        thresh = self.get_circuit_breaker_threshold()
        logger.info(
            f"[{self.symbol}][VAMM_15M_UPDATE] Updated NATR 15m = {self.current_natr_15m*100:.2f}% "
            f"(Circuit Breaker Threshold = {thresh*100:.2f}%)"
        )

    def get_circuit_breaker_threshold(self) -> float:
        """
        Compute dynamic threshold: T_cb = max(circuit_breaker_min_threshold_pct, circuit_breaker_natr_multiplier * current_natr_15m)
        """
        min_thresh = getattr(self.config, "circuit_breaker_min_threshold_pct", 0.008)
        multiplier = getattr(self.config, "circuit_breaker_natr_multiplier", 1.0)
        return max(min_thresh, multiplier * self.current_natr_15m)

    def check_circuit_breaker(self, current_time: Optional[float] = None) -> Tuple[bool, float, float]:
        """
        Check 60s sliding window peak-to-trough price surge:
        delta_p_60s = (max(P_60s) - min(P_60s)) / min(P_60s)
        Returns: (is_tripped, delta_60s, threshold)
        """
        if not getattr(self.config, "circuit_breaker_enabled", True):
            return False, 0.0, 0.0

        now = current_time if current_time is not None else time.time()
        lookback = getattr(self.config, "circuit_breaker_lookback_sec", 60)

        # Evict old data points beyond lookback window
        while self.price_history_60s and (now - self.price_history_60s[0][0]) > lookback:
            self.price_history_60s.popleft()

        if len(self.price_history_60s) < 2:
            return False, 0.0, self.get_circuit_breaker_threshold()

        prices = [p for _, p in self.price_history_60s]
        min_p = min(prices)
        max_p = max(prices)
        if min_p <= 0:
            return False, 0.0, self.get_circuit_breaker_threshold()

        delta_60s = (max_p - min_p) / min_p
        threshold = self.get_circuit_breaker_threshold()
        is_tripped = delta_60s >= threshold

        return is_tripped, delta_60s, threshold

    def is_circuit_breaker_active(self, current_time: Optional[float] = None) -> bool:
        """Return True if circuit breaker pause is currently active."""
        if not getattr(self.config, "circuit_breaker_enabled", True):
            return False
        now = current_time if current_time is not None else time.time()
        return now < self.circuit_breaker_paused_until

    def update_top_of_book(self, bid: float, ask: float, mark: Optional[float] = None) -> None:
        """Update market state with new top-of-book bid, ask and optional mark price."""
        now = time.time()
        if bid > 0:
            self.best_bid = bid
        if ask > 0:
            self.best_ask = ask
        if mark is not None and mark > 0:
            self.mark_price = mark

        tracked_price = mark if (mark is not None and mark > 0) else ((bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0)
        if tracked_price > 0:
            self.price_history_60s.append((now, tracked_price))
            # Clean old items
            lookback = getattr(self.config, "circuit_breaker_lookback_sec", 60)
            while self.price_history_60s and (now - self.price_history_60s[0][0]) > lookback:
                self.price_history_60s.popleft()

        if self.best_bid > 0 and self.best_ask > 0 and self.best_bid < self.best_ask:
            self.raw_mid = (self.best_bid + self.best_ask) / 2.0
            if self.smoothed_mid <= 0.0:
                self.smoothed_mid = self.raw_mid
            else:
                self.smoothed_mid = self.ema_alpha * self.raw_mid + (1.0 - self.ema_alpha) * self.smoothed_mid
            self.price_history.append((now, self.raw_mid))
        elif self.mark_price > 0 and self.raw_mid <= 0.0:
            self.raw_mid = self.mark_price
            if self.smoothed_mid <= 0.0:
                self.smoothed_mid = self.raw_mid

        self.last_update_time = now

    def update_ticker(self, bid: float, ask: float, mark: float) -> None:
        """Update market state with new ticker/orderbook top."""
        self.update_top_of_book(bid=bid, ask=ask, mark=mark)

    def get_volatility_multiplier(self) -> float:
        """
        Compute volatility ratio: vol_mult = max(1.0, sigma_recent / sigma_baseline).
        """
        now = time.time()
        lookback = self.config.vol_lookback_sec
        recent_prices = [p for t, p in self.price_history if now - t <= lookback]

        if len(recent_prices) < 5:
            return 1.0

        mean = sum(recent_prices) / len(recent_prices)
        if mean <= 0:
            return 1.0

        variance = sum((p - mean) ** 2 for p in recent_prices) / len(recent_prices)
        std_dev = math.sqrt(variance)
        rel_std_dev = std_dev / mean

        if self.baseline_vol <= 0:
            return 1.0

        ratio = rel_std_dev / self.baseline_vol
        return max(1.0, min(ratio, 4.0))  # Cap multiplier between 1.0 and 4.0

    def check_market_sanity(self) -> Tuple[bool, str]:
        """
        Verify that market conditions are safe for quoting:
        - Dynamic Volatility Circuit Breaker is not active.
        - Spread is not crossed or excessively wide.
        - Basis |mid - mark| / mark <= max_basis_pct.
        - Price movement within lookback <= vol_pause_pct.
        """
        # Dynamic Volatility Circuit Breaker check
        if self.is_circuit_breaker_active():
            return False, f"Dynamic Volatility Circuit Breaker active (paused until {self.circuit_breaker_paused_until:.0f})"

        if self.best_bid <= 0 or self.best_ask <= 0 or self.best_bid >= self.best_ask:
            return False, f"Invalid or crossed top-of-book: bid={self.best_bid}, ask={self.best_ask}"

        # Book spread check
        book_spread_pct = (self.best_ask - self.best_bid) / self.smoothed_mid
        if book_spread_pct > self.config.max_book_spread_pct:
            return False, f"Orderbook spread too wide: {book_spread_pct:.4f} > max {self.config.max_book_spread_pct:.4f}"

        # Basis check vs mark price
        if self.mark_price > 0:
            basis_pct = abs(self.smoothed_mid - self.mark_price) / self.mark_price
            if basis_pct > self.config.max_basis_pct:
                return False, f"Basis |mid - mark| / mark too high: {basis_pct:.4f} > max {self.config.max_basis_pct:.4f}"

        # Volatility pause check (max drawdown / runup within lookback)
        now = time.time()
        lookback = self.config.vol_lookback_sec
        recent_prices = [p for t, p in self.price_history if now - t <= lookback]
        if len(recent_prices) >= 2:
            min_p = min(recent_prices)
            max_p = max(recent_prices)
            if min_p > 0:
                swing_pct = (max_p - min_p) / min_p
                if swing_pct > self.config.vol_pause_pct:
                    return False, f"Volatility surge: {swing_pct:.4f} > max allowed {self.config.vol_pause_pct:.4f} in {lookback}s"

        return True, "OK"
