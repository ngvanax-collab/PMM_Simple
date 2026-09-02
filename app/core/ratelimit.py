"""Global Token Bucket Rate Limiter with Dual Budgets (Orders + Weight) and Jitter."""
import asyncio
import random
import time
from typing import Optional
from loguru import logger


class TokenBucket:
    """Token Bucket rate limiter with token refill over time."""

    def __init__(self, capacity: float, refill_rate_per_sec: float, name: str = "Bucket"):
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_rate = float(refill_rate_per_sec)
        self.name = name
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def consume(self, amount: float = 1.0, priority: bool = False) -> None:
        """Wait until tokens are available and consume them."""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
                self.last_update = now

                # If priority (e.g. emergency kill-all), allow slight deficit down to -capacity*0.2
                allowance = -self.capacity * 0.2 if priority else 0.0

                if self.tokens - amount >= allowance:
                    self.tokens -= amount
                    return

                # Calculate required sleep duration
                missing = amount - self.tokens
                sleep_time = max(0.01, missing / self.refill_rate)

            if sleep_time > 0.5:
                logger.warning(
                    f"[RATE_LIMITER] Token bucket '{self.name}' saturated, throttled for {sleep_time:.2f}s."
                )

            # Apply random jitter ±20% to prevent phase locking across workers
            jitter = random.uniform(0.8, 1.2)
            await asyncio.sleep(sleep_time * jitter)


class GlobalRateLimiter:
    """Coordinates global exchange rate limits across all pairs."""

    def __init__(
        self,
        orders_per_min: int = 900,
        weight_per_min: int = 2400,
    ):
        self.orders_bucket = TokenBucket(
            capacity=orders_per_min,
            refill_rate_per_sec=orders_per_min / 60.0,
            name="Orders",
        )
        self.weight_bucket = TokenBucket(
            capacity=weight_per_min,
            refill_rate_per_sec=weight_per_min / 60.0,
            name="Weight",
        )

    def configure(
        self,
        orders_per_min: Optional[int] = None,
        weight_per_min: Optional[int] = None,
    ) -> None:
        """Dynamically reconfigure token buckets from GlobalConfig."""
        if orders_per_min is not None and orders_per_min > 0:
            self.orders_bucket = TokenBucket(
                capacity=orders_per_min,
                refill_rate_per_sec=orders_per_min / 60.0,
                name="Orders",
            )
        if weight_per_min is not None and weight_per_min > 0:
            self.weight_bucket = TokenBucket(
                capacity=weight_per_min,
                refill_rate_per_sec=weight_per_min / 60.0,
                name="Weight",
            )
        logger.info(
            f"[RATE_LIMITER] Reconfigured rate limits: orders={orders_per_min or 'unchanged'}/min, "
            f"weight={weight_per_min or 'unchanged'}/min"
        )

    async def acquire_order(self, count: int = 1, weight: int = 1, priority: bool = False) -> None:
        """Acquire rate limit budget before sending order request(s)."""
        await asyncio.gather(
            self.orders_bucket.consume(count, priority=priority),
            self.weight_bucket.consume(weight, priority=priority)
        )

    async def acquire_weight(self, weight: int = 1, priority: bool = False) -> None:
        """Acquire rate limit budget for read/REST queries."""
        await self.weight_bucket.consume(weight, priority=priority)


# Global singleton instance
rate_limiter = GlobalRateLimiter()
