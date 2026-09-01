"""Account and Per-Pair Circuit Breakers for Daily Loss & Gross Margin Ratio."""
import time
from typing import Dict, Optional, Tuple
from loguru import logger

from app.models.config import GlobalConfig, PairConfig
from app.persistence.db import db


class CircuitBreaker:
    """Monitors daily loss limits and gross margin ratios to protect account capital."""

    def __init__(self, global_config: GlobalConfig):
        self.global_config = global_config
        self._tripped: bool = False
        self._trip_reason: str = ""
        self._trip_time: float = 0.0
        self._reset_timestamp: float = 0.0

    @property
    def is_tripped(self) -> bool:
        """Return whether global circuit breaker has been tripped."""
        return self._tripped

    @property
    def trip_reason(self) -> str:
        """Return the reason why circuit breaker tripped."""
        return self._trip_reason

    def reset(self) -> None:
        """Manually reset the circuit breaker and reset the monitoring window."""
        self._tripped = False
        self._trip_reason = ""
        self._trip_time = 0.0
        self._reset_timestamp = time.time()
        logger.info("Account Circuit Breaker manually reset.")

    async def check_account_health(
        self,
        total_account_balance: Optional[float] = None,
        total_maintenance_margin: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Check account-wide daily loss and margin ratio.
        Margin Ratio = total_account_balance / total_maintenance_margin
        """
        if self._tripped:
            return True, self._trip_reason

        # ── 1. Daily Realized Loss Check ──
        # Calculate daily loss from midnight UTC or since last manual reset
        now = time.time()
        start_of_day = now - (now % 86400)
        since_ts = max(start_of_day, self._reset_timestamp)
        pnl_summary = await db.get_pnl_summary(since_timestamp=since_ts)
        net_daily_pnl = pnl_summary.get("total_net_pnl", 0.0)

        # Only trip if daily loss limit is configured (> 0) and net loss is negative exceeding the threshold
        limit_usdt = self.global_config.account_daily_loss_limit_usdt
        if limit_usdt > 0 and net_daily_pnl < 0 and abs(net_daily_pnl) >= limit_usdt:
            self._tripped = True
            self._trip_reason = f"Account daily loss exceeded: {net_daily_pnl:.2f} USDT <= -{limit_usdt:.2f} USDT"
            self._trip_time = now
            logger.critical(f"CIRCUIT BREAKER TRIPPED! {self._trip_reason}")
            return True, self._trip_reason

        # ── 2. Gross Margin Ratio Check ──
        # Only check margin ratio if there are active positions with maintenance margin > 0
        if (
            total_maintenance_margin is not None and total_maintenance_margin > 0 and
            total_account_balance is not None and total_account_balance > 0
        ):
            margin_ratio = total_account_balance / total_maintenance_margin
            if margin_ratio < self.global_config.kill_margin_ratio:
                self._tripped = True
                self._trip_reason = f"Emergency Margin Ratio breached: {margin_ratio:.2f} < kill threshold {self.global_config.kill_margin_ratio:.2f}"
                self._trip_time = now
                logger.critical(f"CIRCUIT BREAKER TRIPPED! {self._trip_reason}")
                return True, self._trip_reason
            elif margin_ratio < self.global_config.min_margin_ratio:
                logger.warning(
                    f"Warning: Margin ratio low: {margin_ratio:.2f} < warning threshold {self.global_config.min_margin_ratio:.2f}"
                )

        return False, "OK"

    async def check_pair_loss(self, config: PairConfig) -> Tuple[bool, str]:
        """Check if a specific pair has exceeded its individual daily loss limit."""
        now = time.time()
        start_of_day = now - (now % 86400)
        since_ts = max(start_of_day, self._reset_timestamp)
        pnl_summary = await db.get_pnl_summary(symbol=config.symbol, since_timestamp=since_ts)
        
        realized_pnl = pnl_summary.get("total_net_pnl", pnl_summary.get("total_realized_pnl", 0.0))
        max_daily_loss = getattr(config, "max_daily_loss_usdt", getattr(config, "daily_loss_limit_usdt", getattr(config, "max_loss_usdt", 0.0)))

        if max_daily_loss > 0 and realized_pnl < 0 and abs(realized_pnl) >= max_daily_loss:
            reason = f"Pair {config.symbol} daily loss limit exceeded: {realized_pnl:.2f} USDT >= {max_daily_loss:.2f} USDT"
            logger.critical(f"PAIR CIRCUIT BREAKER TRIPPED! {reason}")
            return True, reason

        return False, "OK"
