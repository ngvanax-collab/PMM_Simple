"""Dynamic Pair Rebalancer: Portfolio Lifecycle & Anti-Churning Orchestration."""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from loguru import logger
from pydantic import BaseModel, Field

from app.core.screener import MarketMetric, QuantitativeScreener, compute_vamm_parameters, screener_engine
from app.models.config import PairConfig, TrailingStopConfig
from app.persistence.store import config_store


@dataclass
class RebalanceEvent:
    """Log record of rebalancing actions."""
    timestamp: float = field(default_factory=time.time)
    action: str = "SCAN"  # "ADD", "DRAIN", "RETIRE", "RETAIN", "SCAN"
    symbol: str = ""
    score: float = 0.0
    rank: int = 0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "symbol": self.symbol,
            "score": round(self.score, 2),
            "rank": self.rank,
            "reason": self.reason,
        }


class RebalancerConfig(BaseModel):
    """Dynamic Rebalancer Configuration."""
    enabled: bool = Field(default=True, description="Enable automatic periodic screening and rebalancing")
    max_active_pairs: int = Field(default=5, ge=1, le=10, description="Max concurrent active trading pair slots")
    scan_interval_minutes: int = Field(default=60, ge=1, le=1440, description="Periodic scan interval in minutes")
    rank_threshold: int = Field(default=7, ge=1, le=20, description="Evict running pair if rank drops beyond this threshold")
    score_delta_threshold_pct: float = Field(
        default=0.10, ge=0.01, le=0.50,
        description="Challenger must have >= 10% higher score to displace an active pair within buffer"
    )
    total_margin_budget_usdt: float = Field(default=400.0, gt=0, description="Total portfolio margin budget across all slots (80 USDT * 5)")
    default_leverage: int = Field(default=5, ge=1, le=20, description="Default leverage for auto-promoted pairs")
    default_margin_mode: str = Field(default="isolated", description="Default margin mode: isolated or cross")
    default_order_amount_usdt: float = Field(default=33.0, gt=0, description="Default base quote level order amount")



class PairRebalancer:

    """
    Manages portfolio lifecycle for top N pairs:
    - Periodically runs Screener.
    - Evaluates incumbent pairs vs new candidates.
    - Applies Hysteresis & Anti-churning rules.
    - Orchestrates Graceful Drain -> Flat Check -> Retire -> Promote Candidate lifecycle.
    """

    def __init__(self, screener: Optional[QuantitativeScreener] = None, config: Optional[RebalancerConfig] = None):
        self.screener = screener or screener_engine
        self.config = config or RebalancerConfig()
        self.events: List[RebalanceEvent] = []
        self._loop_task: Optional[asyncio.Task] = None
        self._is_running: bool = False
        self._lock = asyncio.Lock()
        self.last_rebalance_time: float = 0.0

    @property
    def is_running(self) -> bool:
        return self._is_running

    def add_event(self, action: str, symbol: str, score: float, rank: int, reason: str) -> RebalanceEvent:
        evt = RebalanceEvent(
            timestamp=time.time(),
            action=action,
            symbol=symbol,
            score=score,
            rank=rank,
            reason=reason
        )
        self.events.append(evt)
        if len(self.events) > 200:
            self.events.pop(0)
        logger.info(f"[REBALANCER EVENT] [{action}] {symbol} (Rank {rank}, Score {score:.1f}): {reason}")
        return evt

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in reversed(self.events[-limit:])]

    async def execute_rebalance_cycle(self, bot_manager: Any) -> Dict[str, Any]:
        """
        Execute one full rebalance evaluation cycle:
        1. Check draining workers: if is_flat == True, stop and retire worker to free margin quota.
        2. Run Screener across all USDT perpetual futures.
        3. Compare active workers with ranked candidate list applying Hysteresis.
        4. Switch demoted pairs to DRAINING.
        5. Fill free slots with Top Candidates.
        """
        async with self._lock:
            start_t = time.time()
            logger.info("Executing Dynamic Pair Rebalancing Cycle...")
            summary: Dict[str, Any] = {
                "timestamp": start_t,
                "retired_pairs": [],
                "draining_pairs": [],
                "retained_pairs": [],
                "added_pairs": [],
                "total_candidates": 0,
            }

            if not bot_manager.gateway or not bot_manager.gateway._is_connected:
                logger.warning("Rebalancer: Gateway not connected. Aborting cycle.")
                return summary

            # ── Step 1: Check Draining Workers for Flat Retirement ──
            for symbol, worker in list(bot_manager.workers.items()):
                if getattr(worker, "is_draining", False):
                    if worker.is_flat:
                        logger.info(f"[{symbol}] Draining worker is 100% Flat. Stopping and retiring worker.")
                        await bot_manager.stop_pair(symbol)
                        summary["retired_pairs"].append(symbol)
                        self.add_event(
                            action="RETIRE",
                            symbol=symbol,
                            score=0.0,
                            rank=0,
                            reason="Positions 100% Flat. Worker safely stopped & margin reclaimed."
                        )
                    else:
                        long_amt = worker.tracker.long_pos.amount
                        short_amt = worker.tracker.short_pos.amount
                        logger.info(f"[{symbol}] Worker still draining positions: Long={long_amt:.4f}, Short={short_amt:.4f}")

            # ── Step 2: Screen and Rank All Pairs ──
            candidates = await self.screener.scan_and_rank_all_pairs(bot_manager.gateway)
            summary["total_candidates"] = len(candidates)
            if not candidates:
                logger.warning("Rebalancer: No candidates returned from screener.")
                return summary

            cand_map: Dict[str, MarketMetric] = {c.symbol: c for c in candidates}

            # ── Step 3: Evaluate Currently Running Workers (Hysteresis Check) ──
            running_workers = {
                s: w for s, w in bot_manager.workers.items()
                if w.config.enabled and not getattr(w, "is_draining", False)
            }

            per_pair_margin = self.config.total_margin_budget_usdt / max(1, self.config.max_active_pairs)

            for symbol, worker in list(running_workers.items()):
                metric = cand_map.get(symbol)
                rank = metric.rank if metric else 999
                score = metric.pmm_score if metric else 0.0

                # Check eviction condition (Rank Buffer & Hysteresis Delta)
                # Rule: Evict if rank > rank_threshold (default 7)
                should_drain = False
                drain_reason = ""

                if rank > self.config.rank_threshold:
                    should_drain = True
                    drain_reason = f"Rank dropped to #{rank} (exceeded Rank Buffer #{self.config.rank_threshold})"
                elif rank > self.config.max_active_pairs:
                    # Marginal cutoff candidate (Rank #5) pushing incumbent out of Top 5
                    if len(candidates) >= self.config.max_active_pairs:
                        cutoff_candidate = candidates[self.config.max_active_pairs - 1]
                        delta = (cutoff_candidate.pmm_score - score) / max(1.0, score)
                        if delta >= self.config.score_delta_threshold_pct:
                            should_drain = True
                            drain_reason = (
                                f"Marginal candidate {cutoff_candidate.symbol} (Rank #{self.config.max_active_pairs}, Score {cutoff_candidate.pmm_score:.1f}) exceeds "
                                f"incumbent {symbol} (Rank #{rank}, Score {score:.1f}) by {delta*100:.1f}% (>= {self.config.score_delta_threshold_pct*100:.0f}%)"
                            )


                if should_drain:
                    worker.set_drain_mode(True)
                    summary["draining_pairs"].append(symbol)
                    self.add_event(
                        action="DRAIN",
                        symbol=symbol,
                        score=score,
                        rank=rank,
                        reason=drain_reason
                    )
                else:
                    summary["retained_pairs"].append(symbol)
                    self.add_event(
                        action="RETAIN",
                        symbol=symbol,
                        score=score,
                        rank=rank,
                        reason=f"Retained in portfolio (Rank #{rank}, Score {score:.1f})"
                    )

            # ── Step 4: Promote New Top Candidates into Available Slots ──
            # Count currently occupied slots (running + draining not yet flat)
            occupied_slots = sum(
                1 for w in bot_manager.workers.values()
                if w.config.enabled and (not getattr(w, "is_draining", False) or not w.is_flat)
            )
            available_slots = max(0, self.config.max_active_pairs - occupied_slots)

            logger.info(f"Rebalancer: Occupied slots={occupied_slots}/{self.config.max_active_pairs}, Available={available_slots}")

            if available_slots > 0:
                for cand in candidates:
                    if available_slots <= 0:
                        break
                    # Skip if already exists and enabled
                    if cand.symbol in bot_manager.workers:
                        w = bot_manager.workers[cand.symbol]
                        if w.config.enabled and not getattr(w, "is_draining", False):
                            continue

                    # Create and configure standard PairConfig for candidate
                    pair_cfg = config_store.load_pair_config(cand.symbol)
                    exchange_val = "binance"
                    if bot_manager.gateway and hasattr(bot_manager.gateway, "exchange_name"):
                        ex_name = bot_manager.gateway.exchange_name
                        if isinstance(ex_name, str) and ex_name:
                            exchange_val = ex_name

                    # Compute VAMM Dynamic Parameters based on cand.natr_14
                    vamm_params = compute_vamm_parameters(
                        natr_pct=cand.natr_14 if cand.natr_14 > 0 else 1.2,
                        allocated_margin=21.0,
                        leverage=5,
                        order_levels=3,
                    )

                    if not pair_cfg:
                        pair_cfg = PairConfig(
                            symbol=cand.symbol,
                            exchange=exchange_val,
                            enabled=True,
                            leverage=5,
                            margin_mode="isolated",
                            order_amount_usdt=vamm_params["order_amount_usdt"],
                            bid_spread=vamm_params["bid_spread"],
                            ask_spread=vamm_params["ask_spread"],
                            minimum_spread=vamm_params["minimum_spread"],
                            order_levels=vamm_params["order_levels"],
                            order_level_spread=vamm_params["order_level_spread"],
                            order_level_amount=vamm_params["order_level_amount"],
                            level_cooldown_sec=vamm_params["level_cooldown_sec"],
                            order_refresh_time=45,
                            requote_threshold_pct=0.001,
                            min_holding_sec=3.0,
                            inventory_skew_enabled=True,
                            allocated_margin_usdt=vamm_params["allocated_margin_usdt"],
                            max_long_usdt=vamm_params["max_long_usdt"],
                            max_short_usdt=vamm_params["max_short_usdt"],
                            gross_exposure_cap_usdt=vamm_params["gross_exposure_cap_usdt"],
                            skew_kappa=1.0,
                            skew_gamma_net=0.001,
                            tp_skew_boost=0.5,
                            take_profit=vamm_params["take_profit"],
                            take_profit_order_type="LIMIT_MAKER",
                            trailing_tp_enabled=vamm_params["trailing_tp_enabled"],
                            trailing_tp_activation_pct=vamm_params["trailing_tp_activation_pct"],
                            trailing_tp_callback_pct=vamm_params["trailing_tp_callback_pct"],
                            stop_loss=vamm_params["stop_loss"],
                            stop_loss_order_type="MARKET",
                            time_limit=14400,
                            passive_exit_timeout_sec=120.0,
                            base_cooldown_sec=600,
                            cooldown_multiplier=2.0,
                            max_cooldown_sec=43200,
                            worker_max_loss_usdt=20.0,
                            worker_max_drawdown_usdt=15.0,
                            vol_pause_pct=0.025,
                            vol_lookback_sec=60,
                        )
                    else:
                        pair_cfg.enabled = True
                        pair_cfg.order_amount_usdt = vamm_params["order_amount_usdt"]
                        pair_cfg.bid_spread = vamm_params["bid_spread"]
                        pair_cfg.ask_spread = vamm_params["ask_spread"]
                        pair_cfg.minimum_spread = vamm_params["minimum_spread"]
                        pair_cfg.order_levels = vamm_params["order_levels"]
                        pair_cfg.order_level_spread = vamm_params["order_level_spread"]
                        pair_cfg.order_level_amount = vamm_params["order_level_amount"]
                        pair_cfg.level_cooldown_sec = vamm_params["level_cooldown_sec"]
                        pair_cfg.order_refresh_time = 45
                        pair_cfg.requote_threshold_pct = 0.001
                        pair_cfg.min_holding_sec = 3.0
                        pair_cfg.inventory_skew_enabled = True
                        pair_cfg.allocated_margin_usdt = vamm_params["allocated_margin_usdt"]
                        pair_cfg.max_long_usdt = vamm_params["max_long_usdt"]
                        pair_cfg.max_short_usdt = vamm_params["max_short_usdt"]
                        pair_cfg.gross_exposure_cap_usdt = vamm_params["gross_exposure_cap_usdt"]
                        pair_cfg.skew_kappa = 1.0
                        pair_cfg.skew_gamma_net = 0.001
                        pair_cfg.tp_skew_boost = 0.5
                        pair_cfg.take_profit = vamm_params["take_profit"]
                        pair_cfg.take_profit_order_type = "LIMIT_MAKER"
                        pair_cfg.trailing_tp_enabled = vamm_params["trailing_tp_enabled"]
                        pair_cfg.trailing_tp_activation_pct = vamm_params["trailing_tp_activation_pct"]
                        pair_cfg.trailing_tp_callback_pct = vamm_params["trailing_tp_callback_pct"]
                        pair_cfg.stop_loss = vamm_params["stop_loss"]
                        pair_cfg.stop_loss_order_type = "MARKET"
                        pair_cfg.time_limit = 14400
                        pair_cfg.passive_exit_timeout_sec = 120.0
                        pair_cfg.base_cooldown_sec = 600
                        pair_cfg.cooldown_multiplier = 2.0
                        pair_cfg.max_cooldown_sec = 43200
                        pair_cfg.worker_max_loss_usdt = 20.0
                        pair_cfg.worker_max_drawdown_usdt = 15.0
                        pair_cfg.vol_pause_pct = 0.025
                        pair_cfg.vol_lookback_sec = 60



                    success = await bot_manager.register_dynamic_pair(pair_cfg)
                    if success:
                        summary["added_pairs"].append(cand.symbol)
                        available_slots -= 1
                        if hasattr(bot_manager, "workers") and cand.symbol in bot_manager.workers:
                            bot_manager.workers[cand.symbol].market_state.update_natr_15m(cand.natr_14 if cand.natr_14 > 0 else 1.2)
                        self.add_event(
                            action="ADD",
                            symbol=cand.symbol,
                            score=cand.pmm_score,
                            rank=cand.rank,
                            reason=f"Promoted to portfolio (Rank #{cand.rank}, Score {cand.pmm_score:.1f}, Margin ${per_pair_margin:.0f})"
                        )

            self.last_rebalance_time = time.time()
            logger.info(f"Rebalance cycle complete: {summary}")
            return summary

    async def start_background_loop(self, bot_manager: Any) -> None:
        """Start periodic rebalancing loop."""
        if self._is_running:
            return

        self._is_running = True
        self._loop_task = asyncio.create_task(self._rebalance_worker_loop(bot_manager))
        logger.info(f"Dynamic Pair Rebalancer background loop started (interval={self.config.scan_interval_minutes}m).")

    async def stop_background_loop(self) -> None:
        """Stop periodic rebalancing loop."""
        self._is_running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        logger.info("Dynamic Pair Rebalancer background loop stopped.")

    async def _rebalance_worker_loop(self, bot_manager: Any) -> None:
        """Background loop executing periodically."""
        # Initial delay to allow gateway and initial workers to initialize
        await asyncio.sleep(10.0)

        while self._is_running:
            try:
                if self.config.enabled:
                    await self.execute_rebalance_cycle(bot_manager)

                interval_sec = max(60, self.config.scan_interval_minutes * 60)
                await asyncio.sleep(interval_sec)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rebalancer loop exception: {e}")
                await asyncio.sleep(30.0)


# Global singleton instance
rebalancer_service = PairRebalancer()
