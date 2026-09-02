"""PMM Worker: Event-Driven Market Making Cycle per Symbol."""
import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from app.core.circuit_breaker import utc_day_start
from app.core.executor import TripleBarrierExecutor
from app.core.gateway import ExchangeGateway
from app.core.market_state import MarketState, calculate_atr_from_candles
from app.core.position_tracker import PositionTracker
from app.core.quoter import PMMQuoter, QuoteLevel
from app.models.config import PairConfig
from app.models.state import FillRecord, OrderPurpose, OrderRecord, OrderSide, OrderStatus, OrderType, PositionSide
from app.persistence.db import db
from app.persistence.store import config_store


class PMMWorker:
    """Independent worker coordinating quoting and triple-barrier execution for one trading pair."""

    def __init__(self, config: PairConfig, gateway: ExchangeGateway):
        self.config = config
        self.symbol = config.symbol
        self.gateway = gateway

        # Market Precision & Limits
        price_prec, amt_prec, tick_sz, step_sz = gateway.get_market_precision(self.symbol)
        min_amt, min_notional = 0.0, 5.0
        if hasattr(gateway, "get_market_limits"):
            try:
                min_amt, min_notional = gateway.get_market_limits(self.symbol)
            except Exception as e:
                logger.warning(f"[{self.symbol}] Failed to fetch market limits: {e}")

        # Components
        self.market_state = MarketState(config)
        self.quoter = PMMQuoter(
            config=config,
            price_precision=price_prec,
            amount_precision=amt_prec,
            tick_size=tick_sz,
            step_size=step_sz,
            min_amount=min_amt,
            min_notional=min_notional,
        )
        self.tracker = PositionTracker(config, gateway)

        # 2 Independent Triple Barrier Executors (LONG & SHORT)
        self.executor_long = TripleBarrierExecutor(
            config=config,
            position_side=PositionSide.LONG,
            gateway=gateway,
            position_tracker=self.tracker,
            quoter=self.quoter,
            on_isolated_kill=self._trigger_isolated_kill,
        )
        self.executor_short = TripleBarrierExecutor(
            config=config,
            position_side=PositionSide.SHORT,
            gateway=gateway,
            position_tracker=self.tracker,
            quoter=self.quoter,
            on_isolated_kill=self._trigger_isolated_kill,
        )

        self._running = False
        self._paused = False
        self._loop_task: Optional[asyncio.Task] = None
        self._ws_ticker_task: Optional[asyncio.Task] = None
        self._active_quote_orders: Dict[str, OrderRecord] = {}  # order_id -> record
        self._last_quote_mid: float = 0.0
        self._last_quote_time: float = 0.0
        self._last_margin_warning_time: float = 0.0
        self._lock = asyncio.Lock()

        # ── Worker-Level Risk & Drawdown Tracking ──
        self.session_realized_pnl: float = 0.0
        self.peak_pnl: float = 0.0
        self.is_locked_killed: bool = bool(self.config.is_locked)
        self.is_draining: bool = False

    @staticmethod
    def _handle_task_exception(task: asyncio.Task) -> None:
        """Callback to capture and log unhandled background task exceptions."""
        try:
            if not task.cancelled():
                exc = task.exception()
                if exc:
                    logger.error(f"[BACKGROUND_TASK_ERROR] Background task raised unhandled exception: {exc}")
        except Exception as e:
            logger.error(f"[BACKGROUND_TASK_ERROR] Error retrieving task exception: {e}")

    def _create_background_task(self, coro) -> asyncio.Task:
        """Create and track background task with error logging callback."""
        task = asyncio.create_task(coro)
        task.add_done_callback(self._handle_task_exception)
        return task

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_flat(self) -> bool:
        """Returns True if both LONG and SHORT gross position sizes are 0."""
        return (abs(self.tracker.long_pos.amount) < 1e-6 and abs(self.tracker.short_pos.amount) < 1e-6)

    def set_drain_mode(self, draining: bool = True) -> None:
        """
        Enable or disable Graceful Drain mode.
        When draining:
        - No new entry quote orders will be placed.
        - Any active quote orders are immediately cancelled.
        - Existing positions remain protected by TripleBarrierExecutor (TP/SL) until is_flat.
        """
        self.is_draining = draining
        if draining:
            logger.info(f"[{self.symbol}] Worker entered GRACEFUL DRAINING mode. Halting new entries.")
            self._create_background_task(self._cancel_active_quotes())
        else:
            logger.info(f"[{self.symbol}] Worker exited DRAINING mode.")

    async def _init_market_state(self) -> None:
        """Fetch initial ticker/orderbook and 1h candles to prime market_state and trend bias before quoting."""
        if hasattr(self.gateway, "fetch_ohlcv") and getattr(self.config, "trend_bias_enabled", True):
            try:
                candles_1h = await self.gateway.fetch_ohlcv(self.symbol, timeframe="1h", limit=100)
                if candles_1h:
                    regime = self.market_state.update_trend_bias(candles_1h)
                    self.tracker.long_pos.trend_bias_regime = regime
                    self.tracker.short_pos.trend_bias_regime = regime
                    self.tracker.long_pos.is_trend_blocked = (regime == "BEARISH")
                    self.tracker.short_pos.is_trend_blocked = (regime == "BULLISH")
            except Exception as e:
                logger.warning(f"[{self.symbol}] Initial trend bias load note: {e}")

        for attempt in range(3):
            try:
                ticker = await self.gateway.fetch_ticker_and_mark(self.symbol)
                if ticker and ticker["bid"] > 0 and ticker["ask"] > 0 and ticker["bid"] < ticker["ask"]:
                    await self.on_ticker_update(ticker["bid"], ticker["ask"], ticker["mark"])
                    logger.info(
                        f"[{self.symbol}] Initial market state primed: bid={ticker['bid']}, ask={ticker['ask']}, mark={ticker['mark']}"
                    )
                    return
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(f"[{self.symbol}] Attempt {attempt+1} to load initial market state: {e}")
                await asyncio.sleep(0.5)

    async def start(self) -> None:
        """Start the worker quoting cycle."""
        if self.config.is_locked or self.is_locked_killed:
            logger.warning(f"[{self.symbol}] Cannot start: Worker is locked by Isolated Kill Switch. Please unlock in UI.")
            return

        if self._running:
            return

        logger.info(f"[{self.symbol}] Starting PMM Worker (Hedge Mode)...")
        # Setup symbol leverage and margin mode
        await self.gateway.setup_symbol(self.symbol, self.config.leverage, self.config.margin_mode)

        # Initial reconcile with exchange
        await self.tracker.reconcile_with_exchange()

        # Prime market state with initial top of book
        await self._init_market_state()

        # Auto-arm and restore barrier protection (TP/SL/Overflow Recovery)
        await self.reconcile_barriers()

        # Restore daily realized PnL from DB (TASK H-1)
        try:
            now = time.time()
            start_of_day_utc = utc_day_start(now)
            pnl_summary = await db.get_pnl_summary(symbol=self.symbol, since_timestamp=start_of_day_utc)
            if pnl_summary:
                self.session_realized_pnl = float(pnl_summary.get("total_net_pnl", pnl_summary.get("total_realized_pnl", 0.0)) or 0.0)
                self.peak_pnl = max(0.0, self.session_realized_pnl)
                logger.info(
                    f"[{self.symbol}] Restored daily realized PnL from DB (since UTC 00:00 {start_of_day_utc:.0f}): "
                    f"${self.session_realized_pnl:+.4f} (Peak: ${self.peak_pnl:+.4f})"
                )
        except Exception as e:
            logger.warning(f"[{self.symbol}] Could not restore daily realized PnL on start: {e}")

        self._running = True
        self._paused = False

        # Reset 60s sliding window on start (TASK CB-5)
        self.market_state.price_history_60s.clear()

        # Launch public WS ticker stream for realtime top-of-book updates
        if hasattr(self.gateway, "watch_public_ticker"):
            self._ws_ticker_task = self._create_background_task(
                self.gateway.watch_public_ticker(
                    self.symbol,
                    lambda bid, ask, mark: self._create_background_task(self.on_ticker_update(bid, ask, mark)),
                    on_reconnect=lambda: self.market_state.price_history_60s.clear(),
                )
            )

        self._loop_task = self._create_background_task(self._main_worker_loop())

    async def stop(self) -> None:
        """Stop worker and cancel all open quote orders."""
        self._running = False
        if self._ws_ticker_task:
            self._ws_ticker_task.cancel()
            try:
                await self._ws_ticker_task
            except asyncio.CancelledError:
                pass
            self._ws_ticker_task = None

        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

        await self._cancel_active_quotes()
        logger.info(f"[{self.symbol}] PMM Worker stopped.")

    def pause(self) -> None:
        """Pause quoting (active executors continue to protect positions)."""
        self._paused = True
        logger.info(f"[{self.symbol}] PMM Worker paused.")

    def resume(self) -> None:
        """Resume quoting."""
        if self.config.is_locked or self.is_locked_killed:
            logger.warning(f"[{self.symbol}] Cannot resume: Worker is locked.")
            return
        self._paused = False
        logger.info(f"[{self.symbol}] PMM Worker resumed.")

    async def unlock(self) -> None:
        """Unlock worker from Max Loss/Drawdown isolated kill state."""
        self.config.is_locked = False
        self.is_locked_killed = False
        self.session_realized_pnl = 0.0
        self.peak_pnl = 0.0
        self.tracker.reset_sl_cooldown()
        config_store.save_pair_config(self.config)
        self._paused = False
        logger.info(f"[{self.symbol}] Worker UNLOCKED and risk metrics reset.")

    async def reconcile_barriers(self) -> None:
        """Auto-arm and reconcile Triple Barrier Executors for both LONG and SHORT slots."""
        mark = self.market_state.mark_price or self.tracker.long_pos.current_price or 0.0
        await self.executor_long.reconcile_barrier(mark)
        await self.executor_short.reconcile_barrier(mark)

    def get_risk_stats(self) -> Dict[str, Any]:
        """Return current session PnL, peak PnL, and drawdown metrics."""
        long_upnl = self.tracker.long_pos.unrealized_pnl
        short_upnl = self.tracker.short_pos.unrealized_pnl
        current_pnl = self.session_realized_pnl + (long_upnl + short_upnl)
        peak_pnl = max(self.peak_pnl, current_pnl)
        drawdown = peak_pnl - current_pnl
        return {
            "current_pnl": round(current_pnl, 2),
            "session_realized_pnl": round(self.session_realized_pnl, 2),
            "peak_pnl": round(peak_pnl, 2),
            "drawdown": round(drawdown, 2),
            "max_loss_usdt": getattr(self.config, "worker_max_loss_usdt", 30.0),
            "max_drawdown_usdt": getattr(self.config, "worker_max_drawdown_usdt", 40.0),
            "is_locked": bool(self.config.is_locked or self.is_locked_killed),
        }

    async def _check_isolated_risk_breach(self) -> None:
        """Check if Worker Max Loss or Max Drawdown limit is breached."""
        if self.config.is_locked or self.is_locked_killed:
            return

        long_upnl = self.tracker.long_pos.unrealized_pnl
        short_upnl = self.tracker.short_pos.unrealized_pnl
        current_pnl = self.session_realized_pnl + (long_upnl + short_upnl)
        self.peak_pnl = max(self.peak_pnl, current_pnl)
        drawdown = self.peak_pnl - current_pnl

        max_loss_cap = getattr(self.config, "worker_max_loss_usdt", 30.0)
        max_dd_cap = getattr(self.config, "worker_max_drawdown_usdt", 40.0)

        if current_pnl <= -abs(max_loss_cap):
            reason = f"Worker Max Loss breached: current_pnl=${current_pnl:.2f} <= -${max_loss_cap:.2f} USDT"
            await self._trigger_isolated_kill(reason)
        elif drawdown >= abs(max_dd_cap):
            reason = f"Worker Max Drawdown breached: drawdown=${drawdown:.2f} >= ${max_dd_cap:.2f} USDT (Peak: ${self.peak_pnl:.2f}, Current: ${current_pnl:.2f})"
            await self._trigger_isolated_kill(reason)

    async def _trigger_isolated_kill(self, reason: str) -> None:
        """Execute Micro Isolated Kill for this symbol only."""
        logger.critical(f"[{self.symbol}] 🚨 ISOLATED KILL SWITCH ACTIVATED! {reason}")
        self.is_locked_killed = True
        self.config.is_locked = True
        self.config.enabled = False
        config_store.save_pair_config(self.config)

        # 1. Cancel active quote orders
        await self._cancel_active_quotes()

        # 2. Cancel active executor barrier orders
        await self.executor_long._cleanup_all_barrier_orders()
        await self.executor_short._cleanup_all_barrier_orders()
        self.executor_long.state.active = False
        self.executor_short.state.active = False

        # 3. Market close open positions without reduceOnly
        if self.tracker.long_pos.amount > 0:
            long_qty = self.quoter.quantize_amount(self.tracker.long_pos.amount)
            if long_qty > 0:
                logger.warning(f"[{self.symbol}] Isolated Kill: Market closing LONG position ({long_qty:.4f})")
                await self.gateway.create_exit_order(
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    position_side=PositionSide.LONG,
                    order_type=OrderType.MARKET,
                    amount=long_qty,
                    client_order_id=f"kill_long_{int(time.time()*1000)}",
                    purpose=OrderPurpose.CIRCUIT_BREAKER_EXIT,
                )

        if self.tracker.short_pos.amount > 0:
            short_qty = self.quoter.quantize_amount(self.tracker.short_pos.amount)
            if short_qty > 0:
                logger.warning(f"[{self.symbol}] Isolated Kill: Market closing SHORT position ({short_qty:.4f})")
                await self.gateway.create_exit_order(
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    position_side=PositionSide.SHORT,
                    order_type=OrderType.MARKET,
                    amount=short_qty,
                    client_order_id=f"kill_short_{int(time.time()*1000)}",
                    purpose=OrderPurpose.CIRCUIT_BREAKER_EXIT,
                )

        self._running = False
        self._paused = True

    async def _check_dynamic_circuit_breaker(self) -> None:
        """Check 60s dynamic NATR volatility circuit breaker."""
        if not getattr(self.config, "circuit_breaker_enabled", True):
            return

        now = time.time()
        is_tripped, delta_60s, threshold = self.market_state.check_circuit_breaker(now)
        pause_sec = getattr(self.config, "circuit_breaker_pause_sec", 60)

        if is_tripped:
            was_active = self.market_state.is_circuit_breaker_active(now)
            if not was_active:
                self.market_state.circuit_breaker_paused_until = now + pause_sec
                logger.warning(
                    f"[{self.symbol}][CIRCUIT_BREAKER_TRIGGERED][DYNAMIC] {self.symbol} spike: "
                    f"delta={delta_60s*100:.2f}% >= threshold={threshold*100:.2f}% (NATR_15m={self.market_state.current_natr_15m*100:.2f}%). "
                    f"Entry paused for {pause_sec}s."
                )
                await self._cancel_active_quotes()
        else:
            # Check if paused period expired and we should resume
            if self.market_state.circuit_breaker_paused_until > 0 and now >= self.market_state.circuit_breaker_paused_until:
                self.market_state.circuit_breaker_paused_until = 0.0
                logger.info(
                    f"[{self.symbol}][CIRCUIT_BREAKER_RESUMED] Volatility subsided (delta={delta_60s*100:.2f}% < threshold={threshold*100:.2f}%). "
                    f"Resuming normal quoting."
                )
                self._create_background_task(self._requote())

    # ── Event Callbacks ──

    async def on_ticker_update(self, bid: float, ask: float, mark: float) -> None:
        """Handle incoming ticker / mark price tick."""
        self.market_state.update_ticker(bid, ask, mark)
        self.tracker.update_mark_price(mark)

        # Check dynamic NATR volatility circuit breaker
        await self._check_dynamic_circuit_breaker()

        # Check isolated risk breach
        await self._check_isolated_risk_breach()

        # Check runtime barriers (Virtual SL / Trailing TP / Passive Exit / Pyramiding)
        await self.executor_long.check_runtime_barriers(mark, best_bid=bid, best_ask=ask, market_state=self.market_state)
        await self.executor_short.check_runtime_barriers(mark, best_bid=bid, best_ask=ask, market_state=self.market_state)

    async def on_fill(self, fill: FillRecord) -> None:
        """Handle trade fill event."""
        if fill.symbol != self.symbol:
            return

        cid = str(fill.client_order_id or "").lower()

        # 1. Routing classification based on standardized client_order_id prefixes
        if cid.startswith("q_pyr_"):
            if "short" in cid:
                fill.position_side = PositionSide.SHORT
            elif "long" in cid:
                fill.position_side = PositionSide.LONG
            is_entry = True
        elif cid.startswith("q_buy_"):
            fill.position_side = PositionSide.LONG
            is_entry = True
        elif cid.startswith("q_sell_"):
            fill.position_side = PositionSide.SHORT
            is_entry = True
        elif cid.startswith(("tp_long", "sl_long", "pe_long", "pexit_long", "kill_long")):
            fill.position_side = PositionSide.LONG
            is_entry = False
        elif cid.startswith(("tp_short", "sl_short", "pe_short", "pexit_short", "kill_short")):
            fill.position_side = PositionSide.SHORT
            is_entry = False
        elif cid.startswith(("tp_", "sl_", "pe_", "pexit_", "exit_", "kill_")):
            if "short" in cid:
                fill.position_side = PositionSide.SHORT
            elif "long" in cid:
                fill.position_side = PositionSide.LONG
            is_entry = False
        else:
            if "short" in cid or "q_sell" in cid:
                fill.position_side = PositionSide.SHORT
            elif "long" in cid or "q_buy" in cid:
                fill.position_side = PositionSide.LONG

            is_entry = (
                (fill.position_side == PositionSide.LONG and fill.side == OrderSide.BUY) or
                (fill.position_side == PositionSide.SHORT and fill.side == OrderSide.SELL)
            )

        logger.info(
            f"[{self.symbol}] Fill event received: {fill.side.value} {fill.position_side.value} "
            f"{fill.amount:.4f} @ {fill.price:.2f} (cid={fill.client_order_id}, is_entry={is_entry})"
        )

        # Update position tracker
        await self.tracker.on_fill(fill)
        try:
            await db.save_fill(fill)
        except Exception as e:
            logger.warning(f"[{self.symbol}] Non-fatal DB save_fill error: {e}")

        # Strict routing to executor
        if is_entry:
            if fill.position_side == PositionSide.LONG:
                await self.executor_long.on_entry_fill(fill)
            else:
                await self.executor_short.on_entry_fill(fill)
        else:
            if fill.position_side == PositionSide.LONG:
                await self.executor_long.on_exit_fill(fill)
            else:
                await self.executor_short.on_exit_fill(fill)

        # Update session realized PnL on exit fills
        if not is_entry:
            net_fill_pnl = fill.realized_pnl - fill.fee
            self.session_realized_pnl += net_fill_pnl
            logger.info(
                f"[{self.symbol}] Exit fill PnL: ${net_fill_pnl:+.4f} -> Session Realized PnL: ${self.session_realized_pnl:+.4f}"
            )
            await self._check_isolated_risk_breach()

        # Trigger immediate requote after fill if not killed and not draining
        if not (self.config.is_locked or self.is_locked_killed or self.is_draining):
            self._create_background_task(self._requote())

    async def on_order_update(self, order: OrderRecord) -> None:
        """Handle order status update."""
        if order.symbol != self.symbol:
            return

        try:
            await db.save_order(order)
        except Exception as e:
            logger.warning(f"[{self.symbol}] Non-fatal DB save_order error: {e}")
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED):
            self._active_quote_orders.pop(order.id, None)
            self._active_quote_orders.pop(order.client_order_id, None)

    # ── Main Worker Execution Loop ──

    async def _main_worker_loop(self) -> None:
        """Main loop checking market sanity, refresh timers, hanging orders, and reconcile."""
        last_reconcile = time.time()
        last_trend_check = time.time()
        last_natr_15m_check = 0.0
        last_sanity_warning = 0.0

        while self._running:
            try:
                # 1. Fetch latest ticker only if WS stream is stale (TASK H-8)
                is_ws_stale = (time.time() - self.market_state.last_update_time >= 5.0)
                if is_ws_stale:
                    ticker = await self.gateway.fetch_ticker_and_mark(self.symbol)
                    if ticker and ticker.get("bid", 0.0) > 0 and ticker.get("ask", 0.0) > 0:
                        await self.on_ticker_update(ticker["bid"], ticker["ask"], ticker.get("mark", 0.0))

                # If still no valid top of book, wait with backoff
                if self.market_state.best_bid <= 0 or self.market_state.best_ask <= 0 or self.market_state.best_bid >= self.market_state.best_ask:
                    now = time.time()
                    if now - last_sanity_warning >= 10.0:
                        logger.warning(
                            f"[{self.symbol}] Waiting for valid market top-of-book (bid={self.market_state.best_bid}, ask={self.market_state.best_ask})..."
                        )
                        last_sanity_warning = now
                    await asyncio.sleep(1.0)
                    continue

                # 2. Periodic Reconciliation Heartbeat & Trend Bias Refresh
                now = time.time()
                reconcile_interval = getattr(self.config, "reconcile_interval_sec", 60)
                if now - last_reconcile >= reconcile_interval:
                    await self.tracker.reconcile_with_exchange()
                    await self.reconcile_barriers()
                    last_reconcile = now

                if getattr(self.config, "trend_bias_enabled", True) and (now - last_trend_check >= 300.0):
                    if hasattr(self.gateway, "fetch_ohlcv"):
                        try:
                            candles_1h = await self.gateway.fetch_ohlcv(self.symbol, timeframe="1h", limit=100)
                            if candles_1h:
                                regime = self.market_state.update_trend_bias(candles_1h)
                                self.tracker.long_pos.trend_bias_regime = regime
                                self.tracker.short_pos.trend_bias_regime = regime
                                self.tracker.long_pos.is_trend_blocked = (regime == "BEARISH")
                                self.tracker.short_pos.is_trend_blocked = (regime == "BULLISH")
                                last_trend_check = now
                        except Exception as e:
                            logger.warning(f"[{self.symbol}] Periodic trend bias update note: {e}")

                # ── Periodic 15m NATR Anchor Refresh (TASK CB-1) ──
                if hasattr(self.gateway, "fetch_ohlcv") and (now - last_natr_15m_check >= 300.0):
                    try:
                        candles_15m = await self.gateway.fetch_ohlcv(self.symbol, timeframe="15m", limit=20)
                        if candles_15m and len(candles_15m) >= 2:
                            atr_15m = calculate_atr_from_candles(candles_15m, period=14)
                            last_close = float(candles_15m[-1][4])
                            if last_close > 0 and atr_15m > 0:
                                natr_decimal = atr_15m / last_close
                                self.market_state.update_natr_15m(natr_decimal)
                                last_natr_15m_check = now
                    except Exception as e:
                        logger.warning(f"[{self.symbol}] Periodic 15m NATR refresh note: {e}")

                # 3. Check if requote is triggered
                should_requote, reason = self._check_should_requote()
                if should_requote:
                    await self._requote()

                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.symbol}] Error in worker loop: {e}")
                await asyncio.sleep(3.0)

    def _check_should_requote(self) -> Tuple[bool, str]:
        """Check whether conditions warrant placing/updating quote orders."""
        if not self._running or self._paused or not self.config.enabled or self.config.is_locked or self.is_locked_killed:
            return False, "Disabled/Paused/Locked"

        if self.is_draining:
            return False, "Draining mode active (no new entry quotes)"

        # Ensure top of book is ready before triggering quote generation
        if self.market_state.best_bid <= 0 or self.market_state.best_ask <= 0 or self.market_state.best_bid >= self.market_state.best_ask:
            return False, "Top of book (bid/ask) not ready or invalid"

        smoothed_mid = self.market_state.smoothed_mid
        if smoothed_mid <= 0:
            return False, "No price yet"

        now = time.time()

        # If no active quotes, immediately trigger quote placement
        if len(self._active_quote_orders) == 0:
            return True, "No active quotes"

        time_since_last_quote = now - self._last_quote_time
        min_lifespan = getattr(self.config, "min_quote_lifespan_sec", 5.0)

        # Urgent risk check: active quotes crossing best book
        best_bid = self.market_state.best_bid
        best_ask = self.market_state.best_ask
        for ord_rec in self._active_quote_orders.values():
            price = getattr(ord_rec, "price", 0.0)
            if ord_rec.side == OrderSide.BUY and best_ask > 0 and price >= best_ask:
                return True, f"Urgent safety: active BUY quote {price} >= best_ask {best_ask}"
            if ord_rec.side == OrderSide.SELL and best_bid > 0 and price <= best_bid:
                return True, f"Urgent safety: active SELL quote {price} <= best_bid {best_bid}"

        # Timeout fallback (order_refresh_time)
        if time_since_last_quote >= self.config.order_refresh_time:
            return True, "Timeout fallback"

        # Hanging orders distance threshold check
        if self.config.hanging_orders_enabled and self._active_quote_orders:
            for ord_rec in self._active_quote_orders.values():
                price = getattr(ord_rec, "price", 0.0)
                if isinstance(price, (int, float)) and price > 0:
                    dist_pct = abs(price - smoothed_mid) / smoothed_mid
                    if dist_pct >= self.config.hanging_orders_cancel_pct:
                        return True, f"Hanging order distance {dist_pct:.4f} >= {self.config.hanging_orders_cancel_pct:.4f}"

        # Mid move threshold (throttled by min_quote_lifespan unless move is significant)
        if self._last_quote_mid > 0:
            mid_delta_pct = abs(smoothed_mid - self._last_quote_mid) / self._last_quote_mid
            if mid_delta_pct >= self.config.requote_threshold_pct:
                if time_since_last_quote >= min_lifespan or mid_delta_pct >= (self.config.requote_threshold_pct * 2.5):
                    return True, f"Mid moved {mid_delta_pct:.4f} >= {self.config.requote_threshold_pct:.4f}"

        return False, "No trigger"

    async def _requote(self) -> None:
        """Calculate and place fresh Post-Only quote orders with smart queue retention."""
        async with self._lock:
            # If in draining mode, ensure quotes are cancelled and exit early
            if self.is_draining:
                await self._cancel_active_quotes()
                return

            # Check market sanity
            is_sane, reason = self.market_state.check_market_sanity()
            if not is_sane:
                logger.warning(f"[{self.symbol}] Market sanity check failed: {reason}. Pausing new quotes.")
                await self._cancel_active_quotes()
                return

            smoothed_mid = self.market_state.smoothed_mid
            vol_mult = self.market_state.get_volatility_multiplier()

            # Ping-Pong & Cooldown per side
            long_state = self.tracker.long_pos
            short_state = self.tracker.short_pos

            pause_long = (
                (self.config.ping_pong_enabled and long_state.amount > 0) or
                self.tracker.is_in_cooldown(PositionSide.LONG)
            )
            pause_short = (
                (self.config.ping_pong_enabled and short_state.amount > 0) or
                self.tracker.is_in_cooldown(PositionSide.SHORT)
            )

            # ── Isolated Per-Pair Margin Quota & Account Free Margin Calculation ──
            # Bước 1: Lấy Account Margin (với 0.95 safety buffer)
            raw_account_free = 0.0
            if hasattr(self.gateway, "fetch_free_balance"):
                try:
                    raw_account_free = await self.gateway.fetch_free_balance("USDT")
                except Exception as e:
                    logger.warning(f"[{self.symbol}] Failed to fetch free balance: {e}")
            account_free = raw_account_free * 0.95

            # Bước 2: Tính Used Margin của riêng cặp hiện tại từ open position
            mark_price = self.market_state.smoothed_mid if self.market_state.smoothed_mid > 0 else (self.market_state.best_bid or 1.0)
            pos_notional = (abs(long_state.amount) + abs(short_state.amount)) * mark_price
            lev = max(1, self.config.leverage)
            pair_used_margin = pos_notional / lev

            # Bước 3: Tính Margin khả dụng thực tế của riêng cặp
            pair_remaining_margin = max(0.0, self.config.effective_margin_cap - pair_used_margin)
            effective_available_margin = min(account_free, pair_remaining_margin)

            # Bước 4: Guard Check
            min_req = self.quoter.estimate_minimum_margin_required(
                active_long=not pause_long,
                active_short=not pause_short,
            )

            if min_req > 0 and effective_available_margin < min_req:
                now = time.time()
                if now - self._last_margin_warning_time >= 10.0:
                    logger.warning(
                        f"[{self.symbol}] Margin quota exhausted: available={effective_available_margin:.2f} USDT "
                        f"(Account Free: {account_free:.2f}, Pair Cap Remaining: {pair_remaining_margin:.2f}) < required={min_req:.2f} USDT. Pausing quotes."
                    )
                    self._last_margin_warning_time = now
                return

            # Bước 5: Tính Quote với effective_available_margin và ladder throttling
            real_mark = self.market_state.mark_price if self.market_state.mark_price > 0 else mark_price
            bids, asks = self.quoter.calculate_quotes(
                smoothed_mid=smoothed_mid,
                long_value_usdt=long_state.notional,
                short_value_usdt=short_state.notional,
                vol_mult=vol_mult,
                pause_long_entry=pause_long,
                pause_short_entry=pause_short,
                best_bid=self.market_state.best_bid,
                best_ask=self.market_state.best_ask,
                available_margin=effective_available_margin,
                long_state=long_state,
                short_state=short_state,
                mark_price=real_mark,
                trend_bias_regime=self.market_state.trend_bias_regime,
            )

            target_quotes = bids + asks
            tick_sz = self.quoter.tick_size if self.quoter.tick_size > 0 else 0.01

            # ── Smart Quote Differential: Preserve valid existing orders ──
            orders_to_keep = {}
            matched_targets = set()

            for ord_id, existing_ord in list(self._active_quote_orders.items()):
                matched = False
                for idx, tq in enumerate(target_quotes):
                    if idx in matched_targets:
                        continue
                    # Match if side, position_side, price (within 1 tick) and amount match
                    price_matched = abs(existing_ord.price - tq.price) <= (tick_sz * 1.01)
                    amt_diff = abs(existing_ord.amount - tq.amount)
                    amt_matched = amt_diff <= (self.quoter.step_size if self.quoter.step_size > 0 else 0.001) or (amt_diff / max(1e-5, tq.amount) < 0.05)

                    if (
                        existing_ord.side == tq.side and
                        existing_ord.position_side == tq.position_side and
                        price_matched and
                        amt_matched
                    ):
                        matched = True
                        matched_targets.add(idx)
                        orders_to_keep[ord_id] = existing_ord
                        break

                if not matched:
                    # Cancel only orders that have drifted from target
                    await self.gateway.cancel_order(self.symbol, ord_id)
                    self._active_quote_orders.pop(ord_id, None)

            # ── Place only new target quotes that are not already active ──
            for idx, tq in enumerate(target_quotes):
                if idx in matched_targets:
                    continue  # Keep existing active order (retaining maker queue priority)

                client_id = f"q_{tq.side.value.lower()}_{tq.level}_{int(time.time()*1000)}"
                resp = await self.gateway.create_quote_order(
                    symbol=self.symbol,
                    side=tq.side,
                    position_side=tq.position_side,
                    price=tq.price,
                    amount=tq.amount,
                    client_order_id=client_id,
                )
                if resp:
                    order_rec = OrderRecord(
                        id=str(resp.get("id") or client_id),
                        client_order_id=client_id,
                        exchange_order_id=str(resp.get("id")),
                        symbol=self.symbol,
                        side=tq.side,
                        position_side=tq.position_side,
                        order_type=OrderType.LIMIT_MAKER,
                        price=tq.price,
                        amount=tq.amount,
                        remaining_amount=tq.amount,
                        status=OrderStatus.NEW,
                        purpose=OrderPurpose.ENTRY_QUOTE,
                        level=tq.level,
                        created_at=time.time(),
                        updated_at=time.time(),
                        raw_response=resp,
                    )
                    self._active_quote_orders[order_rec.id] = order_rec
                    try:
                        await db.save_order(order_rec)
                    except Exception as e:
                        logger.warning(f"[{self.symbol}] Non-fatal DB save_order error: {e}")

            self._last_quote_mid = smoothed_mid
            self._last_quote_time = time.time()

    async def _cancel_active_quotes(self) -> None:
        """Cancel all registered active quote orders."""
        order_ids = list(self._active_quote_orders.keys())
        for oid in order_ids:
            await self.gateway.cancel_order(self.symbol, oid)
            self._active_quote_orders.pop(oid, None)
