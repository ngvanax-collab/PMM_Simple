"""Exchange Gateway wrapping CCXT.pro with Hedge Mode Invariants."""
import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import ccxt.pro as ccxtpro
import ccxt
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.ratelimit import rate_limiter
from app.models.config import ExchangeCredentials
from app.models.state import FillRecord, OrderPurpose, OrderRecord, OrderSide, OrderStatus, OrderType, PositionSide, SidePositionState


class ExchangeGateway:
    """Unified Exchange Gateway for Futures Hedge Mode."""

    def __init__(self, credentials: ExchangeCredentials):
        self.credentials = credentials
        self.exchange_name = credentials.exchange.lower()
        self.testnet = credentials.testnet
        self._exchange: Optional[ccxtpro.Exchange] = None
        self._market_info: Dict[str, Any] = {}
        self._is_connected = False
        self._ws_tasks: List[asyncio.Task] = []
        self._running = False

        # Callbacks for private WS events
        self.on_order_update: Optional[Callable[[OrderRecord], Any]] = None
        self.on_fill_update: Optional[Callable[[FillRecord], Any]] = None
        self.on_position_update: Optional[Callable[[str, PositionSide, float, float, float], Any]] = None

        # Free balance cache (currency -> (timestamp, free_amount)) with TTL 1.5s
        self._free_balance_cache: Dict[str, Tuple[float, float]] = {}

    async def initialize(self) -> bool:
        """Initialize ccxt.pro instance, load markets, and verify Hedge Mode."""
        try:
            exchange_class = getattr(ccxtpro, self.exchange_name, None)
            if exchange_class is None:
                raise ValueError(f"Unsupported exchange in ccxt.pro: {self.exchange_name}")

            config = {
                "apiKey": self.credentials.api_key,
                "secret": self.credentials.api_secret,
                "enableRateLimit": False,  # Managed by our custom TokenBucket
                "options": {
                    "defaultType": "future",
                    "adjustForTimeDifference": True,
                }
            }
            if self.credentials.passphrase:
                config["password"] = self.credentials.passphrase

            self._exchange = exchange_class(config)

            if self.testnet:
                self._exchange.set_sandbox_mode(True)

            logger.info(f"Connecting to {self.exchange_name} (testnet={self.testnet})...")
            await rate_limiter.acquire_weight(weight=5)
            self._market_info = await self._exchange.load_markets()

            # Verify and enforce Hedge Mode
            hedge_ok = await self.verify_and_set_hedge_mode()
            if not hedge_ok:
                logger.error("Hedge Mode verification failed! Aborting gateway initialization.")
                await self.close()
                return False

            self._is_connected = True
            logger.info(f"Gateway initialized successfully for {self.exchange_name} with Hedge Mode verified.")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize exchange gateway: {e}")
            await self.close()
            return False

    async def verify_and_set_hedge_mode(self) -> bool:
        """Verify that exchange account is in Hedge (Dual-side) Mode or switch to it."""
        if not self._exchange:
            return False

        try:
            await rate_limiter.acquire_weight(weight=2)
            if self.exchange_name == "binance":
                # Binance Futures positionSide check
                try:
                    res = await self._exchange.fapiPrivateGetPositionSideDual()
                    is_dual = res.get("dualSidePosition", False)
                    if isinstance(is_dual, str):
                        is_dual = (is_dual.lower() == "true")

                    if is_dual:
                        logger.info("Binance Futures is already in DUAL (Hedge) position mode.")
                        return True

                    logger.warning("Binance is in One-Way mode. Attempting to switch to Hedge Mode (dualSidePosition=true)...")
                    await rate_limiter.acquire_weight(weight=2)
                    switch_res = await self._exchange.fapiPrivatePostPositionSideDual({"dualSidePosition": "true"})
                    logger.info(f"Successfully switched Binance to Hedge Mode: {switch_res}")
                    return True
                except Exception as e:
                    logger.critical(f"Cannot switch Binance to Hedge Mode (ensure no open orders/positions exist): {e}")
                    return False

            elif self.exchange_name == "bybit":
                # Bybit Hedge mode check via set_position_mode
                try:
                    # Bybit positionIdx: 1=Buy/Long, 2=Sell/Short in hedge mode
                    res = await self._exchange.set_position_mode(True)
                    logger.info(f"Bybit position mode set to Hedge: {res}")
                    return True
                except Exception as e:
                    err_msg = str(e).lower()
                    if "not modified" in err_msg or "110025" in err_msg or "already in" in err_msg or "no need to change" in err_msg:
                        logger.info("Bybit is already configured in Hedge (Dual-side) mode.")
                        return True
                    else:
                        logger.critical(f"Cannot switch Bybit to Hedge Mode: {e}")
                        return False

            return True

        except Exception as e:
            logger.error(f"Error during verify_and_set_hedge_mode: {e}")
            return False

    async def setup_symbol(self, symbol: str, leverage: int, margin_mode: str) -> bool:
        """Set leverage and margin mode for a symbol."""
        if not self._exchange:
            return False

        try:
            await rate_limiter.acquire_weight(weight=2)
            try:
                await self._exchange.set_leverage(leverage, symbol)
                logger.info(f"[{symbol}] Leverage set to {leverage}x")
            except Exception as e:
                err_msg = str(e)
                if "-4421" in err_msg or "leverage greater than" in err_msg.lower() or "subaccount" in err_msg.lower():
                    # Subaccount leverage cap — auto-fallback to max permitted (5x)
                    fallback_leverage = 5
                    logger.warning(
                        f"[{symbol}] Subaccount leverage cap hit (code -4421). Auto-falling back to {fallback_leverage}x."
                    )
                    try:
                        await self._exchange.set_leverage(fallback_leverage, symbol)
                        logger.info(f"[{symbol}] Leverage fallback set to {fallback_leverage}x successfully.")
                    except Exception as e2:
                        logger.warning(f"[{symbol}] Leverage fallback also failed: {e2}")
                elif "-4028" in err_msg or "no need to change" in err_msg.lower():
                    logger.info(f"[{symbol}] Leverage already set correctly — no change needed.")
                else:
                    logger.warning(f"[{symbol}] Set leverage note: {e}")

            await rate_limiter.acquire_weight(weight=2)
            try:
                # CCXT set_margin_mode accepts 'ISOLATED' or 'CROSS'
                await self._exchange.set_margin_mode(margin_mode.upper(), symbol)
                logger.info(f"[{symbol}] Margin mode set to {margin_mode.upper()}")
            except Exception as e:
                err_msg = str(e)
                if "-4046" in err_msg or "no need to change margin type" in err_msg.lower():
                    # Already the correct margin mode — not an error
                    logger.info(f"[{symbol}] Margin mode already correct ({margin_mode.upper()}) — no change needed.")
                else:
                    logger.warning(f"[{symbol}] Set margin mode note: {e}")

            return True
        except Exception as e:
            logger.error(f"[{symbol}] Failed to setup symbol: {e}")
            return False

    # ── Order Operations (Strict Hedge Mode) ──

    async def create_quote_order(
        self,
        symbol: str,
        side: OrderSide,
        position_side: PositionSide,
        price: float,
        amount: float,
        client_order_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Create Post-Only Limit Entry Quote Order in Hedge Mode.
        CRITICAL: Never uses reduceOnly. Must include positionSide.
        """
        assert position_side in (PositionSide.LONG, PositionSide.SHORT), "Hedge mode requires LONG or SHORT"
        if not self._exchange:
            return None

        params: Dict[str, Any] = {
            "positionSide": position_side.value,
            "newClientOrderId": client_order_id,
            "clientOrderId": client_order_id,
        }

        if self.exchange_name == "binance":
            params["timeInForce"] = "GTX"  # Post-Only Maker order on Binance
        elif self.exchange_name == "bybit":
            params["postOnly"] = True
            params["positionIdx"] = 1 if position_side == PositionSide.LONG else 2

        price_prec, _, tick_size, _ = self.get_market_precision(symbol)
        curr_price = self.price_to_precision(symbol, price)
        fmt_amount = self.amount_to_precision(symbol, amount)
        max_retries = 2

        for attempt in range(max_retries + 1):
            await rate_limiter.acquire_order(count=1, weight=1)
            try:
                attempt_client_id = f"{client_order_id}_{attempt}" if attempt > 0 else client_order_id
                params["newClientOrderId"] = attempt_client_id
                params["clientOrderId"] = attempt_client_id

                order = await self._exchange.create_order(
                    symbol=symbol,
                    type="limit",
                    side=side.value.lower(),
                    amount=fmt_amount,
                    price=curr_price,
                    params=params,
                )
                if attempt > 0:
                    logger.info(
                        f"[{symbol}] Post-Only maker succeeded on retry #{attempt} at stepped price {curr_price}"
                    )
                return order
            except Exception as e:
                err_msg = str(e)
                is_post_only_rejection = (
                    "-5022" in err_msg or
                    "not be executed as maker" in err_msg.lower() or
                    "would immediately match" in err_msg.lower() or
                    "post only" in err_msg.lower()
                )

                if is_post_only_rejection and attempt < max_retries and tick_size > 0:
                    # Step back by 1 tick: lower bid for BUY, higher ask for SELL
                    if side == OrderSide.BUY:
                        curr_price = self.price_to_precision(symbol, curr_price - tick_size)
                    else:
                        curr_price = self.price_to_precision(symbol, curr_price + tick_size)

                    logger.warning(
                        f"[{symbol}] Post-Only maker rejected (-5022). Stepping price by 1 tick ({tick_size}) to {curr_price} (retry {attempt+1}/{max_retries})..."
                    )
                    continue
                else:
                    if is_post_only_rejection:
                        logger.warning(f"[{symbol}] Post-Only maker rejected (crossed book): {err_msg}")
                    elif "-2019" in err_msg or "margin is insufficient" in err_msg.lower():
                        logger.info(f"[{symbol}] Quote order skipped due to exchange insufficient margin (-2019): {err_msg}")
                    else:
                        logger.error(f"[{symbol}] Failed to create quote order ({side} {position_side}): {err_msg}")
                    return None

    async def create_entry_market_order(
        self,
        symbol: str,
        side: OrderSide,
        position_side: PositionSide,
        amount: float,
        client_order_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Create Market Entry Order in Hedge Mode (e.g. for Momentum Pyramiding).
        CRITICAL: Never uses reduceOnly or closePosition!
        LONG entry = BUY with positionSide=LONG.
        SHORT entry = SELL with positionSide=SHORT.
        """
        assert position_side in (PositionSide.LONG, PositionSide.SHORT), "Hedge mode requires LONG or SHORT"
        if not self._exchange:
            return None

        fmt_amount = self.amount_to_precision(symbol, amount)
        cid = client_order_id or f"q_pyr_{int(time.time()*1000)}"
        params: Dict[str, Any] = {
            "positionSide": position_side.value,
            "newClientOrderId": cid,
            "clientOrderId": cid,
        }
        if self.exchange_name == "bybit":
            params["positionIdx"] = 1 if position_side == PositionSide.LONG else 2

        await rate_limiter.acquire_order(count=1, weight=1)
        try:
            order = await self._exchange.create_order(
                symbol=symbol,
                type="market",
                side=side.value.lower(),
                amount=fmt_amount,
                params=params,
            )
            return order
        except Exception as e:
            logger.error(f"[{symbol}] Failed to create market entry order ({side} {position_side}): {e}")
            return None

    async def create_exit_order(
        self,
        symbol: str,
        side: OrderSide,
        position_side: PositionSide,
        order_type: OrderType,
        amount: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        purpose: OrderPurpose = OrderPurpose.TAKE_PROFIT,
    ) -> Optional[Dict[str, Any]]:
        """
        Create Exit Order in Hedge Mode (TP Limit / SL STOP_MARKET / Market Exit).
        CRITICAL:
        - Never uses reduceOnly or closePosition!
        - Closing LONG = SELL with positionSide=LONG.
        - Closing SHORT = BUY with positionSide=SHORT.
        """
        assert position_side in (PositionSide.LONG, PositionSide.SHORT), "Hedge mode requires LONG or SHORT"
        if not self._exchange:
            return None

        # Format price and amount using CCXT precision rules
        fmt_amount = self.amount_to_precision(symbol, amount)
        fmt_price = self.price_to_precision(symbol, price) if price is not None else None
        fmt_stop_price = self.price_to_precision(symbol, stop_price) if stop_price is not None else None

        client_id = client_order_id or f"exit_{int(time.time()*1000)}"
        params: Dict[str, Any] = {
            "positionSide": position_side.value,
            "newClientOrderId": client_id,
            "clientOrderId": client_id,
        }

        if self.exchange_name == "bybit":
            params["positionIdx"] = 1 if position_side == PositionSide.LONG else 2

        await rate_limiter.acquire_order(count=1, weight=1, priority=(purpose in (OrderPurpose.KILL_ALL_EXIT, OrderPurpose.STOP_LOSS)))

        try:
            if order_type == OrderType.LIMIT_MAKER or (order_type == OrderType.LIMIT and purpose == OrderPurpose.TAKE_PROFIT):
                if self.exchange_name == "binance":
                    params["timeInForce"] = "GTX"
                elif self.exchange_name == "bybit":
                    params["postOnly"] = True
                return await self._exchange.create_order(
                    symbol=symbol,
                    type="limit",
                    side=side.value.lower(),
                    amount=fmt_amount,
                    price=fmt_price,
                    params=params,
                )

            elif order_type == OrderType.STOP_MARKET:
                # Server-side conditional Stop Loss
                if self.exchange_name == "binance":
                    params["stopPrice"] = fmt_stop_price
                    return await self._exchange.create_order(
                        symbol=symbol,
                        type="STOP_MARKET",
                        side=side.value.lower(),
                        amount=fmt_amount,
                        price=None,
                        params=params,
                    )
                elif self.exchange_name == "bybit":
                    params["triggerPrice"] = str(fmt_stop_price)
                    params["triggerDirection"] = 2 if side == OrderSide.SELL else 1  # 2: falling for Long SL, 1: rising for Short SL
                    return await self._exchange.create_order(
                        symbol=symbol,
                        type="market",
                        side=side.value.lower(),
                        amount=fmt_amount,
                        price=None,
                        params=params,
                    )

            elif order_type == OrderType.MARKET:
                # Direct market exit for Trailing / Time-limit / Kill-All
                return await self._exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=side.value.lower(),
                    amount=fmt_amount,
                    price=None,
                    params=params,
                )

            else:
                return await self._exchange.create_order(
                    symbol=symbol,
                    type=order_type.value.lower(),
                    side=side.value.lower(),
                    amount=amount,
                    price=price,
                    params=params,
                )

        except Exception as e:
            logger.error(f"[{symbol}] Failed to create exit order ({order_type} {side} {position_side} qty={amount}): {e}")
            return None

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a single order."""
        if not self._exchange:
            return False

        await rate_limiter.acquire_order(count=1, weight=1)
        try:
            await self._exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.warning(f"[{symbol}] Cancel order {order_id} note: {e}")
            return False

    async def cancel_all_symbol_orders(self, symbol: str) -> bool:
        """Cancel all open orders for a symbol."""
        if not self._exchange:
            return False

        await rate_limiter.acquire_order(count=1, weight=1, priority=True)
        try:
            await self._exchange.cancel_all_orders(symbol)
            logger.info(f"[{symbol}] Cancelled all open orders")
            return True
        except Exception as e:
            logger.error(f"[{symbol}] Failed to cancel all orders: {e}")
            return False

    # ── Fetch & Reconcile Methods ──

    async def fetch_open_orders(self, symbol: str) -> List[OrderRecord]:
        """Fetch real open orders from exchange with normalized positionSide."""
        if not self._exchange:
            return []

        await rate_limiter.acquire_weight(weight=2)
        try:
            raw_orders = await self._exchange.fetch_open_orders(symbol)
            records: List[OrderRecord] = []
            now = time.time()

            for o in raw_orders:
                # Extract Hedge positionSide
                info = o.get("info", {})
                cid = str(o.get("clientOrderId") or info.get("clientOrderId") or info.get("c") or "")
                pos_side_raw = (info.get("positionSide") or info.get("ps") or o.get("positionSide") or "").upper()
                if pos_side_raw not in ("LONG", "SHORT"):
                    if "short" in cid.lower() or "q_sell" in cid.lower():
                        pos_side_raw = "SHORT"
                    elif "long" in cid.lower() or "q_buy" in cid.lower():
                        pos_side_raw = "LONG"
                    else:
                        pos_idx = info.get("positionIdx", 0)
                        pos_side_raw = "LONG" if pos_idx == 1 else ("SHORT" if pos_idx == 2 else "BOTH")

                pos_side = PositionSide(pos_side_raw) if pos_side_raw in ("LONG", "SHORT") else PositionSide.LONG

                record = OrderRecord(
                    id=str(o["id"]),
                    client_order_id=str(o.get("clientOrderId") or o["id"]),
                    exchange_order_id=str(o["id"]),
                    symbol=symbol,
                    side=OrderSide(o["side"].upper()),
                    position_side=pos_side,
                    order_type=OrderType(o["type"].upper()) if o["type"].upper() in OrderType.__members__ else OrderType.LIMIT,
                    price=float(o.get("price") or 0.0),
                    stop_price=float(o.get("stopPrice") or 0.0) if o.get("stopPrice") else None,
                    amount=float(o["amount"]),
                    filled_amount=float(o.get("filled") or 0.0),
                    remaining_amount=float(o.get("remaining") or o["amount"]),
                    status=OrderStatus.NEW if o["status"] == "open" else OrderStatus.CANCELED,
                    purpose=OrderPurpose.ENTRY_QUOTE,
                    created_at=float(o.get("timestamp") or (now * 1000)) / 1000.0,
                    updated_at=now,
                    raw_response=o,
                )
                records.append(record)
            return records
        except Exception as e:
            logger.error(f"[{symbol}] Failed to fetch open orders: {e}")
            return []

    async def fetch_positions_hedge(self, symbol: str) -> Tuple[Optional[SidePositionState], Optional[SidePositionState]]:
        """
        Fetch real positions for both LONG and SHORT sides from exchange.
        Returns: (long_state, short_state)
        """
        if not self._exchange:
            return None, None

        await rate_limiter.acquire_weight(weight=5)
        try:
            positions = await self._exchange.fetch_positions([symbol])
            long_pos: Optional[SidePositionState] = None
            short_pos: Optional[SidePositionState] = None

            for pos in positions:
                if pos["symbol"] != symbol:
                    continue

                info = pos.get("info", {})
                side_raw = (pos.get("side") or info.get("positionSide") or "").upper()
                if not side_raw:
                    pos_idx = info.get("positionIdx", 0)
                    side_raw = "LONG" if pos_idx == 1 else ("SHORT" if pos_idx == 2 else "BOTH")

                amt = abs(float(pos.get("contracts") or pos.get("contractSize") or info.get("positionAmt") or pos.get("size") or 0.0))
                entry_p = float(pos.get("entryPrice") or info.get("entryPrice") or 0.0)
                mark_p = float(pos.get("markPrice") or info.get("markPrice") or entry_p)
                upnl = float(pos.get("unrealizedPnl") or info.get("unRealizedProfit") or 0.0)
                lev = int(pos.get("leverage") or info.get("leverage") or 5)
                notional = float(pos.get("notional") or (amt * mark_p))

                state = SidePositionState(
                    symbol=symbol,
                    position_side=PositionSide.LONG if side_raw == "LONG" else PositionSide.SHORT,
                    amount=amt,
                    entry_price=entry_p,
                    current_price=mark_p,
                    notional=abs(notional),
                    unrealized_pnl=upnl,
                    leverage=lev,
                    initial_margin=abs(notional) / lev if lev > 0 else 0.0,
                )

                if side_raw == "LONG":
                    long_pos = state
                elif side_raw == "SHORT":
                    short_pos = state

            # If exchange returned only active side, fill 0-amount default for the other
            if not long_pos:
                long_pos = SidePositionState(symbol=symbol, position_side=PositionSide.LONG, amount=0.0)
            if not short_pos:
                short_pos = SidePositionState(symbol=symbol, position_side=PositionSide.SHORT, amount=0.0)

            return long_pos, short_pos

        except Exception as e:
            logger.error(f"[{symbol}] Failed to fetch hedge positions: {e}")
            return None, None

    async def fetch_free_balance(self, currency: str = "USDT") -> float:
        """
        Fetch available free margin balance from exchange with 1.5s Cache TTL.
        Prevents rate-limit exhaustion when multiple symbol workers requote simultaneously.
        """
        if not self._exchange:
            return 0.0

        now = time.time()
        if currency in self._free_balance_cache:
            cache_ts, cached_val = self._free_balance_cache[currency]
            if now - cache_ts < 1.5:
                return cached_val

        await rate_limiter.acquire_weight(weight=2)
        try:
            balance = await self._exchange.fetch_balance()
            free_amt = 0.0

            # 1. CCXT parsed balance['free'][currency]
            if isinstance(balance.get("free"), dict) and currency in balance["free"]:
                val = balance["free"].get(currency)
                if val is not None:
                    free_amt = float(val)
            # 2. CCXT balance[currency]['free']
            elif isinstance(balance.get(currency), dict) and "free" in balance[currency]:
                val = balance[currency].get("free")
                if val is not None:
                    free_amt = float(val)

            # 3. Fallback to exchange-specific raw fields if free_amt == 0
            if free_amt <= 0 and isinstance(balance.get("info"), dict):
                info = balance["info"]
                # Binance Futures: availableBalance or maxWithdrawAmount
                raw_avail = info.get("availableBalance") or info.get("maxWithdrawAmount")
                if raw_avail is not None:
                    try:
                        free_amt = float(raw_avail)
                    except (ValueError, TypeError):
                        pass

                # Binance assets list: [{"asset": "USDT", "availableBalance": "..."}]
                if free_amt <= 0 and isinstance(info.get("assets"), list):
                    for asset_info in info["assets"]:
                        if asset_info.get("asset") == currency:
                            raw_avail = asset_info.get("availableBalance") or asset_info.get("maxWithdrawAmount") or asset_info.get("free")
                            if raw_avail is not None:
                                try:
                                    free_amt = float(raw_avail)
                                    break
                                except (ValueError, TypeError):
                                    pass

            final_free = max(0.0, free_amt)
            self._free_balance_cache[currency] = (now, final_free)
            return final_free
        except Exception as e:
            logger.warning(f"Failed to fetch free balance for {currency}: {e}")
            return 0.0

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> List[List[float]]:
        """Fetch historical OHLCV candles from exchange with rate-limiting."""
        if not self._exchange:
            return []

        await rate_limiter.acquire_weight(weight=2)
        try:
            if hasattr(self._exchange, "fetch_ohlcv"):
                candles = await self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                return candles or []
            return []
        except Exception as e:
            logger.warning(f"[{symbol}] Failed to fetch OHLCV ({timeframe}): {e}")
            return []

    async def fetch_ticker_and_mark(self, symbol: str) -> Optional[Dict[str, float]]:
        """Fetch bid, ask, last, mark price and funding rate with orderbook fallback."""
        if not self._exchange:
            return None

        await rate_limiter.acquire_weight(weight=1)
        try:
            ticker = await self._exchange.fetch_ticker(symbol)
            bid = float(ticker.get("bid") or 0.0)
            ask = float(ticker.get("ask") or 0.0)
            last = float(ticker.get("last") or 0.0)
            info = ticker.get("info", {})
            mark = float(info.get("markPrice") or last or (bid + ask) / 2.0 if (bid > 0 and ask > 0) else last)

            # Fallback to fetch_order_book if bid or ask is missing/invalid
            if bid <= 0 or ask <= 0 or bid >= ask:
                await rate_limiter.acquire_weight(weight=1)
                ob = await self._exchange.fetch_order_book(symbol, limit=5)
                bids = ob.get("bids", [])
                asks = ob.get("asks", [])
                if bids and len(bids) > 0:
                    bid = float(bids[0][0])
                if asks and len(asks) > 0:
                    ask = float(asks[0][0])

            mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0 and bid < ask) else (last or mark)
            if mark <= 0:
                mark = mid

            return {
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "mark": mark,
                "last": last,
            }
        except Exception as e:
            logger.error(f"[{symbol}] Failed to fetch ticker/orderbook: {e}")
            return None

    async def watch_public_ticker(
        self,
        symbol: str,
        on_update: Callable[[float, float, float], Any],
        on_reconnect: Optional[Callable[[], Any]] = None,
    ) -> None:
        """Continuously watch public ticker or orderbook for a symbol via WebSocket."""
        while self._running and self._exchange:
            try:
                if self._exchange.has.get("watchTicker"):
                    ticker = await self._exchange.watch_ticker(symbol)
                    bid = float(ticker.get("bid") or 0.0)
                    ask = float(ticker.get("ask") or 0.0)
                    last = float(ticker.get("last") or 0.0)
                    info = ticker.get("info", {})
                    mark = float(info.get("markPrice") or last or (bid + ask) / 2.0 if (bid > 0 and ask > 0) else last)
                    if bid > 0 and ask > 0:
                        res = on_update(bid, ask, mark)
                        if asyncio.iscoroutine(res):
                            await res
                elif self._exchange.has.get("watchOrderBook"):
                    ob = await self._exchange.watch_order_book(symbol, limit=5)
                    bids = ob.get("bids", [])
                    asks = ob.get("asks", [])
                    bid = float(bids[0][0]) if bids else 0.0
                    ask = float(asks[0][0]) if asks else 0.0
                    if bid > 0 and ask > 0:
                        res = on_update(bid, ask, (bid + ask) / 2.0)
                        if asyncio.iscoroutine(res):
                            await res
                else:
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    logger.warning(f"[{symbol}] Public WS watch_ticker exception (reconnecting in 2s): {e}")
                    if on_reconnect:
                        try:
                            rec_res = on_reconnect()
                            if asyncio.iscoroutine(rec_res):
                                await rec_res
                        except Exception as rec_err:
                            logger.warning(f"[{symbol}] on_reconnect callback error: {rec_err}")
                    await asyncio.sleep(2.0)

    # ── Precision & Filter Helpers ──

    def get_market_precision(self, symbol: str) -> Tuple[int, int, float, float]:
        """
        Return (price_precision, amount_precision, tick_size, step_size) from CCXT market metadata.
        Accurately extracts tick_size and step_size without hardcoding or integer truncation.
        """
        market = self._market_info.get(symbol, {})
        if not market and self._exchange and getattr(self._exchange, "markets", None):
            market = self._exchange.markets.get(symbol, {})

        precision = market.get("precision", {})
        limits = market.get("limits", {})
        price_limits = limits.get("price", {})
        amount_limits = limits.get("amount", {})

        # ── 1. Price Precision & Tick Size ──
        tick_size: Optional[float] = None
        price_prec: Optional[int] = None

        # Check Binance/Bybit market filters in info
        info = market.get("info", {})
        if isinstance(info, dict):
            filters = info.get("filters", [])
            if isinstance(filters, list):
                for f in filters:
                    if f.get("filterType") == "PRICE_FILTER" and f.get("tickSize"):
                        try:
                            ts = float(f["tickSize"])
                            if ts > 0:
                                tick_size = ts
                        except Exception:
                            pass

        # Check price limits
        if tick_size is None and price_limits.get("min") is not None:
            try:
                min_p = float(price_limits["min"])
                if min_p > 0:
                    tick_size = min_p
            except Exception:
                pass

        # Check CCXT precision['price']
        price_p = precision.get("price")
        if isinstance(price_p, float) and 0 < price_p < 1:
            if tick_size is None:
                tick_size = price_p
            s = f"{price_p:.10f}".rstrip("0")
            price_prec = len(s.split(".")[1]) if "." in s else 2
        elif isinstance(price_p, (int, float)) and price_p >= 1:
            price_prec = int(price_p)
            if tick_size is None:
                tick_size = 10 ** (-price_prec)

        if tick_size is not None and tick_size > 0:
            if price_prec is None:
                s = f"{tick_size:.10f}".rstrip("0")
                price_prec = len(s.split(".")[1]) if "." in s else 2
        else:
            price_prec = 2
            tick_size = 0.01

        # ── 2. Amount Precision & Step Size ──
        step_size: Optional[float] = None
        amt_prec: Optional[int] = None

        if isinstance(info, dict):
            filters = info.get("filters", [])
            if isinstance(filters, list):
                for f in filters:
                    if f.get("filterType") == "LOT_SIZE" and f.get("stepSize"):
                        try:
                            ss = float(f["stepSize"])
                            if ss > 0:
                                step_size = ss
                        except Exception:
                            pass

        if step_size is None and amount_limits.get("min") is not None:
            try:
                min_a = float(amount_limits["min"])
                if min_a > 0:
                    step_size = min_a
            except Exception:
                pass

        amount_p = precision.get("amount")
        if isinstance(amount_p, float) and 0 < amount_p < 1:
            if step_size is None:
                step_size = amount_p
            s = f"{amount_p:.10f}".rstrip("0")
            amt_prec = len(s.split(".")[1]) if "." in s else 3
        elif isinstance(amount_p, (int, float)) and amount_p >= 0:
            amt_prec = int(amount_p)
            if step_size is None:
                step_size = 10 ** (-amt_prec) if amt_prec > 0 else 1.0

        if step_size is not None and step_size > 0:
            if amt_prec is None:
                s = f"{step_size:.10f}".rstrip("0")
                amt_prec = len(s.split(".")[1]) if "." in s else 0
        else:
            amt_prec = 3
            step_size = 0.001

        return price_prec, amt_prec, tick_size, step_size

    def get_market_limits(self, symbol: str) -> Tuple[float, float]:
        """Return (min_amount, min_notional) for a symbol."""
        market = self._market_info.get(symbol, {})
        if not market and self._exchange and getattr(self._exchange, "markets", None):
            market = self._exchange.markets.get(symbol, {})

        limits = market.get("limits", {})
        amount_limits = limits.get("amount", {})
        cost_limits = limits.get("cost", {})

        min_amount = float(amount_limits.get("min") or 0.0)
        min_notional = float(cost_limits.get("min") or 5.0)

        # Check info filters for MIN_NOTIONAL or NOTIONAL filter
        info = market.get("info", {})
        if isinstance(info, dict):
            filters = info.get("filters", [])
            if isinstance(filters, list):
                for f in filters:
                    if f.get("filterType") in ("MIN_NOTIONAL", "NOTIONAL"):
                        try:
                            n = float(f.get("notional") or f.get("minNotional") or 0.0)
                            if n > 0:
                                min_notional = max(min_notional, n)
                        except Exception:
                            pass
                    elif f.get("filterType") == "LOT_SIZE" and f.get("minQty"):
                        try:
                            mq = float(f["minQty"])
                            if mq > 0:
                                min_amount = max(min_amount, mq)
                        except Exception:
                            pass

        return min_amount, min_notional

    def price_to_precision(self, symbol: str, price: float) -> float:
        """Format price using CCXT exchange.price_to_precision."""
        if self._exchange and hasattr(self._exchange, "price_to_precision"):
            try:
                formatted = self._exchange.price_to_precision(symbol, price)
                if isinstance(formatted, (str, int, float)) and type(formatted).__name__ not in ("MagicMock", "AsyncMock"):
                    return float(formatted)
            except Exception:
                pass
        price_prec, _, tick_sz, _ = self.get_market_precision(symbol)
        return round(price, price_prec)

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        """Format amount using CCXT exchange.amount_to_precision."""
        if self._exchange and hasattr(self._exchange, "amount_to_precision"):
            try:
                formatted = self._exchange.amount_to_precision(symbol, amount)
                if isinstance(formatted, (str, int, float)) and type(formatted).__name__ not in ("MagicMock", "AsyncMock"):
                    return float(formatted)
            except Exception:
                pass
        _, amt_prec, _, step_sz = self.get_market_precision(symbol)
        return round(amount, amt_prec)

    # ── Private User WebSocket Watchers ──

    async def start_private_watchers(self) -> None:
        """Start background WebSocket loops to listen for private user order, trade, and position events."""
        if not self._exchange:
            logger.warning("Exchange not initialized, skipping private watchers.")
            return

        self._running = True
        self._ws_tasks.append(asyncio.create_task(self._watch_orders_loop()))
        self._ws_tasks.append(asyncio.create_task(self._watch_trades_loop()))
        self._ws_tasks.append(asyncio.create_task(self._watch_positions_loop()))
        logger.info("Private WebSocket watcher loops started.")

    async def _watch_orders_loop(self) -> None:
        """Watch orders loop."""
        while self._running and self._exchange:
            try:
                orders = await self._exchange.watch_orders()
                for o in orders:
                    info = o.get("info", {})
                    cid = str(o.get("clientOrderId") or info.get("clientOrderId") or info.get("c") or "")
                    pos_side_raw = (info.get("positionSide") or info.get("ps") or o.get("positionSide") or "").upper()
                    if pos_side_raw not in ("LONG", "SHORT"):
                        if "short" in cid.lower() or "q_sell" in cid.lower():
                            pos_side_raw = "SHORT"
                        elif "long" in cid.lower() or "q_buy" in cid.lower():
                            pos_side_raw = "LONG"
                        else:
                            pos_idx = info.get("positionIdx", 0)
                            pos_side_raw = "LONG" if pos_idx == 1 else ("SHORT" if pos_idx == 2 else ("SHORT" if o.get("side") == "sell" else "LONG"))
                    pos_side = PositionSide(pos_side_raw) if pos_side_raw in ("LONG", "SHORT") else PositionSide.LONG

                    status_raw = o.get("status", "open").upper()
                    status = OrderStatus.FILLED if status_raw == "CLOSED" and float(o.get("remaining", 1)) == 0 else (
                        OrderStatus.CANCELED if status_raw == "CANCELED" else (
                            OrderStatus.PARTIALLY_FILLED if float(o.get("filled", 0)) > 0 else OrderStatus.NEW
                        )
                    )
                    record = OrderRecord(
                        id=str(o["id"]),
                        client_order_id=str(o.get("clientOrderId") or o["id"]),
                        exchange_order_id=str(o["id"]),
                        symbol=o["symbol"],
                        side=OrderSide(o["side"].upper()),
                        position_side=pos_side,
                        order_type=OrderType(o["type"].upper()) if o["type"].upper() in OrderType.__members__ else OrderType.LIMIT,
                        price=float(o.get("price") or 0.0),
                        stop_price=float(o.get("stopPrice") or 0.0) if o.get("stopPrice") else None,
                        amount=float(o["amount"]),
                        filled_amount=float(o.get("filled") or 0.0),
                        remaining_amount=float(o.get("remaining") or 0.0),
                        status=status,
                        purpose=OrderPurpose.ENTRY_QUOTE,
                        created_at=float(o.get("timestamp") or time.time() * 1000) / 1000.0,
                        updated_at=time.time(),
                        raw_response=o,
                    )
                    if self.on_order_update:
                        self.on_order_update(record)
            except Exception as e:
                if self._running:
                    logger.warning(f"watch_orders WS exception: {e}")
                    await asyncio.sleep(2)

    async def _watch_trades_loop(self) -> None:
        """Watch user fills loop (Source of Truth for trade executions)."""
        while self._running and self._exchange:
            try:
                trades = await self._exchange.watch_my_trades()
                for t in trades:
                    info = t.get("info", {})
                    cid = str(info.get("clientOrderId") or info.get("c") or t.get("clientOrderId") or "")
                    pos_side_raw = (info.get("positionSide") or info.get("ps") or t.get("positionSide") or "").upper()
                    if pos_side_raw not in ("LONG", "SHORT"):
                        if "short" in cid.lower() or "q_sell" in cid.lower():
                            pos_side_raw = "SHORT"
                        elif "long" in cid.lower() or "q_buy" in cid.lower():
                            pos_side_raw = "LONG"
                        else:
                            pos_idx = info.get("positionIdx", 0)
                            pos_side_raw = "LONG" if pos_idx == 1 else ("SHORT" if pos_idx == 2 else ("SHORT" if t.get("side") == "sell" else "LONG"))
                    pos_side = PositionSide(pos_side_raw) if pos_side_raw in ("LONG", "SHORT") else PositionSide.LONG

                    raw_rp = (
                        info.get("realizedPnl") or
                        info.get("rp") or
                        info.get("realizedProfit") or
                        info.get("realized_pnl") or
                        t.get("realizedPnl") or
                        t.get("realizedProfit") or
                        0.0
                    )
                    try:
                        realized_pnl = float(raw_rp)
                    except (ValueError, TypeError):
                        realized_pnl = 0.0

                    fill = FillRecord(
                        id=str(t["id"]),
                        order_id=str(t.get("order") or info.get("orderId") or ""),
                        client_order_id=info.get("clientOrderId") or info.get("c"),
                        symbol=t["symbol"],
                        side=OrderSide(t["side"].upper()),
                        position_side=pos_side,
                        price=float(t["price"]),
                        amount=float(t["amount"]),
                        quote_amount=float(t.get("cost") or (float(t["price"]) * float(t["amount"]))),
                        fee=float(t.get("fee", {}).get("cost") or 0.0) if isinstance(t.get("fee"), dict) else 0.0,
                        fee_currency=t.get("fee", {}).get("currency", "USDT") if isinstance(t.get("fee"), dict) else "USDT",
                        is_maker=t.get("takerOrMaker") == "maker",
                        timestamp=float(t.get("timestamp") or time.time() * 1000) / 1000.0,
                        realized_pnl=realized_pnl,
                    )
                    if self.on_fill_update:
                        self.on_fill_update(fill)
            except Exception as e:
                if self._running:
                    logger.warning(f"watch_my_trades WS exception: {e}")
                    await asyncio.sleep(2)

    async def _watch_positions_loop(self) -> None:
        """Watch position updates loop."""
        while self._running and self._exchange:
            try:
                if self._exchange.has.get("watchPositions"):
                    positions = await self._exchange.watch_positions()
                    for p in positions:
                        symbol = p.get("symbol")
                        if not symbol:
                            continue
                        info = p.get("info", {})
                        side_raw = (p.get("side") or info.get("positionSide") or info.get("ps") or "").upper()
                        if side_raw not in ("LONG", "SHORT"):
                            pos_idx = info.get("positionIdx", 0)
                            side_raw = "LONG" if pos_idx == 1 else ("SHORT" if pos_idx == 2 else ("SHORT" if float(p.get("contracts") or info.get("positionAmt") or 0) < 0 else "LONG"))
                        pos_side = PositionSide.LONG if side_raw == "LONG" else PositionSide.SHORT
                        amt = abs(float(p.get("contracts") or info.get("positionAmt") or 0.0))
                        entry_p = float(p.get("entryPrice") or 0.0)
                        upnl = float(p.get("unrealizedPnl") or 0.0)

                        if self.on_position_update:
                            self.on_position_update(symbol, pos_side, amt, entry_p, upnl)
                else:
                    await asyncio.sleep(60)
            except Exception as e:
                if self._running:
                    logger.warning(f"watch_positions WS exception: {e}")
                    await asyncio.sleep(5)

    async def close(self) -> None:
        """Gracefully close exchange connections and cancel WS tasks."""
        self._running = False
        for task in self._ws_tasks:
            task.cancel()
        if self._exchange:
            try:
                await self._exchange.close()
            except Exception as e:
                logger.warning(f"Error closing exchange: {e}")
            self._exchange = None
        self._is_connected = False
        logger.info("ExchangeGateway closed.")
