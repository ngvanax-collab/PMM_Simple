"""Multi-Exchange CCXT.pro Gateway for Funding Rate Arbitrage (Dual-Exchange Hedge Mode)."""
import asyncio
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import ccxt.pro as ccxtpro
import ccxt
from loguru import logger

from app.core.ratelimit import rate_limiter
from app.models.config import ExchangeCredentials


class MultiExchangeGateway:
    """Unified Gateway managing simultaneous Binance Futures & Bybit Linear connections in Hedge Mode."""

    def __init__(self, binance_creds: Optional[ExchangeCredentials] = None, bybit_creds: Optional[ExchangeCredentials] = None):
        self.creds = {
            "binance": binance_creds,
            "bybit": bybit_creds,
        }
        self.exchanges: Dict[str, ccxtpro.Exchange] = {}
        self.market_info: Dict[str, Dict[str, Any]] = {"binance": {}, "bybit": {}}
        self._connected: Dict[str, bool] = {"binance": False, "bybit": False}
        self._free_balance_cache: Dict[str, Tuple[float, float]] = {}  # exchange -> (timestamp, free_usdt)
        self._ws_tasks: List[asyncio.Task] = []
        self._running = False

        # Callbacks for private updates
        self.on_position_update: Optional[Callable[[str, str, str, float, float, float], Any]] = None
        self.on_order_update: Optional[Callable[[Dict[str, Any]], Any]] = None

    async def initialize(self) -> bool:
        """Initialize both exchanges, load markets, and verify Hedge Mode."""
        all_ok = True
        for ex_name, creds in self.creds.items():
            if not creds or not creds.api_key or not creds.api_secret:
                logger.warning(f"[{ex_name.upper()}] No credentials provided for Funding Rate Arbitrage gateway.")
                continue

            try:
                exchange_class = getattr(ccxtpro, ex_name, None)
                if exchange_class is None:
                    raise ValueError(f"Exchange {ex_name} not supported by ccxt.pro")

                config = {
                    "apiKey": creds.api_key,
                    "secret": creds.api_secret,
                    "enableRateLimit": False,
                    "options": {
                        "defaultType": "future",
                        "adjustForTimeDifference": True,
                    }
                }
                if creds.passphrase:
                    config["password"] = creds.passphrase

                ex_instance = exchange_class(config)
                if creds.testnet:
                    ex_instance.set_sandbox_mode(True)

                logger.info(f"Connecting to {ex_name.upper()} Futures (testnet={creds.testnet})...")
                self.market_info[ex_name] = await ex_instance.load_markets()
                self.exchanges[ex_name] = ex_instance

                # Verify & enforce Hedge Mode
                hedge_ok = await self.verify_and_set_hedge_mode(ex_name)
                if not hedge_ok:
                    logger.error(f"[{ex_name.upper()}] Failed to verify/switch to Hedge Mode.")
                    all_ok = False
                    continue

                self._connected[ex_name] = True
                logger.info(f"[{ex_name.upper()}] Gateway initialized successfully with Hedge Mode active.")
            except Exception as e:
                logger.error(f"[{ex_name.upper()}] Gateway initialization error: {e}")
                self._connected[ex_name] = False
                all_ok = False

        self._running = True
        return any(self._connected.values())

    def is_connected(self, exchange: str) -> bool:
        """Check if specific exchange is initialized and connected."""
        return self._connected.get(exchange.lower(), False) and exchange.lower() in self.exchanges

    async def verify_and_set_hedge_mode(self, exchange: str) -> bool:
        """Verify that account on given exchange is in Dual-side (Hedge) Mode or switch to it."""
        ex_name = exchange.lower()
        ex = self.exchanges.get(ex_name)
        if not ex:
            return False

        try:
            await rate_limiter.acquire_weight(weight=2)
            if ex_name == "binance":
                try:
                    res = await ex.fapiPrivateGetPositionSideDual()
                    is_dual = res.get("dualSidePosition", False)
                    if isinstance(is_dual, str):
                        is_dual = (is_dual.lower() == "true")

                    if is_dual:
                        logger.info("[BINANCE] Verified Dual-Side (Hedge) Mode is already ACTIVE.")
                        return True

                    logger.warning("[BINANCE] Account is in One-Way mode. Switching to Hedge Mode...")
                    await rate_limiter.acquire_weight(weight=2)
                    switch_res = await ex.fapiPrivatePostPositionSideDual({"dualSidePosition": "true"})
                    logger.info(f"[BINANCE] Switched to Hedge Mode: {switch_res}")
                    return True
                except Exception as e:
                    logger.error(f"[BINANCE] Hedge mode check/switch error: {e}")
                    return False

            elif ex_name == "bybit":
                try:
                    res = await ex.set_position_mode(True)
                    logger.info(f"[BYBIT] Position mode set to Hedge (Dual): {res}")
                    return True
                except Exception as e:
                    err_msg = str(e).lower()
                    if any(s in err_msg for s in ("not modified", "110025", "already in", "no need to change")):
                        logger.info("[BYBIT] Verified Hedge Mode is already ACTIVE.")
                        return True
                    logger.error(f"[BYBIT] Hedge mode switch error: {e}")
                    return False

            return True
        except Exception as e:
            logger.error(f"[{ex_name.upper()}] Exception in verify_and_set_hedge_mode: {e}")
            return False

    def normalize_symbol(self, exchange: str, symbol: str) -> str:
        """Normalize symbol string for exchange compatibility (e.g. BTC/USDT:USDT or BTCUSDT)."""
        ex_name = exchange.lower()
        sym = symbol.strip().upper()
        
        # If symbol is already formatted with '/', check market info
        markets = self.market_info.get(ex_name, {})
        if sym in markets:
            return sym

        # Standard conversion: e.g. BTCUSDT -> BTC/USDT:USDT
        if "/" not in sym:
            if sym.endswith("USDT"):
                base = sym[:-4]
                candidate = f"{base}/USDT:USDT"
                if candidate in markets:
                    return candidate
                candidate_simple = f"{base}/USDT"
                if candidate_simple in markets:
                    return candidate_simple

        return sym

    def get_market_precision(self, exchange: str, symbol: str) -> Tuple[int, int, float, float]:
        """Return (price_precision, amount_precision, tick_size, min_amount) for symbol on exchange."""
        ex_name = exchange.lower()
        norm_sym = self.normalize_symbol(ex_name, symbol)
        market = self.market_info.get(ex_name, {}).get(norm_sym)

        if not market:
            return (2, 3, 0.01, 0.001)

        precision = market.get("precision", {})
        price_prec = precision.get("price", 2)
        amount_prec = precision.get("amount", 3)

        if isinstance(price_prec, float):
            price_prec = int(round(-math.log10(price_prec))) if price_prec > 0 else 2
        if isinstance(amount_prec, float):
            amount_prec = int(round(-math.log10(amount_prec))) if amount_prec > 0 else 3

        limits = market.get("limits", {})
        tick_size = limits.get("price", {}).get("min") or (10 ** (-price_prec))
        min_amount = limits.get("amount", {}).get("min") or (10 ** (-amount_prec))

        return (int(price_prec), int(amount_prec), float(tick_size), float(min_amount))

    def price_to_precision(self, exchange: str, symbol: str, price: float) -> float:
        """Quantize price according to exchange precision limits."""
        ex_name = exchange.lower()
        price_prec, _, tick_size, _ = self.get_market_precision(ex_name, symbol)
        if tick_size > 0:
            rounded = round(price / tick_size) * tick_size
            return round(rounded, price_prec)
        return round(price, price_prec)

    def amount_to_precision(self, exchange: str, symbol: str, amount: float) -> float:
        """Quantize amount according to exchange precision limits (floor to avoid balance overflow)."""
        ex_name = exchange.lower()
        _, amount_prec, _, min_amount = self.get_market_precision(ex_name, symbol)
        factor = 10 ** amount_prec
        floored = math.floor(amount * factor) / factor
        return round(floored, amount_prec)

    async def get_free_margin(self, exchange: str) -> float:
        """Get cached or live free USDT balance."""
        ex_name = exchange.lower()
        ex = self.exchanges.get(ex_name)
        if not ex:
            return 0.0

        now = time.time()
        cached = self._free_balance_cache.get(ex_name)
        if cached and (now - cached[0]) < 2.0:
            return cached[1]

        try:
            await rate_limiter.acquire_weight(weight=2)
            balance = await ex.fetch_balance()
            free_usdt = float(balance.get("USDT", {}).get("free", 0.0) or balance.get("free", {}).get("USDT", 0.0))
            self._free_balance_cache[ex_name] = (now, free_usdt)
            return free_usdt
        except Exception as e:
            logger.warning(f"[{ex_name.upper()}] Failed to fetch balance: {e}")
            return cached[1] if cached else 0.0

    async def fetch_ticker_price(self, exchange: str, symbol: str) -> float:
        """Fetch current mid/mark price for symbol."""
        ex_name = exchange.lower()
        ex = self.exchanges.get(ex_name)
        if not ex:
            return 0.0

        norm_sym = self.normalize_symbol(ex_name, symbol)
        try:
            await rate_limiter.acquire_weight(weight=1)
            ticker = await ex.fetch_ticker(norm_sym)
            return float(ticker.get("last") or ticker.get("close") or ticker.get("mark") or 0.0)
        except Exception as e:
            logger.warning(f"[{ex_name.upper()}] Error fetching ticker for {symbol}: {e}")
            return 0.0

    # ── Strict Hedge Mode Order Operations ──

    async def create_hedge_order(
        self,
        exchange: str,
        symbol: str,
        side: str,  # "buy" or "sell"
        position_side: str,  # "LONG" or "SHORT"
        amount: float,
        price: Optional[float] = None,
        order_type: str = "market",
        client_order_id: Optional[str] = None,
        is_post_only: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Create an order in Hedge Mode.
        CRITICAL: Never sets reduceOnly or closePosition.
        Uses positionSide='LONG'|'SHORT' for Binance and positionIdx=1|2 for Bybit.
        """
        ex_name = exchange.lower()
        ex = self.exchanges.get(ex_name)
        if not ex:
            logger.error(f"[{ex_name.upper()}] Exchange instance not available")
            return None

        norm_sym = self.normalize_symbol(ex_name, symbol)
        pos_side_upper = position_side.upper()
        assert pos_side_upper in ("LONG", "SHORT"), f"Invalid position side {position_side}"

        params: Dict[str, Any] = {
            "positionSide": pos_side_upper,
        }
        if client_order_id:
            params["newClientOrderId"] = client_order_id
            params["clientOrderId"] = client_order_id

        if ex_name == "binance":
            if is_post_only and order_type.lower() == "limit":
                params["timeInForce"] = "GTX"
        elif ex_name == "bybit":
            params["positionIdx"] = 1 if pos_side_upper == "LONG" else 2
            if is_post_only and order_type.lower() == "limit":
                params["postOnly"] = True

        fmt_amount = self.amount_to_precision(ex_name, symbol, amount)
        fmt_price = self.price_to_precision(ex_name, symbol, price) if price is not None else None

        price_prec, _, tick_size, min_amount = self.get_market_precision(ex_name, symbol)
        if fmt_amount < min_amount:
            logger.error(f"[{ex_name.upper()}] Amount {fmt_amount} below minimum {min_amount} for {symbol}")
            return None

        max_retries = 2 if is_post_only else 0
        curr_price = fmt_price

        for attempt in range(max_retries + 1):
            await rate_limiter.acquire_order(count=1, weight=1)
            try:
                if client_order_id and attempt > 0:
                    attempt_id = f"{client_order_id}_{attempt}"
                    params["newClientOrderId"] = attempt_id
                    params["clientOrderId"] = attempt_id

                order = await ex.create_order(
                    symbol=norm_sym,
                    type=order_type.lower(),
                    side=side.lower(),
                    amount=fmt_amount,
                    price=curr_price if order_type.lower() == "limit" else None,
                    params=params,
                )
                return order
            except Exception as e:
                err_msg = str(e)
                is_post_only_err = any(
                    x in err_msg.lower() for x in ("-5022", "not be executed as maker", "would immediately match", "post only")
                )

                if is_post_only_err and attempt < max_retries and tick_size > 0 and curr_price is not None:
                    # Step back 1 tick
                    if side.lower() == "buy":
                        curr_price = self.price_to_precision(ex_name, symbol, curr_price - tick_size)
                    else:
                        curr_price = self.price_to_precision(ex_name, symbol, curr_price + tick_size)

                    logger.warning(
                        f"[{ex_name.upper()}][{symbol}] Post-Only maker rejected (-5022). Stepping 1 tick to {curr_price} (retry {attempt+1}/{max_retries})..."
                    )
                    continue
                else:
                    logger.error(f"[{ex_name.upper()}][{symbol}] Failed order {side.upper()} {pos_side_upper}: {err_msg}")
                    return None

        return None

    async def emergency_market_close(
        self,
        exchange: str,
        symbol: str,
        position_side: str,
        amount: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Emergency close an open position using MARKET order in Hedge Mode.
        - To close LONG: Send SELL with positionSide='LONG'
        - To close SHORT: Send BUY with positionSide='SHORT'
        CRITICAL: Never sets reduceOnly or closePosition.
        """
        pos_side = position_side.upper()
        close_side = "sell" if pos_side == "LONG" else "buy"
        logger.warning(f"[{exchange.upper()}][{symbol}] EMERGENCY MARKET CLOSE: {close_side.upper()} {pos_side} size={amount}")
        return await self.create_hedge_order(
            exchange=exchange,
            symbol=symbol,
            side=close_side,
            position_side=pos_side,
            amount=amount,
            order_type="market",
            is_post_only=False,
        )

    async def cancel_all_orders(self, exchange: str, symbol: Optional[str] = None) -> bool:
        """Cancel all open orders on exchange."""
        ex_name = exchange.lower()
        ex = self.exchanges.get(ex_name)
        if not ex:
            return False

        try:
            await rate_limiter.acquire_order(count=1, weight=2)
            norm_sym = self.normalize_symbol(ex_name, symbol) if symbol else None
            await ex.cancel_all_orders(symbol=norm_sym)
            logger.info(f"[{ex_name.upper()}] Cancelled all open orders (symbol={symbol})")
            return True
        except Exception as e:
            logger.error(f"[{ex_name.upper()}] Error cancelling all orders: {e}")
            return False

    async def fetch_positions(self, exchange: str, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Fetch open positions for exchange."""
        ex_name = exchange.lower()
        ex = self.exchanges.get(ex_name)
        if not ex:
            return []

        try:
            await rate_limiter.acquire_weight(weight=2)
            norm_symbols = [self.normalize_symbol(ex_name, s) for s in symbols] if symbols else None
            raw_positions = await ex.fetch_positions(symbols=norm_symbols)
            return raw_positions or []
        except Exception as e:
            logger.warning(f"[{ex_name.upper()}] Error fetching positions: {e}")
            return []

    async def close(self) -> None:
        """Gracefully close all exchange connections."""
        self._running = False
        for task in self._ws_tasks:
            task.cancel()
        for ex_name, ex in self.exchanges.items():
            try:
                await ex.close()
                logger.info(f"[{ex_name.upper()}] Gateway closed.")
            except Exception as e:
                logger.warning(f"[{ex_name.upper()}] Error closing gateway: {e}")
        self.exchanges.clear()
        self._connected = {"binance": False, "bybit": False}
