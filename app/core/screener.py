"""Quantitative Market Screener & PMM Scoring Engine (Futures Hedge Mode)."""
import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from app.models.config import ScreenerConfig


@dataclass
class MarketMetric:
    """Quantitative evaluation record for a single trading pair."""
    symbol: str
    price: float = 0.0
    volume_24h: float = 0.0
    price_change_24h: float = 0.0
    hurst: float = 0.5
    natr_14: float = 0.0
    volatility_ratio: float = 1.0
    spread_bps: float = 0.0
    depth_1pct: float = 0.0
    funding_rate: float = 0.0
    score_mean_revert: float = 0.0
    score_natr: float = 0.0
    score_liquidity: float = 0.0
    score_funding: float = 0.0
    pmm_score: float = 0.0
    rank: int = 0
    status: str = "CANDIDATE"  # "ACTIVE", "DRAINING", "CANDIDATE", "FILTERED"
    rejection_reason: Optional[str] = None
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "volume_24h": self.volume_24h,
            "price_change_24h": round(self.price_change_24h, 2),
            "hurst": round(self.hurst, 4),
            "natr_14": round(self.natr_14, 4),
            "volatility_ratio": round(self.volatility_ratio, 2),
            "spread_bps": round(self.spread_bps, 2),
            "depth_1pct": round(self.depth_1pct, 2),
            "funding_rate": round(self.funding_rate, 6),
            "score_mean_revert": round(self.score_mean_revert, 2),
            "score_natr": round(self.score_natr, 2),
            "score_liquidity": round(self.score_liquidity, 2),
            "score_funding": round(self.score_funding, 2),
            "pmm_score": round(self.pmm_score, 2),
            "rank": self.rank,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "last_updated": self.last_updated,
        }


# ─────────────────────────────────────────────────────────────
# Quantitative Statistical Mathematical Functions
# ─────────────────────────────────────────────────────────────

def calculate_hurst_exponent(prices: List[float]) -> float:
    """
    Calculate Hurst Exponent (H) via multi-lag log price dispersion / variance scaling.
    - H < 0.5: Mean-Reverting (Variance grows sub-linearly with lag)
    - H = 0.5: Random Walk (Brownian Motion)
    - H > 0.5: Trending / Momentum (Variance grows super-linearly with lag)

    Returns H clamped to [0.0, 1.0].
    """
    if len(prices) < 20:
        return 0.5

    log_prices = []
    for p in prices:
        if p > 0:
            log_prices.append(math.log(p))

    N = len(log_prices)
    if N < 20:
        return 0.5

    lags = [2, 4, 8, 16, 32]
    lags = [lag for lag in lags if lag < N // 2]
    if len(lags) < 2:
        return 0.5

    log_lags = []
    log_stds = []

    for lag in lags:
        diffs = [log_prices[i + lag] - log_prices[i] for i in range(N - lag)]
        if not diffs:
            continue
        mean_diff = sum(diffs) / len(diffs)
        var = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
        std = math.sqrt(var) if var > 1e-12 else 1e-6

        log_lags.append(math.log(lag))
        log_stds.append(math.log(std))

    if len(log_lags) < 2:
        return 0.5

    # Linear regression: log_std = H * log_lag + C
    n_pts = len(log_lags)
    sum_x = sum(log_lags)
    sum_y = sum(log_stds)
    sum_xx = sum(x * x for x in log_lags)
    sum_xy = sum(x * y for x, y in zip(log_lags, log_stds))

    denom = (n_pts * sum_xx - sum_x * sum_x)
    if abs(denom) < 1e-12:
        return 0.5

    slope = (n_pts * sum_xy - sum_x * sum_y) / denom
    return float(max(0.0, min(1.0, slope)))


def calculate_natr(candles: List[List[float]], period: int = 14) -> float:
    """
    Calculate Normalized Average True Range (NATR_14) in percent.
    NATR = (ATR_14 / Close_last) * 100.0 (%)
    Candle format: [timestamp, open, high, low, close, volume]
    """
    if len(candles) < period + 1:
        return 0.0

    tr_list = []
    for i in range(1, len(candles)):
        high = float(candles[i][2])
        low = float(candles[i][3])
        prev_close = float(candles[i - 1][4])

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_list.append(tr)

    if len(tr_list) < period:
        return 0.0

    # Simple RMA / SMA over the latest period
    recent_tr = tr_list[-period:]
    atr = sum(recent_tr) / len(recent_tr)
    last_close = float(candles[-1][4])

    if last_close <= 0:
        return 0.0

    natr = (atr / last_close) * 100.0
    return float(max(0.0, natr))


def calculate_volatility_ratio(close_prices: List[float], short_window: int = 24, long_window: int = 100) -> float:
    """
    Calculate short-term vs long-term volatility ratio (sigma_short / sigma_long).
    If short-term volatility spikes relative to baseline, ratio > 1.30 indicating pump/dump phase.
    """
    if len(close_prices) < 20:
        return 1.0

    # Calculate log returns
    returns = [
        math.log(close_prices[i] / close_prices[i - 1])
        for i in range(1, len(close_prices))
        if close_prices[i - 1] > 0 and close_prices[i] > 0
    ]
    if len(returns) < 10:
        return 1.0

    n_short = min(len(returns), max(5, short_window))
    n_long = min(len(returns), max(n_short + 5, long_window))

    short_returns = returns[-n_short:]
    long_returns = returns[-n_long:]

    mean_short = sum(short_returns) / len(short_returns)
    var_short = sum((r - mean_short) ** 2 for r in short_returns) / max(1, len(short_returns) - 1)
    std_short = math.sqrt(var_short)

    mean_long = sum(long_returns) / len(long_returns)
    var_long = sum((r - mean_long) ** 2 for r in long_returns) / max(1, len(long_returns) - 1)
    std_long = math.sqrt(var_long)

    if std_long <= 1e-8:
        return 1.0

    return float(std_short / std_long)


# ─────────────────────────────────────────────────────────────
# PMM Sub-scoring & Composite Scoring Formulas
# ─────────────────────────────────────────────────────────────

def score_mean_reversion(hurst: float) -> float:
    """
    S_MeanRevert in [0, 100].
    - Max 100 pts for H <= 0.35 (strong mean-reverting regime)
    - Smooth decay from 100 down to 0 as H approaches 0.46
    - 0 pts for H >= 0.46 (trending/momentum regime)
    """
    h = max(0.0, min(1.0, float(hurst)))
    if h <= 0.35:
        return 100.0
    elif 0.35 < h < 0.46:
        return 100.0 * (1.0 - ((h - 0.35) / (0.46 - 0.35)))
    else:
        return 0.0


def score_natr_volatility(natr_pct: float) -> float:
    """
    S_NATR in [0, 100] with steep penalty towards 1.8% ceiling.
    - Ideal zone: 1.0% <= NATR <= 1.5% -> 100.0 pts
    - 0.8% <= NATR < 1.0%: scale up from 60.0 to 100.0 pts
    - 1.5% < NATR <= 1.8%: steep penalty from 100.0 down to 0.0 pts
    - NATR > 1.8%: 0.0 pts (ceiling breached)
    - NATR < 0.8%: linear decay towards 0.0 pts (too calm)
    """
    natr = max(0.0, float(natr_pct))
    if 1.0 <= natr <= 1.5:
        return 100.0
    elif 0.8 <= natr < 1.0:
        return 60.0 + ((natr - 0.8) / 0.2) * 40.0
    elif 1.5 < natr <= 1.8:
        return 100.0 * ((1.8 - natr) / 0.3)
    elif natr > 1.8:
        return 0.0
    else:  # natr < 0.8
        return max(0.0, (natr / 0.8) * 60.0)


def score_liquidity(volume_24h: float, depth_1pct: float, spread_bps: float) -> float:
    """
    S_Liquidity in [0, 100].
    - Volume 24h (min 20M up to 200M+) -> 50% weight
    - Depth +-1% (target >= 50k up to 200k+) -> 50% weight
    """
    vol = max(0.0, float(volume_24h))
    depth = max(0.0, float(depth_1pct))

    # Volume Score: 20M -> 50 pts, 200M+ -> 100 pts
    if vol < 20_000_000.0:
        vol_score = (vol / 20_000_000.0) * 50.0
    else:
        vol_score = min(100.0, 50.0 + ((vol - 20_000_000.0) / 180_000_000.0) * 50.0)

    # Depth Score: 50k -> 70 pts, 200k+ -> 100 pts
    if depth < 50_000.0:
        depth_score = (depth / 50_000.0) * 70.0
    else:
        depth_score = min(100.0, 70.0 + ((depth - 50_000.0) / 150_000.0) * 30.0)

    return float(max(0.0, min(100.0, 0.5 * vol_score + 0.5 * depth_score)))


def score_funding_rate(funding_rate: float) -> float:
    """
    S_Funding in [0, 100].
    - |FR| <= 0.01% (0.0001) / 8h -> 100.0
    - Decays towards 0 at |FR| >= 0.10% (0.0010)
    """
    abs_fr = abs(float(funding_rate))
    if abs_fr <= 0.0001:
        return 100.0
    else:
        decay = max(0.0, 1.0 - ((abs_fr - 0.0001) / 0.0009))
        return float(100.0 * decay)


def compute_pmm_composite_score(
    hurst: float,
    natr_pct: float,
    volume_24h: float,
    depth_1pct: float,
    spread_bps: float,
    funding_rate: float
) -> Tuple[float, float, float, float, float]:
    """
    Compute PMM Composite Score:
    PMM_Score = 0.35 * S_MeanRevert + 0.25 * S_NATR + 0.20 * S_Liquidity + 0.20 * S_Funding
    Returns (pmm_score, s_mr, s_natr, s_liq, s_fr) all in [0, 100].
    """
    s_mr = score_mean_reversion(hurst)
    s_natr = score_natr_volatility(natr_pct)
    s_liq = score_liquidity(volume_24h, depth_1pct, spread_bps)
    s_fr = score_funding_rate(funding_rate)

    composite = 0.35 * s_mr + 0.25 * s_natr + 0.20 * s_liq + 0.20 * s_fr
    composite = max(0.0, min(100.0, composite))
    return float(composite), float(s_mr), float(s_natr), float(s_liq), float(s_fr)


def compute_vamm_parameters(
    natr_pct: float,
    allocated_margin: float = 21.0,
    leverage: int = 5,
    order_levels: int = 3,
    maker_fee: float = 0.0002,
    taker_fee: float = 0.0005,
) -> Dict[str, Any]:
    """
    Volatility-Anchored Market Making (VAMM) Dynamic Parameter Calculation.
    Derives grid spreads, take profit, trailing callback, stop loss barrier, and flat order sizing from 1h NATR_14.
    """
    natr_dec = float(natr_pct) / 100.0 if float(natr_pct) > 0.05 else float(natr_pct)
    natr_dec = max(0.0001, natr_dec)

    # 1. Spread Floor: 2 * maker_fee + taker_fee + 0.0010 buffer (clamped at min 0.0025)
    s_floor = max(0.0025, (2.0 * maker_fee) + taker_fee + 0.0010)

    # 2. Base Spread (Level 0)
    base_spread = max(s_floor, 0.30 * natr_dec)

    # 3. Level Spread (Step size)
    order_level_spread = 0.45 * natr_dec

    # 4. Take Profit & Trailing TP
    take_profit = max(0.005, 0.60 * natr_dec)
    trailing_tp_activation_pct = take_profit
    trailing_tp_callback_pct = round(0.35 * take_profit, 6)

    # 5. Barrier Stop Loss (Outside Grid + Volatility Buffer)
    order_lvls = max(1, int(order_levels))
    s_max = base_spread + (order_lvls - 1) * order_level_spread
    stop_loss = s_max + 0.60 * natr_dec

    # 6. Standard Flat Capital Allocation (21 USDT Margin @ 5x -> 105 Notional Power, Flat 30 USDT per level)
    total_power = allocated_margin * leverage
    base_order_amount = 30.0  # Khóa chuẩn 30 USDT/level cho 21 USDT margin 5x
    order_level_amount = 0.0
    max_side_cap = 96.0  # Bao trọn 3 tầng 30x3 = 90 USDT + đệm trượt giá
    gross_cap = round(total_power, 1)  # 105.0 USDT

    params = {
        "allocated_margin_usdt": allocated_margin,
        "bid_spread": round(base_spread, 6),
        "ask_spread": round(base_spread, 6),
        "minimum_spread": round(s_floor, 6),
        "order_level_spread": round(order_level_spread, 6),
        "order_levels": order_lvls,
        "take_profit": round(take_profit, 6),
        "trailing_tp_enabled": True,
        "trailing_tp_activation_pct": round(trailing_tp_activation_pct, 6),
        "trailing_tp_callback_pct": round(trailing_tp_callback_pct, 6),
        "stop_loss": round(stop_loss, 6),
        "order_amount_usdt": base_order_amount,
        "order_level_amount": order_level_amount,
        "max_long_usdt": max_side_cap,
        "max_short_usdt": max_side_cap,
        "gross_exposure_cap_usdt": gross_cap,
        "level_cooldown_sec": 1800,
    }
    logger.info(
        f"[VAMM_DYNAMIC] Computed parameters for NATR={natr_pct:.3f}%: base_spread={params['bid_spread']*100:.3f}%, "
        f"level_spread={params['order_level_spread']*100:.3f}%, TP={params['take_profit']*100:.3f}%, "
        f"SL={params['stop_loss']*100:.3f}%, order_amt=${params['order_amount_usdt']:.2f} USDT, "
        f"margin=${params['allocated_margin_usdt']:.1f} USDT, gross_cap=${params['gross_exposure_cap_usdt']:.1f} USDT"
    )
    return params


# ─────────────────────────────────────────────────────────────
# Quantitative Screener Engine Class
# ─────────────────────────────────────────────────────────────

class QuantitativeScreener:
    """
    Automated Quantitative Screener scanning USDT perpetual contracts on exchange,
    applying strict Volatility Ceilings, 24h Pump-Dump Filters, Hurst Trending Rejections,
    and calculating PMM Composite Scores to rank safe, high-yield liquidity pairs.
    """

    def __init__(self, config: Optional[ScreenerConfig] = None):
        self.config = config or ScreenerConfig()
        self.last_metrics: Dict[str, MarketMetric] = {}
        self.last_scan_time: float = 0.0
        self._is_scanning: bool = False
        self._lock = asyncio.Lock()

    @property
    def is_scanning(self) -> bool:
        return self._is_scanning

    async def scan_and_rank_all_pairs(self, gateway: Any) -> List[MarketMetric]:
        """
        Execute full market scan across exchange USDT futures pairs:
        1. Fetch all markets & filter by volume and pure crypto perpetual contract type.
        2. Filter out blacklisted tokens and dated delivery contracts.
        3. Pre-filter by 24h volume and 24h price change pump/dump limit.
        4. Concurrently fetch 1h candles, evaluate NATR, Hurst, and Volatility Ratio.
        5. Apply strict Volatility Ceiling & Hurst Filters.
        6. Compute PMM Composite Score for qualified candidates, sort descending by rank.
        """
        async with self._lock:
            self._is_scanning = True
            logger.info("Starting Quantitative Market Screener scan across all USDT Perpetual Futures...")
            start_t = time.time()

            try:
                # 1. Load markets from gateway
                if not gateway or not gateway._exchange:
                    logger.warning("QuantitativeScreener: Gateway exchange not available.")
                    return []

                markets = gateway._market_info or await gateway._exchange.load_markets()

                # 2. Hard Filters
                candidate_symbols: List[str] = []
                for symbol, market in markets.items():
                    if not market.get("active", True):
                        continue
                    if market.get("quote") != "USDT":
                        continue
                    if not market.get("linear", True) and not market.get("swap", False) and not market.get("future", False):
                        continue
                    if symbol in self.config.blacklist:
                        continue

                    # Filter out non-perpetual dated delivery contracts (e.g. BTC/USDT:USDT-240927)
                    if ":" in symbol and "-" in symbol.split(":")[-1]:
                        continue

                    # Strict Pure Crypto Filter: discard TradFi, Equity, Commodity, Index contracts
                    market_info = market.get("info") or {}
                    contract_type = str(market_info.get("contractType") or "").upper()
                    underlying_type = str(market_info.get("underlyingType") or "").upper()

                    if "TRADIFI" in contract_type:
                        continue
                    if underlying_type and underlying_type in {"EQUITY", "COMMODITY", "INDEX"}:
                        continue
                    if underlying_type and underlying_type != "COIN":
                        continue

                    candidate_symbols.append(symbol)

                logger.info(f"Screener: Found {len(candidate_symbols)} pure crypto perpetual pairs after preliminary filter.")

                # 2.5 Batch fetch 24h volume & funding rate pre-filtering
                all_tickers = {}
                if hasattr(gateway._exchange, "fetch_tickers"):
                    try:
                        all_tickers = await gateway._exchange.fetch_tickers()
                    except Exception as e:
                        logger.debug(f"fetch_tickers batch error: {e}")

                all_funding = {}
                if hasattr(gateway._exchange, "fetch_funding_rates"):
                    try:
                        all_funding = await gateway._exchange.fetch_funding_rates()
                    except Exception as e:
                        logger.debug(f"fetch_funding_rates batch error: {e}")

                def get_cached_ticker(s: str) -> dict:
                    if not all_tickers:
                        return {}
                    if s in all_tickers:
                        return all_tickers[s]
                    base_s = s.split(":")[0]
                    if base_s in all_tickers:
                        return all_tickers[base_s]
                    market_info = markets.get(s, {})
                    m_id = market_info.get("id")
                    if m_id and m_id in all_tickers:
                        return all_tickers[m_id]
                    raw_id = s.replace("/", "").replace(":USDT", "").replace(":", "")
                    if raw_id in all_tickers:
                        return all_tickers[raw_id]
                    return {}

                def get_cached_funding(s: str) -> float:
                    if not all_funding:
                        return 0.0
                    f_info = all_funding.get(s) or all_funding.get(s.split(":")[0]) or {}
                    return float(f_info.get("fundingRate") or 0.0)

                vol_filtered_candidates: List[str] = []
                for sym in candidate_symbols:
                    tick = get_cached_ticker(sym)
                    if tick:
                        vol_24h = float(tick.get("quoteVolume") or 0.0)
                        if vol_24h <= 0 and "baseVolume" in tick and "last" in tick:
                            vol_24h = float(tick.get("baseVolume") or 0.0) * float(tick.get("last") or 0.0)

                        if vol_24h > 0 and vol_24h < self.config.min_volume_24h_usdt:
                            continue

                        # 24h Price Change Filter (Pre-check)
                        pct_change = float(tick.get("percentage") or 0.0)
                        if abs(pct_change) > self.config.max_24h_price_change_pct:
                            logger.debug(
                                f"[SCREENER_REJECT] {sym}: 24h change {pct_change:+.2f}% exceeds ±{self.config.max_24h_price_change_pct:.1f}%"
                            )
                            continue

                    vol_filtered_candidates.append(sym)

                logger.info(
                    f"Screener: {len(vol_filtered_candidates)} pairs qualify for deep statistical evaluation (Vol >= ${self.config.min_volume_24h_usdt:,.0f})."
                )

                # 3. Batch concurrent evaluation with Semaphore
                concurrency = getattr(self.config, "max_concurrency", 10)
                semaphore = asyncio.Semaphore(concurrency)
                results: List[MarketMetric] = []

                async def evaluate_single_symbol(sym: str) -> Optional[MarketMetric]:
                    async with semaphore:
                        try:
                            # 1. Price & Volume from cached ticker or fallback
                            cached_tick = get_cached_ticker(sym)
                            price = float(cached_tick.get("last") or cached_tick.get("close") or 0.0)
                            vol_24h = float(cached_tick.get("quoteVolume") or 0.0)
                            best_bid = float(cached_tick.get("bid") or 0.0)
                            best_ask = float(cached_tick.get("ask") or 0.0)
                            pct_change_24h = float(cached_tick.get("percentage") or 0.0)

                            if price <= 0:
                                ticker = await gateway.fetch_ticker_and_mark(sym)
                                if ticker:
                                    price = ticker.get("mark") or ticker.get("bid") or 0.0
                                    best_bid = ticker.get("bid") or 0.0
                                    best_ask = ticker.get("ask") or 0.0
                                    pct_change_24h = float(ticker.get("percentage") or 0.0)

                            if price <= 0:
                                return None

                            if vol_24h <= 0 and "baseVolume" in cached_tick:
                                vol_24h = float(cached_tick.get("baseVolume") or 0.0) * price

                            # ── Hard Filter 1: 24h Volume ──
                            if vol_24h < self.config.min_volume_24h_usdt:
                                return None

                            # ── Hard Filter 2: 24h Price Change (Pump-and-Dump Filter) ──
                            if abs(pct_change_24h) > self.config.max_24h_price_change_pct:
                                logger.debug(
                                    f"[SCREENER_REJECT] {sym}: 24h change {pct_change_24h:+.2f}% exceeds ±{self.config.max_24h_price_change_pct:.1f}%"
                                )
                                return None

                            # ── Hard Filter 3: Top of Book Natural Spread ──
                            spread_bps = 2.0
                            if best_bid > 0 and best_ask > best_bid:
                                mid = (best_bid + best_ask) / 2.0
                                spread_bps = ((best_ask - best_bid) / mid) * 10000.0

                            if spread_bps > self.config.max_top_of_book_spread_bps:
                                logger.debug(
                                    f"[SCREENER_REJECT] {sym}: Spread={spread_bps:.2f} bps exceeds ceiling {self.config.max_top_of_book_spread_bps:.1f} bps"
                                )
                                return None

                            # 2. Funding Rate from batch or fallback
                            fr = get_cached_funding(sym)
                            if fr == 0.0 and hasattr(gateway._exchange, "fetch_funding_rate"):
                                try:
                                    fr_res = await gateway._exchange.fetch_funding_rate(sym)
                                    fr = float(fr_res.get("fundingRate") or 0.0)
                                except Exception:
                                    pass

                            # 3. Fetch Candles for Statistical Analysis (Hurst, NATR, Volatility Ratio)
                            candles: List[List[float]] = []
                            if hasattr(gateway._exchange, "fetch_ohlcv"):
                                try:
                                    candles = await gateway._exchange.fetch_ohlcv(
                                        sym,
                                        timeframe=self.config.ohlcv_timeframe,
                                        limit=self.config.ohlcv_limit
                                    )
                                except Exception as e:
                                    logger.debug(f"[{sym}] OHLCV fetch error: {e}")

                            if not candles or len(candles) < 20:
                                return None

                            close_prices = [float(c[4]) for c in candles]

                            # ── Hard Filter 4: NATR 14 Volatility Ceiling & Floor ──
                            natr = calculate_natr(candles, period=14)
                            if natr > self.config.max_natr_pct:
                                logger.debug(
                                    f"[SCREENER_REJECT] {sym}: NATR={natr:.2f}% exceeds ceiling {self.config.max_natr_pct:.2f}%"
                                )
                                return None
                            if natr < self.config.min_natr_pct:
                                logger.debug(
                                    f"[SCREENER_REJECT] {sym}: NATR={natr:.2f}% below floor {self.config.min_natr_pct:.2f}%"
                                )
                                return None

                            # ── Hard Filter 5: Hurst Exponent (Trend Rejection) ──
                            hurst = calculate_hurst_exponent(close_prices)
                            if hurst >= self.config.max_hurst_exponent:
                                logger.debug(
                                    f"[SCREENER_REJECT] {sym}: Hurst={hurst:.2f} is trending (>= {self.config.max_hurst_exponent:.2f})"
                                )
                                return None

                            # ── Hard Filter 6: Volatility Ratio (Pump-and-Dump Spike) ──
                            vol_ratio = calculate_volatility_ratio(close_prices)
                            if vol_ratio > self.config.max_volatility_ratio:
                                logger.debug(
                                    f"[SCREENER_REJECT] {sym}: Volatility ratio={vol_ratio:.2f} exceeds {self.config.max_volatility_ratio:.2f}"
                                )
                                return None

                            # Depth calculation from real orderbook snapshot or fallback to volume estimation
                            depth_1pct = max(50_000.0, min(500_000.0, vol_24h * 0.001))
                            if hasattr(gateway._exchange, "fetch_order_book") and price > 0:
                                try:
                                    ob = await gateway._exchange.fetch_order_book(sym, limit=20)
                                    bids = ob.get("bids") or []
                                    asks = ob.get("asks") or []
                                    min_bid = price * 0.99
                                    max_ask = price * 1.01
                                    b_depth = sum(float(p) * float(q) for p, q in bids if float(p) >= min_bid)
                                    a_depth = sum(float(p) * float(q) for p, q in asks if float(p) <= max_ask)
                                    real_depth = b_depth + a_depth
                                    if real_depth > 0:
                                        depth_1pct = float(real_depth)
                                except Exception as e:
                                    logger.warning(
                                        f"[SCREENER_WARNING] Failed to fetch orderbook depth for {sym}: {e}. "
                                        f"Falling back to volume estimation."
                                    )

                            # Compute Composite Score for qualified candidates
                            score, s_mr, s_natr, s_liq, s_fr = compute_pmm_composite_score(
                                hurst=hurst,
                                natr_pct=natr,
                                volume_24h=vol_24h,
                                depth_1pct=depth_1pct,
                                spread_bps=spread_bps,
                                funding_rate=fr
                            )

                            return MarketMetric(
                                symbol=sym,
                                price=price,
                                volume_24h=vol_24h,
                                price_change_24h=pct_change_24h,
                                hurst=hurst,
                                natr_14=natr,
                                volatility_ratio=vol_ratio,
                                spread_bps=spread_bps,
                                depth_1pct=depth_1pct,
                                funding_rate=fr,
                                score_mean_revert=s_mr,
                                score_natr=s_natr,
                                score_liquidity=s_liq,
                                score_funding=s_fr,
                                pmm_score=score,
                                status="CANDIDATE"
                            )

                        except Exception as e:
                            logger.warning(f"[{sym}] Screener evaluation exception: {e}")
                            return None

                tasks = [evaluate_single_symbol(s) for s in vol_filtered_candidates]
                gathered = await asyncio.gather(*tasks, return_exceptions=True)

                for item in gathered:
                    if isinstance(item, Exception):
                        logger.error(f"[SCREENER_TASK_ERROR] {item}")
                    elif isinstance(item, MarketMetric) and item.status != "FILTERED":
                        results.append(item)

                # 4. Sort descending by PMM score and assign ranks
                results.sort(key=lambda m: m.pmm_score, reverse=True)
                for idx, metric in enumerate(results):
                    metric.rank = idx + 1

                self.last_metrics = {m.symbol: m for m in results}
                self.last_scan_time = time.time()
                elapsed = time.time() - start_t
                logger.info(
                    f"Quantitative Screener scan complete in {elapsed:.2f}s. "
                    f"Top 5 candidates: {[f'{m.symbol} ({m.pmm_score:.1f})' for m in results[:5]]}"
                )
                return results

            except Exception as e:
                logger.error(f"Quantitative Screener critical failure: {e}")
                return []
            finally:
                self._is_scanning = False


# Global singleton screener instance
screener_engine = QuantitativeScreener()
