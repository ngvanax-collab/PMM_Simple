"""Direct High-Speed Screener & Top 5 Activator via Binance Futures REST API."""
import asyncio
import os
import sys
import time
from typing import Any, Dict, List
import httpx
from loguru import logger

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.screener import (
    MarketMetric,
    calculate_hurst_exponent,
    calculate_natr,
    compute_pmm_composite_score,
)
from app.models.config import PairConfig, TrailingStopConfig
from app.persistence.store import config_store

BLACKLIST = {
    "USDCUSDT", "FDUSDUSDT", "BUSDUSDT", "TUSDUSDT", "EURUSDT",
    "USDUSDT", "DAIUSDT", "USTCUSDT", "AEURUSDT", "USDPUSDT"
}


async def scan_and_activate():
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 80, flush=True)
    print("  🚀 OPENPMM-ENGINE v3: QUANTITATIVE SCREENER & TOP 5 ACTIVATION", flush=True)
    print("=" * 80, flush=True)
    print("Exchange: Binance Futures (USDT-Margined Perpetuals)", flush=True)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", flush=True)
    print("-" * 80, flush=True)

    async with httpx.AsyncClient(timeout=10.0, limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)) as client:
        # Step 1: Fetch Exchange Info (for contractType & underlyingType) and 24hr Tickers
        print("[1/4] Fetching Binance Futures exchangeInfo & 24hr tickers...", flush=True)
        r_info = await client.get("https://fapi.binance.com/fapi/v1/exchangeInfo")
        info_map = {s["symbol"]: s for s in r_info.json().get("symbols", [])}

        r_tick = await client.get("https://fapi.binance.com/fapi/v1/ticker/24hr")
        all_tickers = r_tick.json()
        print(f"       -> Received {len(all_tickers)} market tickers.", flush=True)

        # Step 2: Fetch Funding Rates (premiumIndex)
        print("[2/4] Fetching live funding rates...", flush=True)
        r_fr = await client.get("https://fapi.binance.com/fapi/v1/premiumIndex")
        funding_data = {item["symbol"]: float(item.get("lastFundingRate") or 0.0) for item in r_fr.json()}

        # Filter candidates: Quote USDT, Vol >= 20M, Pure Crypto (COIN Perpetual), Not blacklisted
        candidates = []
        for t in all_tickers:
            sym_raw = t["symbol"]
            meta = info_map.get(sym_raw, {})

            # Pure Crypto Filter: ignore TradFi, Equity, Commodity, Index
            if meta.get("underlyingType") != "COIN" or meta.get("contractType") != "PERPETUAL":
                continue
            if not sym_raw.endswith("USDT") or "_" in sym_raw or sym_raw in BLACKLIST:
                continue

            vol_24h = float(t.get("quoteVolume") or 0.0)
            if vol_24h < 20_000_000.0:
                continue

            price = float(t.get("lastPrice") or 0.0)
            best_bid = float(t.get("bidPrice") or 0.0)
            best_ask = float(t.get("askPrice") or 0.0)

            spread_bps = 2.0
            if best_bid > 0 and best_ask > best_bid:
                mid = (best_bid + best_ask) / 2.0
                spread_bps = ((best_ask - best_bid) / mid) * 10000.0

            depth_1pct = max(50_000.0, min(500_000.0, vol_24h * 0.001))
            fr = funding_data.get(sym_raw, 0.0)

            base_asset = sym_raw[:-4]
            unified_sym = f"{base_asset}/USDT:USDT"

            candidates.append({
                "raw_symbol": sym_raw,
                "symbol": unified_sym,
                "price": price,
                "vol_24h": vol_24h,
                "spread_bps": spread_bps,
                "depth_1pct": depth_1pct,
                "funding_rate": fr,
            })

        print(f"       -> {len(candidates)} pure crypto pairs qualify for deep statistical evaluation (Vol >= $20M).", flush=True)

        # Step 3: Fetch 100 1h Klines for candidates concurrently with Semaphore
        print("\n[3/4] Calculating Hurst Exponent, NATR 14, and PMM Composite Scores...", flush=True)
        sem = asyncio.Semaphore(15)
        evaluated_metrics: List[MarketMetric] = []

        async def eval_candidate(cand: Dict[str, Any]):
            async with sem:
                try:
                    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={cand['raw_symbol']}&interval=1h&limit=100"
                    res = await client.get(url)
                    klines = res.json()
                    if not isinstance(klines, list) or len(klines) < 20:
                        return

                    # Format for calculate_natr: [timestamp, open, high, low, close, volume]
                    candles = [[float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in klines]
                    close_prices = [c[4] for c in candles]

                    hurst = calculate_hurst_exponent(close_prices)
                    natr = calculate_natr(candles, period=14)

                    score, s_mr, s_natr, s_liq, s_fr = compute_pmm_composite_score(
                        hurst=hurst,
                        natr_pct=natr,
                        volume_24h=cand["vol_24h"],
                        depth_1pct=cand["depth_1pct"],
                        spread_bps=cand["spread_bps"],
                        funding_rate=cand["funding_rate"]
                    )

                    metric = MarketMetric(
                        symbol=cand["symbol"],
                        price=cand["price"],
                        volume_24h=cand["vol_24h"],
                        hurst=hurst,
                        natr_14=natr,
                        spread_bps=cand["spread_bps"],
                        depth_1pct=cand["depth_1pct"],
                        funding_rate=cand["funding_rate"],
                        score_mean_revert=s_mr,
                        score_natr=s_natr,
                        score_liquidity=s_liq,
                        score_funding=s_fr,
                        pmm_score=score,
                        status="CANDIDATE"
                    )
                    evaluated_metrics.append(metric)
                except Exception as e:
                    logger.debug(f"Error evaluating {cand['symbol']}: {e}")

        tasks = [eval_candidate(c) for c in candidates]
        await asyncio.gather(*tasks)

        # Sort descending by PMM Composite Score
        evaluated_metrics.sort(key=lambda m: m.pmm_score, reverse=True)
        for idx, m in enumerate(evaluated_metrics, 1):
            m.rank = idx

        # Step 4: Display Leaderboard
        print("\n" + "=" * 110, flush=True)
        print(f"{'RANK':<5} | {'SYMBOL':<18} | {'PMM SCORE':<10} | {'HURST (H)':<10} | {'NATR 14 (%)':<12} | {'24H VOL (USDT)':<16} | {'FUNDING RATE':<12}", flush=True)
        print("=" * 110, flush=True)
        for m in evaluated_metrics[:15]:
            fr_str = f"{m.funding_rate * 100:+.4f}%"
            vol_str = f"${m.volume_24h:,.0f}"
            print(f"#{m.rank:<4} | {m.symbol:<18} | {m.pmm_score:>9.2f} | {m.hurst:>9.4f} | {m.natr_14:>10.2f}% | {vol_str:>16} | {fr_str:>12}", flush=True)
        print("=" * 110, flush=True)

        # Step 5: Save Top 5 Configurations with Exact Preset Template
        top_5 = evaluated_metrics[:5]
        print(f"\n[4/4] Activating Top 5 Pure Crypto Pairs with allocated_margin_usdt=80.0:", flush=True)

        preset_template = {
            "leverage": 5,
            "margin_mode": "isolated",
            "order_amount_usdt": 33.0,
            "bid_spread": 0.003,
            "ask_spread": 0.003,
            "order_levels": 3,
            "order_level_spread": 0.005,
            "order_level_amount": 25.0,
            "max_long_usdt": 100.0,
            "max_short_usdt": 100.0,
            "gross_exposure_cap_usdt": 300.0,
            "allocated_margin_usdt": 80.0,
            "take_profit": 0.008,
            "stop_loss": 0.02,
            "time_limit": 21600,
            "base_cooldown_sec": 900,
            "cooldown_multiplier": 2.0,
            "max_cooldown_sec": 86400,
            "worker_max_loss_usdt": 20.0,
            "worker_max_drawdown_usdt": 10.0,
            "enabled": True,
        }


        # Clear old pairs in config/pairs/ directory to ensure exactly Top 5 are active
        for fname in os.listdir("config/pairs"):
            if fname.endswith(".json"):
                try:
                    os.remove(os.path.join("config/pairs", fname))
                except Exception:
                    pass

        for m in top_5:
            cfg = PairConfig(
                symbol=m.symbol,
                exchange="binance",
                trailing_stop=TrailingStopConfig(activation_price=0.012, trailing_delta=0.004),
                stop_loss_order_type="MARKET",
                **preset_template
            )
            saved_path = config_store.save_pair_config(cfg)
            print(f"       -> [ACTIVATED] #{m.rank} {m.symbol:<18} (Score: {m.pmm_score:.2f}, H: {m.hurst:.3f}, NATR: {m.natr_14:.2f}%) -> {saved_path}", flush=True)

        print("\n" + "=" * 80, flush=True)
        print("  ✅ TOP 5 PAIRS FULLY SYNCHRONIZED AND READY FOR EXECUTION!", flush=True)
        print("=" * 80, flush=True)


if __name__ == "__main__":
    asyncio.run(scan_and_activate())
