"""Rank, evaluate, and save exactly the Top 5 Pairs into config/pairs."""
import asyncio
import os
import sys
import httpx

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


async def main():
    print("=" * 80)
    print("  🚀 OPENPMM-ENGINE v3: FULL-MARKET QUANTITATIVE LEADERBOARD")
    print("=" * 80)
    async with httpx.AsyncClient(timeout=10.0) as client:
        r_tick = await client.get("https://fapi.binance.com/fapi/v1/ticker/24hr")
        all_tickers = r_tick.json()

        r_fr = await client.get("https://fapi.binance.com/fapi/v1/premiumIndex")
        funding_data = {item["symbol"]: float(item.get("lastFundingRate") or 0.0) for item in r_fr.json()}

        candidates = []
        for t in all_tickers:
            sym_raw = t["symbol"]
            if not sym_raw.endswith("USDT") or "_" in sym_raw or sym_raw in BLACKLIST:
                continue

            vol_24h = float(t.get("quoteVolume") or 0.0)
            if vol_24h < 20_000_000.0:
                continue

            price = float(t.get("lastPrice") or 0.0)
            best_bid = float(t.get("bidPrice") or 0.0)
            best_ask = float(t.get("askPrice") or 0.0)
            spread_bps = ((best_ask - best_bid) / ((best_bid + best_ask) / 2.0)) * 10000.0 if best_bid > 0 else 2.0
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

        sem = asyncio.Semaphore(15)
        metrics = []

        async def eval_cand(cand):
            async with sem:
                try:
                    res = await client.get(f"https://fapi.binance.com/fapi/v1/klines?symbol={cand['raw_symbol']}&interval=1h&limit=100")
                    klines = res.json()
                    if not isinstance(klines, list) or len(klines) < 20:
                        return
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
                    metrics.append(MarketMetric(
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
                    ))
                except Exception:
                    pass

        await asyncio.gather(*[eval_cand(c) for c in candidates])
        metrics.sort(key=lambda m: m.pmm_score, reverse=True)
        for idx, m in enumerate(metrics, 1):
            m.rank = idx

        print("\n" + "=" * 115)
        print(f"{'RANK':<5} | {'SYMBOL':<18} | {'PMM SCORE':<10} | {'HURST (H)':<10} | {'NATR 14 (%)':<12} | {'24H VOL (USDT)':<16} | {'FUNDING RATE':<12}")
        print("=" * 115)
        for m in metrics[:10]:
            fr_str = f"{m.funding_rate * 100:+.4f}%"
            vol_str = f"${m.volume_24h:,.0f}"
            print(f"#{m.rank:<4} | {m.symbol:<18} | {m.pmm_score:>9.2f} | {m.hurst:>9.4f} | {m.natr_14:>10.2f}% | {vol_str:>16} | {fr_str:>12}")
        print("=" * 115)

        # Clear existing configs and save exactly Top 5
        top_5 = metrics[:5]
        for f in os.listdir("config/pairs"):
            if f.endswith(".json"):
                os.remove(os.path.join("config/pairs", f))

        preset = {
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
            "allocated_margin_usdt": 50.0,
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

        print("\n[ACTIVE WORKERS CONFIGURED]")
        for m in top_5:
            cfg = PairConfig(
                symbol=m.symbol,
                exchange="binance",
                trailing_stop=TrailingStopConfig(activation_price=0.012, trailing_delta=0.004),
                stop_loss_order_type="MARKET",
                **preset
            )
            p = config_store.save_pair_config(cfg)
            print(f" -> Slot #{m.rank}: {m.symbol:<18} (PMM Score: {m.pmm_score:.2f}) -> {p}")

if __name__ == "__main__":
    asyncio.run(main())
