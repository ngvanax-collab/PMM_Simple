"""Automated Full-Market Screener & Top 5 Pair Activation Script."""
import asyncio
import os
import sys
import time
from typing import List
import ccxt.async_support as ccxt_async
from loguru import logger

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.screener import MarketMetric, QuantitativeScreener, ScreenerConfig
from app.models.config import PairConfig, TrailingStopConfig
from app.persistence.store import config_store


class PublicGatewayAdapter:
    """Lightweight Gateway adapter providing public market data fetching for Screener."""
    def __init__(self, exchange_name: str = "binance"):
        self.exchange_name = exchange_name
        exchange_class = getattr(ccxt_async, exchange_name)
        self._exchange = exchange_class({
            "enableRateLimit": False,
            "options": {"defaultType": "future", "adjustForTimeDifference": True}
        })
        self._market_info = {}

    async def initialize(self):
        self._market_info = await self._exchange.load_markets()

    async def fetch_ticker_and_mark(self, symbol: str):
        try:
            ticker = await self._exchange.fetch_ticker(symbol)
            bid = float(ticker.get("bid") or 0.0)
            ask = float(ticker.get("ask") or 0.0)
            last = float(ticker.get("last") or ticker.get("close") or 0.0)
            mark = float(ticker.get("info", {}).get("markPrice") or last or (bid + ask) / 2.0)
            return {"bid": bid, "ask": ask, "mark": mark, "last": last}
        except Exception:
            return None

    async def close(self):
        await self._exchange.close()




async def main():
    print("=" * 80)
    print("  🚀 OPENPMM-ENGINE v3: FULL-MARKET SCREENER & TOP 5 PAIR ACTIVATION")
    print("=" * 80)
    print(f"Target Exchange: Binance Futures (USDT-Margined Linear Perpetuals)")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("-" * 80)

    # 1. Initialize Public Gateway Adapter
    gateway = PublicGatewayAdapter("binance")
    print("[1/4] Loading Binance Futures markets...")
    await gateway.initialize()
    print(f"       -> Loaded {len(gateway._market_info)} total market pairs.")

    # 2. Run Quantitative Screener
    screener = QuantitativeScreener(ScreenerConfig(
        min_volume_24h_usdt=20_000_000.0,
        max_concurrency=10,
        ohlcv_timeframe="1h",
        ohlcv_limit=100
    ))

    print("\n[2/4] Scanning & calculating quantitative metrics (Hurst, NATR 14, Depth, Funding)...")
    start_t = time.time()
    candidates: List[MarketMetric] = await screener.scan_and_rank_all_pairs(gateway)
    elapsed = time.time() - start_t
    print(f"       -> Evaluated {len(candidates)} qualifying candidate pairs in {elapsed:.2f}s.")

    if not candidates:
        print("❌ Error: No candidates found. Please verify network connectivity.")
        await gateway.close()
        return

    # 3. Print Screener Leaderboard
    print("\n" + "=" * 105)
    print(f"{'RANK':<5} | {'SYMBOL':<18} | {'PMM SCORE':<10} | {'HURST (H)':<10} | {'NATR 14 (%)':<12} | {'24H VOL (USDT)':<16} | {'FUNDING RATE':<12}")
    print("=" * 105)
    for m in candidates[:15]:
        fr_str = f"{m.funding_rate * 100:+.4f}%"
        vol_str = f"${m.volume_24h:,.0f}"
        print(f"#{m.rank:<4} | {m.symbol:<18} | {m.pmm_score:>9.2f} | {m.hurst:>9.4f} | {m.natr_14:>10.2f}% | {vol_str:>16} | {fr_str:>12}")
    print("=" * 105)

    # 4. Select Top 5 Pairs & Generate Preset Configurations
    top_5 = candidates[:5]
    print(f"\n[3/4] Generating exact preset configuration template for Top 5 Pairs:")
    for idx, cand in enumerate(top_5, 1):
        print(f"       {idx}. {cand.symbol} (PMM Score: {cand.pmm_score:.2f}, Hurst: {cand.hurst:.4f}, NATR: {cand.natr_14:.2f}%)")

    # Exact Balanced Preset Template Parameters
    preset_template = {
        "leverage": 5,
        "margin_mode": "isolated",
        "order_amount_usdt": 30.0,
        "bid_spread": 0.0035,
        "ask_spread": 0.0035,
        "minimum_spread": 0.0025,
        "order_levels": 3,
        "order_level_spread": 0.0025,
        "order_level_amount": 12.5,
        "order_refresh_time": 45,
        "requote_threshold_pct": 0.001,
        "min_holding_sec": 3.0,
        "inventory_skew_enabled": True,
        "allocated_margin_usdt": 50.0,
        "max_long_usdt": 150.0,
        "max_short_usdt": 150.0,
        "gross_exposure_cap_usdt": 250.0,
        "skew_kappa": 1.0,
        "skew_gamma_net": 0.001,
        "tp_skew_boost": 0.5,
        "take_profit": 0.008,
        "take_profit_order_type": "LIMIT_MAKER",
        "trailing_tp_enabled": True,
        "trailing_tp_activation_pct": 0.008,
        "trailing_tp_callback_pct": 0.003,
        "stop_loss": 0.018,
        "stop_loss_order_type": "MARKET",
        "time_limit": 14400,
        "passive_exit_timeout_sec": 120.0,
        "base_cooldown_sec": 600,
        "cooldown_multiplier": 2.0,
        "max_cooldown_sec": 43200,
        "worker_max_loss_usdt": 20.0,
        "worker_max_drawdown_usdt": 15.0,
        "vol_pause_pct": 0.025,
        "vol_lookback_sec": 60,
        "enabled": True,
    }

    print("\n[4/4] Writing configuration files to config/pairs/ ...")
    for cand in top_5:
        pair_cfg = PairConfig(
            symbol=cand.symbol,
            exchange="binance",
            **preset_template
        )
        saved_path = config_store.save_pair_config(pair_cfg)
        print(f"       -> Saved: {cand.symbol} -> {saved_path}")

    await gateway.close()

    print("\n" + "=" * 80)
    print("  ✅ TOP 5 PAIRS ACTIVATED & CONFIGURATIONS SYNCHRONIZED SUCCESSFULLY!")
    print("=" * 80)
    print("To launch the live bot engine and dashboard:")
    print("   python -m app.main")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
