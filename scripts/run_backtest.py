#!/usr/bin/env python3
"""
CLI Runner for OpenPMM Historical Backtest Engine (Futures Hedge Mode Native).
Usage:
    python3 scripts/run_backtest.py --symbol SOLUSDT --days 7 --interval 1m --spread 0.0035 --tp 0.006 --sl 0.012
"""
import argparse
import sys
import time
from pathlib import Path

# Add project root and local .venv site-packages to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

for sp in project_root.glob(".venv/lib/python*/site-packages"):
    if str(sp) not in sys.path:
        sys.path.insert(0, str(sp))

from loguru import logger
from backtest.data_loader import fetch_binance_futures_klines
from backtest.pmm_simulator import PMMBacktestEngine
from backtest.metrics import format_metrics_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Historical PMM Backtest Engine (Binance Futures Hedge Mode Native)"
    )
    parser.add_argument("--symbol", type=str, default="SOLUSDT", help="Symbol (e.g. SOLUSDT, BTCUSDT)")
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days (default: 7)")
    parser.add_argument("--interval", type=str, default="1m", help="Candle timeframe (default: 1m)")
    parser.add_argument("--spread", type=float, default=0.0035, help="Half spread pct (default: 0.0035 = 0.35%%)")
    parser.add_argument("--tp", type=float, default=0.006, help="Take profit pct (default: 0.006 = 0.6%%)")
    parser.add_argument("--sl", type=float, default=0.012, help="Stop loss pct (default: 0.012 = 1.2%%)")
    parser.add_argument("--trailing-cb", type=float, default=0.003, help="Trailing pullback callback pct (default: 0.003 = 0.3%%)")
    parser.add_argument("--trailing-act", type=float, default=None, help="Trailing activation pct (default: matches TP)")
    parser.add_argument("--order-amount", type=float, default=30.0, help="Order amount in USDT (default: 30.0)")
    parser.add_argument("--balance", type=float, default=1000.0, help="Initial balance in USDT (default: 1000.0)")
    parser.add_argument("--maker-fee", type=float, default=0.0002, help="Maker fee (default: 0.0002 = 0.02%%)")
    parser.add_argument("--taker-fee", type=float, default=0.0005, help="Taker fee (default: 0.0005 = 0.05%%)")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage penalty (default: 0.0005 = 0.05%%)")
    parser.add_argument("--tick-size", type=float, default=0.01, help="Exchange price tick size (default: 0.01)")
    parser.add_argument("--force-download", action="store_true", help="Force re-download data from Binance")
    # Dynamic Digital Twin Parameters
    parser.add_argument("--k-spread", type=float, default=0.35, help="VAMM spread multiplier vs NATR (default: 0.35)")
    parser.add_argument("--k-tp", type=float, default=0.65, help="VAMM take profit multiplier vs NATR (default: 0.65)")
    parser.add_argument("--k-sl-mult", type=float, default=1.8, help="VAMM stop loss multiplier vs NATR (default: 1.8)")
    parser.add_argument("--cb-thresh", type=float, default=0.010, help="60s Circuit Breaker surge threshold (default: 0.010 = 1.0%%)")
    parser.add_argument("--vol-spike-thresh", type=float, default=3.0, help="Toxic volume spike multiplier (default: 3.0x)")
    parser.add_argument("--time-decay-mins", type=int, default=30, help="Time decay threshold in minutes (default: 30)")
    parser.add_argument("--max-holding-mins", type=int, default=244, help="Max holding duration in minutes (default: 244)")
    parser.add_argument("--no-vamm", action="store_true", help="Disable dynamic VAMM spread scaling")
    parser.add_argument("--no-cb", action="store_true", help="Disable 60s volatility circuit breaker")
    parser.add_argument("--no-trend-bias", action="store_true", help="Disable 1h EMA50 trend bias filter")
    parser.add_argument("--no-toxic-exit", action="store_true", help="Disable fast toxic volume exit")
    parser.add_argument("--no-time-decay", action="store_true", help="Disable time-decay SL tightening")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}", level="INFO")

    print("\n" + "=" * 68)
    print("  🚀 OPENPMM HISTORICAL BACKTEST ENGINE (DIGITAL TWIN V3)")
    print(f"  Target: {args.symbol} | Lookback: {args.days} days | Timeframe: {args.interval}")
    print(f"  VAMM: {'OFF' if args.no_vamm else 'ON'} (k_spread={args.k_spread}, k_tp={args.k_tp}, k_sl={args.k_sl_mult})")
    print(f"  Circuit Breaker: {'OFF' if args.no_cb else 'ON'} (thresh={args.cb_thresh*100:.2f}%) | Trend Bias 1h: {'OFF' if args.no_trend_bias else 'ON'}")
    print(f"  Toxic Exit: {'OFF' if args.no_toxic_exit else 'ON'} | Time-Decay SL: {'OFF' if args.no_time_decay else 'ON'} ({args.time_decay_mins}m)")
    print(f"  Order Size: ${args.order_amount:.1f} USDT | Init Balance: ${args.balance:.1f} USDT")
    print("=" * 68 + "\n")

    # 1. Fetch / Load historical data
    t0 = time.time()
    try:
        df = fetch_binance_futures_klines(
            symbol=args.symbol,
            interval=args.interval,
            days=args.days,
            force_download=args.force_download if args.force_download else None,
        )
    except Exception as e:
        logger.critical(f"Failed to fetch data: {e}")
        sys.exit(1)

    load_time = time.time() - t0
    logger.info(f"Loaded {len(df):,} candles in {load_time:.2f}s ({df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]})")

    # 2. Initialize and run backtest engine
    t1 = time.time()
    engine = PMMBacktestEngine(
        symbol=args.symbol,
        initial_balance=args.balance,
        order_amount_usdt=args.order_amount,
        base_spread_pct=args.spread,
        take_profit_pct=args.tp,
        stop_loss_pct=args.sl,
        trailing_callback_pct=args.trailing_cb,
        trailing_activation_pct=args.trailing_act,
        maker_fee=args.maker_fee,
        taker_fee=args.taker_fee,
        slippage_pct=args.slippage,
        tick_size=args.tick_size,
        k_spread=args.k_spread,
        k_tp=args.k_tp,
        k_sl_mult=args.k_sl_mult,
        cb_threshold_pct=args.cb_thresh,
        vol_spike_threshold=args.vol_spike_thresh,
        time_decay_mins=args.time_decay_mins,
        max_holding_mins=args.max_holding_mins,
        use_vamm=not args.no_vamm,
        use_circuit_breaker=not args.no_cb,
        use_trend_bias=not args.no_trend_bias,
        use_toxic_exit=not args.no_toxic_exit,
        use_time_decay_sl=not args.no_time_decay,
    )

    metrics = engine.run(df)
    sim_time = time.time() - t1
    logger.info(f"Simulation completed in {sim_time:.2f}s ({len(df) / max(0.001, sim_time):,.0f} candles/sec)")

    # 3. Print quantitative report
    print("\n" + format_metrics_report(metrics, symbol=args.symbol) + "\n")


if __name__ == "__main__":
    main()
