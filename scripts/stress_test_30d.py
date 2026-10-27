#!/usr/bin/env python3
"""
30-Day Survival Stress Replay Benchmark for OpenPMM Digital Twin V3.
Evaluates long-term stability and tail-risk resilience across SOLUSDT, XRPUSDT, NEARUSDT, and AVAXUSDT.

Pass Criteria (Hard Gates):
1. Net Realized PnL > 0 over 30-day cycle.
2. Max Drawdown (MDD) <= 2.50%.
3. Max consecutive Stop Loss streak < 3 (never triggers >= 3 consecutive SLs).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Add project root and local .venv site-packages to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

for sp in project_root.glob(".venv/lib/python*/site-packages"):
    if str(sp) not in sys.path:
        sys.path.insert(0, str(sp))

from backtest.data_loader import fetch_binance_futures_klines, precompute_market_indicators
from backtest.pmm_simulator import PMMBacktestEngine

TARGET_TOKENS = {
    "SOLUSDT": {
        "unified": "SOL/USDT:USDT",
        "tick_size": 0.01,
        "qty_step": 0.01,
        "default_k_spread": 0.85,
        "default_k_tp": 1.0,
        "default_k_sl_mult": 1.8,
        "default_time_decay_mins": 45,
    },
    "XRPUSDT": {
        "unified": "XRP/USDT:USDT",
        "tick_size": 0.0001,
        "qty_step": 0.1,
        "default_k_spread": 0.75,
        "default_k_tp": 1.4,
        "default_k_sl_mult": 1.8,
        "default_time_decay_mins": 45,
    },
    "NEARUSDT": {
        "unified": "NEAR/USDT:USDT",
        "tick_size": 0.001,
        "qty_step": 1.0,
        "default_k_spread": 0.95,
        "default_k_tp": 0.85,
        "default_k_sl_mult": 1.5,
        "default_time_decay_mins": 45,
    },
    "AVAXUSDT": {
        "unified": "AVAX/USDT:USDT",
        "tick_size": 0.001,
        "qty_step": 1.0,
        "default_k_spread": 0.65,
        "default_k_tp": 1.2,
        "default_k_sl_mult": 1.8,
        "default_time_decay_mins": 30,
    },
    "ETHUSDT": {
        "unified": "ETH/USDT:USDT",
        "tick_size": 0.01,
        "qty_step": 0.001,
        "default_k_spread": 0.85,
        "default_k_tp": 1.0,
        "default_k_sl_mult": 1.8,
        "default_time_decay_mins": 45,
    },
    "ADAUSDT": {
        "unified": "ADA/USDT:USDT",
        "tick_size": 0.0001,
        "qty_step": 1.0,
        "default_k_spread": 0.85,
        "default_k_tp": 1.0,
        "default_k_sl_mult": 1.8,
        "default_time_decay_mins": 45,
    },
    "UNIUSDT": {
        "unified": "UNI/USDT:USDT",
        "tick_size": 0.001,
        "qty_step": 1.0,
        "default_k_spread": 0.85,
        "default_k_tp": 1.0,
        "default_k_sl_mult": 1.8,
        "default_time_decay_mins": 45,
    },
    "LINKUSDT": {
        "unified": "LINK/USDT:USDT",
        "tick_size": 0.001,
        "qty_step": 0.01,
        "default_k_spread": 0.85,
        "default_k_tp": 1.0,
        "default_k_sl_mult": 1.8,
        "default_time_decay_mins": 45,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="30-Day Survival Stress Replay Benchmark (Digital Twin V3)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/top5_live_presets.json",
        help="Path to optimized live presets JSON config",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="DEFAULT",
        help="Comma-separated symbols to benchmark (e.g. SOLUSDT,XRPUSDT,NEARUSDT) or ALL",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Stress replay lookback in days (default: 30 days ~ 43,200 candles)",
    )
    parser.add_argument(
        "--order-amount",
        type=float,
        default=35.0,
        help="Order amount in USDT (default: 35.0)",
    )
    parser.add_argument(
        "--max-holding-mins",
        type=int,
        default=244,
        help="Max holding duration in minutes (default: 244)",
    )
    parser.add_argument(
        "--max-mdd-gate",
        type=float,
        default=2.00,
        help="Max Drawdown hard gate percentage (default: 2.00%%)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download historical candle data from Binance",
    )
    return parser.parse_args()


def load_token_preset(config_path: str, unified_sym: str, default_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Load optimized parameters from preset JSON file with fallback to defaults."""
    cfg_file = Path(config_path)
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                presets = json.load(f)
            if unified_sym in presets:
                p = presets[unified_sym]
                return {
                    "k_spread": float(p.get("k_spread", default_meta["default_k_spread"])),
                    "k_tp": float(p.get("k_tp", default_meta["default_k_tp"])),
                    "k_sl_mult": float(p.get("k_sl_mult", default_meta["default_k_sl_mult"])),
                    "time_decay_mins": int(p.get("time_decay_mins", default_meta["default_time_decay_mins"])),
                    "order_amount_usdt": float(p.get("order_amount_usdt", 35.0)),
                    "max_holding_mins": int(p.get("max_holding_mins", 244)),
                }
        except Exception as e:
            logger.warning(f"Could not parse preset for {unified_sym} ({e}), using defaults.")

    return {
        "k_spread": default_meta["default_k_spread"],
        "k_tp": default_meta["default_k_tp"],
        "k_sl_mult": default_meta["default_k_sl_mult"],
        "time_decay_mins": default_meta["default_time_decay_mins"],
        "order_amount_usdt": 35.0,
        "max_holding_mins": 244,
    }


def calculate_sl_streaks(trades: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Calculate max consecutive Stop Loss streaks.
    Returns: (max_hard_sl_streak, max_any_sl_streak)
      - max_hard_sl_streak: Consecutive STOP_LOSS exits
      - max_any_sl_streak: Consecutive (STOP_LOSS or TIME_DECAY_SL) exits
    """
    max_hard_streak = 0
    curr_hard_streak = 0

    max_any_streak = 0
    curr_any_streak = 0

    for t in trades:
        reason = t.get("exit_reason", "")
        net_pnl = t.get("net_pnl", 0.0)

        # 1. Hard SL Streak
        if reason == "STOP_LOSS":
            curr_hard_streak += 1
            max_hard_streak = max(max_hard_streak, curr_hard_streak)
        else:
            curr_hard_streak = 0

        # 2. Any SL Streak (STOP_LOSS or TIME_DECAY_SL)
        if reason in ("STOP_LOSS", "TIME_DECAY_SL") or (net_pnl < 0 and "SL" in reason):
            curr_any_streak += 1
            max_any_streak = max(max_any_streak, curr_any_streak)
        elif net_pnl > 0 or "TAKE_PROFIT" in reason:
            curr_any_streak = 0

    return max_hard_streak, max_any_streak


def run_30d_stress_replay():
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}", level="INFO")

    print("\n" + "=" * 110)
    print("  🔥 30-DAY SURVIVAL STRESS REPLAY BENCHMARK (DIGITAL TWIN V3)")
    print(f"  Target Hardware: DELL SANDBOX (16GB RAM) | Historical Window: {args.days} Days (~43,200 1m Candles)")
    print(f"  Hard Pass Gates: (1) Net PnL > 0 | (2) Max Drawdown <= {args.max_mdd_gate:.2f}% | (3) Max Consecutive SL Streak < 3")
    print(f"  Presets File: {args.config}")
    print("=" * 110 + "\n")

    results: List[Dict[str, Any]] = []

    if args.symbols.upper() == "ALL":
        symbols_to_run = list(TARGET_TOKENS.keys())
    elif args.symbols.upper() == "DEFAULT":
        symbols_to_run = ["SOLUSDT", "XRPUSDT", "NEARUSDT", "AVAXUSDT"]
    else:
        raw_items = [s.strip() for s in args.symbols.split(",") if s.strip()]
        symbols_to_run = []
        for s in raw_items:
            clean = s.replace("/", "").replace(":USDT", "").upper()
            if clean not in TARGET_TOKENS:
                logger.critical(f"Symbol {s} is not supported. Supported: {list(TARGET_TOKENS.keys())}")
                sys.exit(1)
            symbols_to_run.append(clean)

    for sym in symbols_to_run:
        meta = TARGET_TOKENS[sym]
        preset = load_token_preset(args.config, meta["unified"], meta)

        logger.info(f"[{sym}] Fetching {args.days} days of 1m candles...")
        t0 = time.time()
        df = fetch_binance_futures_klines(
            symbol=sym,
            interval="1m",
            days=args.days,
            force_download=args.force_download if args.force_download else None,
        )
        fetch_time = time.time() - t0

        logger.info(f"[{sym}] Precomputing market indicators without lookahead bias ({len(df):,} candles)...")
        t1 = time.time()
        df = precompute_market_indicators(df)
        prep_time = time.time() - t1

        logger.info(
            f"[{sym}] Running 30-day simulation (k_spread={preset['k_spread']}, k_tp={preset['k_tp']}, "
            f"k_sl={preset['k_sl_mult']}, decay={preset['time_decay_mins']}m, max_hold={preset['max_holding_mins']}m)..."
        )
        engine = PMMBacktestEngine(
            symbol=sym,
            initial_balance=1000.0,
            order_amount_usdt=args.order_amount,
            max_holding_mins=preset["max_holding_mins"],
            tick_size=meta["tick_size"],
            qty_step=meta["qty_step"],
            k_spread=preset["k_spread"],
            k_tp=preset["k_tp"],
            k_sl_mult=preset["k_sl_mult"],
            time_decay_mins=preset["time_decay_mins"],
            cb_threshold_pct=0.010,
            vol_spike_threshold=3.0,
            use_vamm=True,
            use_circuit_breaker=True,
            use_trend_bias=True,
            use_toxic_exit=True,
            use_time_decay_sl=True,
        )

        t2 = time.time()
        metrics = engine.run(df)
        sim_time = time.time() - t2

        # Evaluate Stress Gates
        net_pnl = float(metrics["net_pnl"])
        mdd_pct = float(metrics["max_drawdown_pct"])
        max_hard_sl, max_any_sl = calculate_sl_streaks(engine.trades)

        pnl_gate = net_pnl > 0.0
        mdd_gate = mdd_pct <= args.max_mdd_gate
        streak_gate = max_hard_sl < 3

        passed_all = pnl_gate and mdd_gate and streak_gate

        results.append({
            "symbol": sym,
            "candles": len(df),
            "preset": preset,
            "metrics": metrics,
            "trades": engine.trades,
            "net_pnl": net_pnl,
            "gross_pnl": float(metrics["gross_pnl"]),
            "total_fees": float(metrics["total_fees"]),
            "fee_drag": float(metrics["fee_drag_pct"]),
            "win_rate": float(metrics["win_rate_pct"]),
            "total_trades": int(metrics["total_trades"]),
            "mdd_pct": mdd_pct,
            "sharpe": float(metrics["sharpe_ratio"]),
            "max_hard_sl": max_hard_sl,
            "max_any_sl": max_any_sl,
            "pnl_gate": pnl_gate,
            "mdd_gate": mdd_gate,
            "streak_gate": streak_gate,
            "passed_all": passed_all,
            "sim_time": sim_time,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # REPORTING & FORMATTING
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 128)
    print(f"  📊 30-DAY STRESS REPLAY PERFORMANCE MATRIX (LOOKBACK: {args.days} DAYS)")
    print("=" * 128)
    print(
        f"| {'Symbol':<10} | {'Trades':<6} | {'WinRate':<7} | {'Gross PnL':<10} | {'Total Fees':<10} | "
        f"{'Fee Drag':<8} | {'Net PnL':<10} | {'MDD (%)':<7} | {'Max SL Streak':<13} | {'Sharpe':<6} | {'Gate Status':<11} |"
    )
    print("|" + "-" * 12 + "|" + "-" * 8 + "|" + "-" * 9 + "|" + "-" * 12 + "|" + "-" * 12 + "|" +
          "-" * 10 + "|" + "-" * 12 + "|" + "-" * 9 + "|" + "-" * 15 + "|" + "-" * 8 + "|" + "-" * 13 + "|")

    total_net_pnl = 0.0
    total_gross_pnl = 0.0
    total_fees = 0.0
    total_trades = 0
    all_passed = True

    for r in results:
        status_str = "✅ PASSED" if r["passed_all"] else "❌ FAILED"
        if not r["passed_all"]:
            all_passed = False

        total_net_pnl += r["net_pnl"]
        total_gross_pnl += r["gross_pnl"]
        total_fees += r["total_fees"]
        total_trades += r["total_trades"]

        streak_display = f"{r['max_hard_sl']} Hard / {r['max_any_sl']} All"
        print(
            f"| {r['symbol']:<10} | {r['total_trades']:<6d} | {r['win_rate']:5.1f}% | "
            f"{r['gross_pnl']:+8.2f}$  | {r['total_fees']:8.2f}$  | {r['fee_drag']:6.1f}% | "
            f"{r['net_pnl']:+8.2f}$  | {r['mdd_pct']:5.2f}% | {streak_display:<13} | {r['sharpe']:6.2f} | {status_str:<11} |"
        )

    print("+" + "-" * 126 + "+")
    portfolio_fee_drag = (total_fees / abs(total_gross_pnl) * 100.0) if abs(total_gross_pnl) > 0 else 0.0
    print(
        f"  PORTFOLIO TOTAL: {total_trades} trades | Gross: {total_gross_pnl:+.2f}$ | Fees: {total_fees:.2f}$ (Drag: {portfolio_fee_drag:.1f}%) | "
        f"Net PnL: {total_net_pnl:+.2f} USDT across {args.days} days."
    )
    print("=" * 128 + "\n")

    # Diagnostic Deep Dive per Gate
    print("=" * 90)
    print("  🔍 PASS GATES VERIFICATION SUMMARY")
    print("=" * 90)
    for r in results:
        sym = r["symbol"]
        pnl_mark = "PASS" if r["pnl_gate"] else "FAIL"
        mdd_mark = "PASS" if r["mdd_gate"] else "FAIL"
        sl_mark = "PASS" if r["streak_gate"] else "FAIL"

        print(f"  • {sym:<10}:")
        print(f"    - Net Realized PnL: {r['net_pnl']:+.2f} USDT (> 0)              -> [{pnl_mark}]")
        print(f"    - Max Drawdown:     {r['mdd_pct']:.2f}% (<= {args.max_mdd_gate:.2f}%)          -> [{mdd_mark}]")
        print(f"    - Max SL Streak:    {r['max_hard_sl']} consecutive (< 3)           -> [{sl_mark}]")
        print(f"    - Parameters:       k_s={r['preset']['k_spread']}, k_tp={r['preset']['k_tp']}, k_sl={r['preset']['k_sl_mult']}, decay={r['preset']['time_decay_mins']}m")

    # Update config/top5_live_presets.json based on 30-day survival results
    cfg_file = Path(args.config)
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
            updated_any = False
            for r in results:
                sym_meta = TARGET_TOKENS.get(r["symbol"])
                if sym_meta:
                    u_sym = sym_meta["unified"]
                    if u_sym in cfg_data:
                        current_status = cfg_data[u_sym].get("enabled", True)
                        new_status = bool(r["passed_all"])
                        if current_status != new_status:
                            cfg_data[u_sym]["enabled"] = new_status
                            updated_any = True
                            logger.info(f"Updated {u_sym} enabled={new_status} in {args.config} based on 30-day stress replay.")
            if updated_any:
                with open(cfg_file, "w", encoding="utf-8") as f:
                    json.dump(cfg_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not update presets file with 30-day gate status: {e}")

    print("\n" + "=" * 90)
    if all_passed:
        print(f"  🎉 CONCLUSION: ALL {len(results)} TESTED TOKENS PASSED 30-DAY SURVIVAL REPLAY!")
    else:
        print("  ⚠️ CONCLUSION: ONE OR MORE TOKENS FAILED 30-DAY SURVIVAL GATES. REVIEW DETAILS ABOVE.")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    run_30d_stress_replay()
