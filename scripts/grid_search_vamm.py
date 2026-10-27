#!/usr/bin/env python3
"""
VAMM Dynamic Grid Search Scanner with Gaussian Plateau Filter.
Optimizes (k_spread, k_tp, k_sl_mult, time_decay_mins) across a 270-point parameter grid
using in-memory Parquet candle caching and Gaussian neighborhood plateau smoothing.
"""

import argparse
import json
import math
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

# ─────────────────────────────────────────────────────────────────────────────
# 1. SEARCH SPACE & TARGET TOKENS SPECIFICATION
# ─────────────────────────────────────────────────────────────────────────────
K_SPREAD_VALS = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
K_TP_VALS = [0.70, 0.85, 1.00, 1.20, 1.40]
K_SL_MULT_VALS = [1.5, 1.8, 2.0]
TIME_DECAY_VALS = [30, 45, 60]

TARGET_TOKENS = {
    "SOLUSDT": {
        "unified": "SOL/USDT:USDT",
        "tick_size": 0.01,
        "qty_step": 0.01,
    },
    "XRPUSDT": {
        "unified": "XRP/USDT:USDT",
        "tick_size": 0.0001,
        "qty_step": 0.1,
    },
    "1000PEPEUSDT": {
        "unified": "1000PEPE/USDT:USDT",
        "tick_size": 0.0000001,
        "qty_step": 1.0,
    },
    "NEARUSDT": {
        "unified": "NEAR/USDT:USDT",
        "tick_size": 0.001,
        "qty_step": 1.0,
    },
    "AVAXUSDT": {
        "unified": "AVAX/USDT:USDT",
        "tick_size": 0.001,
        "qty_step": 1.0,
    },
    "ETHUSDT": {
        "unified": "ETH/USDT:USDT",
        "tick_size": 0.01,
        "qty_step": 0.001,
    },
    "ADAUSDT": {
        "unified": "ADA/USDT:USDT",
        "tick_size": 0.0001,
        "qty_step": 1.0,
    },
    "UNIUSDT": {
        "unified": "UNI/USDT:USDT",
        "tick_size": 0.001,
        "qty_step": 1.0,
    },
    "LINKUSDT": {
        "unified": "LINK/USDT:USDT",
        "tick_size": 0.001,
        "qty_step": 0.01,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VAMM Grid Search Scanner with Gaussian Plateau Filter"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="ALL",
        help="Target symbol (SOLUSDT, XRPUSDT, 1000PEPEUSDT, NEARUSDT, or ALL)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Historical lookback window in days (default: 7)",
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
        "--export-config",
        type=str,
        default="config/top5_live_presets.json",
        help="File path to export optimized live presets JSON",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download historical candle data from Binance",
    )
    return parser.parse_args()


def calculate_raw_fitness(
    net_pnl: float,
    sharpe: float,
    trades: int,
    fee_drag: float,
    mdd: float,
) -> float:
    """
    Fitness = Sharpe * sqrt(min(Trades, 40)) * (1 - Fee_Drag / 100) * (1 / (1 + 2 * MDD))
    Clamped to 0.0 if Net PnL <= 0 or Sharpe <= 0.
    """
    if net_pnl <= 0 or sharpe <= 0 or trades == 0:
        return 0.0

    fee_factor = max(0.0, 1.0 - (fee_drag / 100.0))
    trade_factor = math.sqrt(min(trades, 40))
    mdd_factor = 1.0 / (1.0 + 2.0 * max(0.0, mdd))

    fitness = sharpe * trade_factor * fee_factor * mdd_factor
    return max(0.0, float(fitness))


def check_hard_gates(
    net_pnl: float,
    fee_drag: float,
    trades: int,
    mdd: float,
) -> bool:
    """
    Hard Gates:
    1. Net Realized PnL > 0
    2. Fee Drag Ratio <= 30.0% (target <= 25.0%)
    3. Total Trades >= 15
    4. Max Drawdown (MDD) <= 1.50%
    """
    return (net_pnl > 0.0) and (fee_drag <= 30.0) and (trades >= 15) and (mdd <= 1.50)


def compute_gaussian_plateau_scores(
    grid_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute Gaussian Neighborhood Plateau score across the 4D parameter grid.
    Grid shape: (len(k_spread), len(k_tp), len(k_sl_mult), len(time_decay)) = (6, 5, 3, 3).

    For each cell (i_s, i_tp, i_sl, i_td), we evaluate the 8 orthogonal neighbors along the 4 axes:
      (i_s ± 1, i_tp, i_sl, i_td)
      (i_s, i_tp ± 1, i_sl, i_td)
      (i_s, i_tp, i_sl ± 1, i_td)
      (i_s, i_tp, i_sl, i_td ± 1)

    Gaussian kernel weighting:
      w_center = 1.0 (d = 0)
      w_neighbor = exp(-0.5 * (1/sigma)^2) ≈ 0.6065 (d = 1, sigma = 1.0)

    Plateau Score = (w_center * Fitness_0 + sum(w_neighbor * Fitness_neighbor)) / (w_center + sum(w_neighbor))
    """
    shape = grid_matrix.shape
    plateau_scores = np.zeros(shape, dtype=np.float64)

    w_center = 1.0
    w_neighbor = math.exp(-0.5)  # ≈ 0.6065

    # 8 directional offsets along the 4 dimensions
    offsets = [
        (-1, 0, 0, 0), (1, 0, 0, 0),   # ± k_spread
        (0, -1, 0, 0), (0, 1, 0, 0),   # ± k_tp
        (0, 0, -1, 0), (0, 0, 1, 0),   # ± k_sl_mult
        (0, 0, 0, -1), (0, 0, 0, 1),   # ± time_decay
    ]

    for i_s in range(shape[0]):
        for i_tp in range(shape[1]):
            for i_sl in range(shape[2]):
                for i_td in range(shape[3]):
                    center_val = grid_matrix[i_s, i_tp, i_sl, i_td]

                    weighted_sum = w_center * center_val
                    weight_sum = w_center

                    for ds, dtp, dsl, dtd in offsets:
                        ns = i_s + ds
                        ntp = i_tp + dtp
                        nsl = i_sl + dsl
                        ntd = i_td + dtd

                        # Boundary check: valid in-grid neighbor
                        if 0 <= ns < shape[0] and 0 <= ntp < shape[1] and 0 <= nsl < shape[2] and 0 <= ntd < shape[3]:
                            weighted_sum += w_neighbor * grid_matrix[ns, ntp, nsl, ntd]
                            weight_sum += w_neighbor
                        else:
                            # Edge penalty: boundary counts with 0 fitness
                            weight_sum += w_neighbor

                    plateau_scores[i_s, i_tp, i_sl, i_td] = weighted_sum / max(1e-9, weight_sum)

    return plateau_scores


def run_token_grid_search(
    symbol: str,
    days: int,
    order_amount_usdt: float,
    max_holding_mins: int,
    force_download: Optional[bool] = None,
) -> Dict[str, Any]:
    """Execute complete 270-combination VAMM sweep on a single token."""
    token_meta = TARGET_TOKENS.get(symbol)
    if not token_meta:
        raise ValueError(f"Unknown target token: {symbol}")

    clean_sym = symbol.replace("/", "").replace(":USDT", "")
    logger.info(f"[{clean_sym}] Step 1/3: Loading 1m candles for lookback of {days} days...")

    # Load candles once into memory and precompute indicators without lookahead
    df = fetch_binance_futures_klines(
        symbol=clean_sym,
        interval="1m",
        days=days,
        force_download=force_download,
    )
    df = precompute_market_indicators(df)
    n_candles = len(df)
    mean_natr = float(df["natr_15m"].dropna().mean()) if "natr_15m" in df.columns else 0.0040
    logger.info(f"[{clean_sym}] Loaded {n_candles:,} candles (mean NATR 15m: {mean_natr*100:.3f}%).")

    # Step 2: Run 270 Grid Simulations
    logger.info(f"[{clean_sym}] Step 2/3: Sweeping 270 VAMM parameter configurations...")
    t0 = time.time()

    n_s = len(K_SPREAD_VALS)
    n_tp = len(K_TP_VALS)
    n_sl = len(K_SL_MULT_VALS)
    n_td = len(TIME_DECAY_VALS)

    raw_fitness_grid = np.zeros((n_s, n_tp, n_sl, n_td), dtype=np.float64)
    run_records: List[Dict[str, Any]] = []

    for i_s, ks in enumerate(K_SPREAD_VALS):
        for i_tp, ktp in enumerate(K_TP_VALS):
            for i_sl, ksl in enumerate(K_SL_MULT_VALS):
                for i_td, ktd in enumerate(TIME_DECAY_VALS):
                    engine = PMMBacktestEngine(
                        symbol=clean_sym,
                        initial_balance=1000.0,
                        order_amount_usdt=order_amount_usdt,
                        max_holding_mins=max_holding_mins,
                        tick_size=token_meta["tick_size"],
                        qty_step=token_meta["qty_step"],
                        k_spread=ks,
                        k_tp=ktp,
                        k_sl_mult=ksl,
                        time_decay_mins=ktd,
                        cb_threshold_pct=0.010,
                        vol_spike_threshold=3.0,
                        use_vamm=True,
                        use_circuit_breaker=True,
                        use_trend_bias=True,
                        use_toxic_exit=True,
                        use_time_decay_sl=True,
                    )
                    metrics = engine.run(df)

                    net_pnl = float(metrics["net_pnl"])
                    fee_drag = float(metrics["fee_drag_pct"])
                    trades = int(metrics["total_trades"])
                    mdd = float(metrics["max_drawdown_pct"])
                    sharpe = float(metrics["sharpe_ratio"])
                    win_rate = float(metrics["win_rate_pct"])

                    passed_gates = check_hard_gates(net_pnl, fee_drag, trades, mdd)
                    raw_fitness = calculate_raw_fitness(net_pnl, sharpe, trades, fee_drag, mdd)

                    raw_fitness_grid[i_s, i_tp, i_sl, i_td] = raw_fitness

                    run_records.append({
                        "coords": (i_s, i_tp, i_sl, i_td),
                        "k_spread": ks,
                        "k_tp": ktp,
                        "k_sl_mult": ksl,
                        "time_decay_mins": ktd,
                        "net_pnl": net_pnl,
                        "trades": trades,
                        "win_rate": win_rate,
                        "fee_drag": fee_drag,
                        "mdd": mdd,
                        "sharpe": sharpe,
                        "passed_gates": passed_gates,
                        "raw_fitness": raw_fitness,
                    })

    sweep_time = time.time() - t0
    logger.info(f"[{clean_sym}] Sweep completed: 270 runs in {sweep_time:.2f}s ({len(run_records)/max(0.001, sweep_time):.1f} runs/sec).")

    # Step 3: Gaussian Plateau Filter
    logger.info(f"[{clean_sym}] Step 3/3: Applying Gaussian Plateau Smoothing over 8-neighbor space...")
    plateau_scores = compute_gaussian_plateau_scores(raw_fitness_grid)

    for rec in run_records:
        coords = rec["coords"]
        rec["plateau_score"] = float(plateau_scores[coords])

    # Rank by Plateau Score, breaking ties with Net PnL, Fee Drag, and Drawdown
    run_records.sort(
        key=lambda x: (x["plateau_score"], x["net_pnl"], -x["fee_drag"], -x["mdd"]),
        reverse=True,
    )

    # Determine best configuration: prioritize candidates passing hard gates
    gate_passing = [r for r in run_records if r["passed_gates"]]
    best_config = gate_passing[0] if gate_passing else run_records[0]

    return {
        "symbol": clean_sym,
        "unified": token_meta["unified"],
        "mean_natr": mean_natr,
        "candles": n_candles,
        "sweep_time": sweep_time,
        "all_records": run_records,
        "gate_passing_count": len(gate_passing),
        "best_config": best_config,
        "top5": run_records[:5],
    }


def print_token_report(results: Dict[str, Any]) -> None:
    """Print high-fidelity Top 5 ranking table for a token."""
    sym = results["symbol"]
    best = results["best_config"]
    top5 = results["top5"]
    gate_count = results["gate_passing_count"]

    print("\n" + "=" * 108)
    print(f"  🏆 TOP 5 VAMM CONFIGURATIONS: {sym} (Passed Hard Gates: {gate_count}/270)")
    print("=" * 108)
    print(
        f"| {'Rank':<4} | {'k_spread':<8} | {'k_tp':<6} | {'k_sl':<5} | {'Decay':<5} | "
        f"{'Net PnL':<10} | {'Trades':<6} | {'WinRate':<7} | {'FeeDrag':<8} | {'MDD':<6} | {'Sharpe':<6} | {'Plateau':<8} | {'Gates':<5} |"
    )
    print("|" + "-" * 6 + "|" + "-" * 10 + "|" + "-" * 8 + "|" + "-" * 7 + "|" + "-" * 7 + "|" +
          "-" * 12 + "|" + "-" * 8 + "|" + "-" * 9 + "|" + "-" * 10 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 10 + "|" + "-" * 7 + "|")

    for rank, r in enumerate(top5, 1):
        is_best = "★" if r == best else " "
        gates_str = "PASS" if r["passed_gates"] else "FAIL"
        print(
            f"| {rank:02d}{is_best} | {r['k_spread']:<8.2f} | {r['k_tp']:<6.2f} | {r['k_sl_mult']:<5.1f} | {r['time_decay_mins']:<3d}m  | "
            f"{r['net_pnl']:+8.2f}$ | {r['trades']:<6d} | {r['win_rate']:5.1f}% | {r['fee_drag']:6.1f}% | {r['mdd']:5.2f}% | {r['sharpe']:6.2f} | {r['plateau_score']:7.2f}  | {gates_str:<5} |"
        )
    print("+" + "-" * 106 + "+")
    print(f"  Selected Optimum: k_spread={best['k_spread']}, k_tp={best['k_tp']}, k_sl={best['k_sl_mult']}, decay={best['time_decay_mins']}m (Score: {best['plateau_score']:.2f})")


def print_cross_token_summary(all_results: List[Dict[str, Any]]) -> None:
    """Print cross-token summary comparison matrix."""
    print("\n" + "=" * 116)
    print("  🌐 CROSS-TOKEN OPTIMIZATION MATRIX (ALL 4 TOKENS BENCHMARKED)")
    print("=" * 116)
    print(
        f"| {'Symbol':<14} | {'k_spread':<8} | {'k_tp':<6} | {'k_sl':<5} | {'Decay':<5} | "
        f"{'Net PnL':<10} | {'Trades':<6} | {'WinRate':<7} | {'FeeDrag':<8} | {'MDD':<6} | {'Sharpe':<6} | {'Score':<6} | {'Status':<6} |"
    )
    print("|" + "-" * 16 + "|" + "-" * 10 + "|" + "-" * 8 + "|" + "-" * 7 + "|" + "-" * 7 + "|" +
          "-" * 12 + "|" + "-" * 8 + "|" + "-" * 9 + "|" + "-" * 10 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 8 + "|")

    total_net_pnl = 0.0
    total_trades = 0

    for res in all_results:
        sym = res["symbol"]
        best = res["best_config"]
        status = "PASSED" if best["passed_gates"] else "RELAX"
        total_net_pnl += best["net_pnl"]
        total_trades += best["trades"]

        print(
            f"| {sym:<14} | {best['k_spread']:<8.2f} | {best['k_tp']:<6.2f} | {best['k_sl_mult']:<5.1f} | {best['time_decay_mins']:<3d}m  | "
            f"{best['net_pnl']:+8.2f}$ | {best['trades']:<6d} | {best['win_rate']:5.1f}% | {best['fee_drag']:6.1f}% | {best['mdd']:5.2f}% | {best['sharpe']:6.2f} | {best['plateau_score']:6.2f} | {status:<6} |"
        )

    print("+" + "-" * 114 + "+")
    print(f"  Total Portfolio Net PnL: {total_net_pnl:+.2f} USDT across {total_trades} trades in 7 days.")
    print("=" * 116 + "\n")


def update_live_presets_json(
    all_results: List[Dict[str, Any]],
    export_path: str,
    order_amount_usdt: float,
) -> None:
    """Save / update optimized preset configurations to config/top5_live_presets.json."""
    config_file = Path(export_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    existing_presets: Dict[str, Any] = {}
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                existing_presets = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load existing config ({e}), creating fresh.")

    for res in all_results:
        unified_sym = res["unified"]
        best = res["best_config"]
        mean_natr = res["mean_natr"]

        # Calculate nominal spread / TP / SL levels from VAMM multipliers
        nom_spread = max(0.0025, best["k_spread"] * mean_natr)
        nom_tp = max(0.0050, best["k_tp"] * mean_natr)
        nom_sl = nom_spread + (best["k_sl_mult"] * mean_natr)

        # Check qualification: Net PnL > 0, Plateau Score >= 1.00, Fee Drag <= 35.0%
        is_qualified = (
            best["net_pnl"] > 0.0
            and best["plateau_score"] >= 1.00
            and best["fee_drag"] <= 35.0
        )
        preset_data = {
            "symbol": unified_sym,
            "enabled": bool(is_qualified),
            "leverage": 5,
            "margin_mode": "isolated",
            "allocated_margin_usdt": 21.0,
            "order_amount_usdt": float(order_amount_usdt),
            "order_levels": 1,
            "order_level_spread": 0.005,
            "order_level_amount": 0.0,
            "bid_spread": round(nom_spread, 6),
            "ask_spread": round(nom_spread, 6),
            "take_profit": round(nom_tp, 6),
            "stop_loss": round(nom_sl, 6),
            "trailing_tp_enabled": True,
            "trailing_tp_activation_pct": round(nom_tp, 6),
            "trailing_tp_callback_pct": 0.003,
            "time_limit_sec": 14400,
            "base_cooldown_sec": 600,
            "max_long_usdt": 75.0,
            "max_short_usdt": 75.0,
            "gross_exposure_cap_usdt": 150.0,
            "backtest_score": round(best["plateau_score"], 2),
            # Digital Twin Dynamic VAMM Parameters
            "k_spread": best["k_spread"],
            "k_tp": best["k_tp"],
            "k_sl_mult": best["k_sl_mult"],
            "time_decay_mins": best["time_decay_mins"],
            "max_holding_mins": 244,
        }

        existing_presets[unified_sym] = preset_data

        # If AVAX qualifies, replace 1000PEPE/USDT:USDT
        if unified_sym == "AVAX/USDT:USDT" and best["net_pnl"] > 0 and best["plateau_score"] > 1.0:
            if "1000PEPE/USDT:USDT" in existing_presets:
                existing_presets.pop("1000PEPE/USDT:USDT", None)
                logger.info("Successfully replaced 1000PEPE/USDT:USDT with AVAX/USDT:USDT in live presets.")

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(existing_presets, f, indent=2)

    logger.info(f"Successfully exported optimized configurations for {len(all_results)} pairs to {config_file}.")


def main():
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}", level="INFO")

    print("\n" + "=" * 90)
    print("  ⚡ OPENPMM VAMM GRID SEARCH SCANNER WITH GAUSSIAN PLATEAU FILTER")
    print("  Hardware: DELL SANDBOX (16GB RAM) | 270 Combinations / Pair")
    print(f"  Target: {args.symbol} | Lookback: {args.days} days | Order Amount: ${args.order_amount:.1f} USDT")
    print("  Hard Gates: Net PnL > 0 | Fee Drag <= 25-35% | Trades >= 15 | Max Drawdown <= 1.50%")
    print("=" * 90)

    # Determine symbols to sweep
    if args.symbol.upper() == "ALL":
        target_symbols = list(TARGET_TOKENS.keys())
    else:
        raw_items = [s.strip() for s in args.symbol.split(",") if s.strip()]
        target_symbols = []
        for s in raw_items:
            clean = s.replace("/", "").replace(":USDT", "").upper()
            if clean not in TARGET_TOKENS:
                logger.critical(f"Symbol {s} is not supported. Supported: {list(TARGET_TOKENS.keys())}")
                sys.exit(1)
            target_symbols.append(clean)

    all_results: List[Dict[str, Any]] = []

    for sym in target_symbols:
        res = run_token_grid_search(
            symbol=sym,
            days=args.days,
            order_amount_usdt=args.order_amount,
            max_holding_mins=args.max_holding_mins,
            force_download=args.force_download if args.force_download else None,
        )
        print_token_report(res)
        all_results.append(res)

    # Cross-token summary
    if len(all_results) > 1:
        print_cross_token_summary(all_results)

    # Export to config
    update_live_presets_json(
        all_results=all_results,
        export_path=args.export_config,
        order_amount_usdt=args.order_amount,
    )


if __name__ == "__main__":
    main()
