"""Volatility-Anchored Market Making (VAMM) Quantitative Calculator & Inverted Grid Sizing."""
import math
from typing import Any, Dict, List, Tuple
from loguru import logger


def determine_order_levels(hurst: float, natr_15m_pct: float) -> int:
    """
    Determine the optimal number of order levels N based on Hurst exponent and 15m NATR:
    - If H > 0.44 -> N = 1 (Trending regime: Single level defense).
    - If H <= 0.35 and NATR_15m <= 1.40% -> N = 3 (Strong mean-reverting & moderate vol: 3 levels).
    - Otherwise -> N = 2 (Moderate regime: 2 levels).
    """
    h = float(hurst)
    # Normalize NATR: if passed as percentage e.g. 1.40 vs decimal 0.014
    natr_pct = float(natr_15m_pct) if float(natr_15m_pct) > 0.05 else float(natr_15m_pct) * 100.0

    if h > 0.44:
        return 1
    elif h <= 0.35 and natr_pct <= 1.40:
        return 3
    else:
        return 2


def compute_dynamic_vamm(
    natr_15m_pct: float,
    hurst: float,
    allocated_margin: float,
    leverage: int,
    maker_fee: float = 0.0002,
    taker_fee: float = 0.0005,
) -> Dict[str, Any]:
    """
    Compute Dynamic VAMM parameters with Inverted Sizing and Hurst volatility multiplier:
    - s_floor = max(0.0025, 2 * maker_fee + taker_fee + 0.0010)
    - s_base = max(s_floor, 0.30 * NATR_15m)
    - delta_s_step = 0.45 * NATR_15m
    - S_max = s_base + (N - 1) * delta_s_step
    - m_H = 1.0 + max(0.0, H - 0.35) * 0.8
    - TP_base = max(0.0065, 0.70 * NATR_15m * m_H)
    - SL = S_max + 0.60 * NATR_15m
    - Inverted weights: N=3 -> [0.50, 0.30, 0.20], N=2 -> [0.60, 0.40], N=1 -> [1.00]
    """
    natr_dec = float(natr_15m_pct) / 100.0 if float(natr_15m_pct) > 0.05 else float(natr_15m_pct)
    natr_dec = max(0.0001, natr_dec)
    h = float(hurst)
    lev = max(1, int(leverage))
    margin = float(allocated_margin)

    # 1. Determine Dynamic Levels N
    n_levels = determine_order_levels(h, natr_dec)

    # 2. Spread Floor
    s_floor = max(0.0025, (2.0 * maker_fee) + taker_fee + 0.0010)

    # 3. Base Spread (Level 0)
    s_base = max(s_floor, 0.30 * natr_dec)

    # 4. Step Spread
    delta_s_step = 0.45 * natr_dec

    # 5. S_max
    s_max = s_base + (n_levels - 1) * delta_s_step

    # 6. Hurst Multiplier & Take Profit
    m_h = 1.0 + max(0.0, h - 0.35) * 0.8
    tp_base = max(0.0065, 0.70 * natr_dec * m_h)

    # 7. Stop Loss Barrier
    sl = s_max + 0.60 * natr_dec

    # 8. Inverted Sizing Weights & Notionals
    if n_levels == 3:
        level_weights = [0.50, 0.30, 0.20]
    elif n_levels == 2:
        level_weights = [0.60, 0.40]
    else:
        level_weights = [1.00]

    total_notional_capacity = margin * lev
    level_notionals = [round(w * total_notional_capacity, 2) for w in level_weights]

    trailing_activation = tp_base
    trailing_callback = max(0.0020, 0.20 * tp_base)

    gross_exposure_cap = round(total_notional_capacity, 1)
    max_side_usdt = round(sum(level_notionals) * 1.05, 2)

    params = {
        "allocated_margin_usdt": margin,
        "leverage": lev,
        "order_levels": n_levels,
        "bid_spread": round(s_base, 6),
        "ask_spread": round(s_base, 6),
        "minimum_spread": round(s_floor, 6),
        "order_level_spread": round(delta_s_step, 6),
        "order_amount_usdt": level_notionals[0],
        "order_level_amount": 0.0,
        "level_weights": level_weights,
        "level_notionals": level_notionals,
        "take_profit": round(tp_base, 6),
        "trailing_tp_enabled": True,
        "trailing_tp_activation_pct": round(trailing_activation, 6),
        "trailing_tp_callback_pct": round(trailing_callback, 6),
        "stop_loss": round(sl, 6),
        "max_long_usdt": max_side_usdt,
        "max_short_usdt": max_side_usdt,
        "gross_exposure_cap_usdt": gross_exposure_cap,
        "level_cooldown_sec": 1800,
        "hurst": h,
        "natr_15m": natr_dec,
        "s_floor": round(s_floor, 6),
        "s_base": round(s_base, 6),
        "s_max": round(s_max, 6),
        "tp_base": round(tp_base, 6),
    }

    logger.info(
        f"[INVERTED_GRID][DYNAMIC_VAMM] Computed N={n_levels} (H={h:.3f}, NATR={natr_dec*100:.2f}%): "
        f"weights={level_weights}, notionals={level_notionals}, s_base={s_base*100:.3f}%, "
        f"step={delta_s_step*100:.3f}%, TP={tp_base*100:.3f}%, SL={sl*100:.3f}%"
    )
    return params
