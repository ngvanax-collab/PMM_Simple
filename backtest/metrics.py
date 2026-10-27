"""Quantitative Performance Metrics & Reporting for PMM Backtest Engine."""
import math
from typing import Any, Dict, List
import numpy as np
import pandas as pd


def calculate_metrics(
    trades: List[Dict[str, Any]],
    equity_curve: pd.DataFrame,
    initial_balance: float = 1000.0,
    interval: str = "1m",
    cb_trigger_count: int = 0,
    blocked_trend_trades: int = 0,
    fast_exit_count: int = 0,
    fast_exit_saved_usdt: float = 0.0,
) -> Dict[str, Any]:
    """
    Calculate quantitative backtest metrics:
    - Total Trades, Win Rate (%)
    - Gross Realized PnL (USDT)
    - Total Trading Fees (USDT)
    - Fee Drag (%) = (Total Fees / |Gross Realized PnL|) * 100%
    - Net Realized PnL (USDT)
    - Max Drawdown (MDD %)
    - Profit Factor = Sum(Gross Wins) / Sum(|Gross Losses|)
    - Annualized Sharpe Ratio based on candle interval
    - Quantitative Risk Diagnostics (CB Triggers, Blocked Trend Trades, Fast Exits, Saved USDT)

    Parameters:
    -----------
    trades: List of completed trade dictionaries
    equity_curve: DataFrame containing ['timestamp', 'equity', 'balance', 'upnl']
    initial_balance: Initial account balance in USDT
    interval: Candle timeframe ('1m', '5m', '15m', '1h', etc.)
    cb_trigger_count: Total circuit breaker triggers
    blocked_trend_trades: Number of entries blocked by 1h trend bias
    fast_exit_count: Number of positions closed early by toxic volume or time decay
    fast_exit_saved_usdt: Estimated USDT loss prevented vs full hard SL

    Returns:
    --------
    Dict containing calculated quantitative metrics.
    """
    total_trades = len(trades)

    if total_trades == 0:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": 0.0,
            "gross_pnl": 0.0,
            "total_fees": 0.0,
            "total_slippage": 0.0,
            "fee_drag_pct": 0.0,
            "net_pnl": 0.0,
            "return_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "max_drawdown_usdt": 0.0,
            "sharpe_ratio": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "long_win_rate_pct": 0.0,
            "short_win_rate_pct": 0.0,
            "exit_reasons": {},
            "cb_trigger_count": int(cb_trigger_count),
            "blocked_trend_trades": int(blocked_trend_trades),
            "fast_exit_count": int(fast_exit_count),
            "fast_exit_saved_usdt": float(fast_exit_saved_usdt),
        }

    # Analyze trades
    gross_pnls = [t["gross_pnl"] for t in trades]
    net_pnls = [t["net_pnl"] for t in trades]
    fees = [t["total_fee"] for t in trades]
    slippages = [t.get("slippage_cost", 0.0) for t in trades]

    total_gross_pnl = float(sum(gross_pnls))
    total_net_pnl = float(sum(net_pnls))
    total_fees = float(sum(fees))
    total_slippage = float(sum(slippages))

    winning_trades = sum(1 for pnl in net_pnls if pnl > 0)
    losing_trades = sum(1 for pnl in net_pnls if pnl <= 0)
    win_rate_pct = (winning_trades / total_trades) * 100.0 if total_trades > 0 else 0.0

    # Fee Drag calculation: (Total Fees / |Gross Realized PnL|) * 100%
    if abs(total_gross_pnl) > 1e-9:
        fee_drag_pct = (total_fees / abs(total_gross_pnl)) * 100.0
    else:
        fee_drag_pct = 100.0 if total_fees > 0 else 0.0

    # Profit Factor
    gross_wins = sum(p for p in gross_pnls if p > 0)
    gross_losses = sum(abs(p) for p in gross_pnls if p < 0)
    if gross_losses > 1e-9:
        profit_factor = gross_wins / gross_losses
    elif gross_wins > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    # Long vs Short breakdown
    long_trades_list = [t for t in trades if t.get("side") == "LONG"]
    short_trades_list = [t for t in trades if t.get("side") == "SHORT"]

    long_wins = sum(1 for t in long_trades_list if t["net_pnl"] > 0)
    short_wins = sum(1 for t in short_trades_list if t["net_pnl"] > 0)

    long_win_rate_pct = (long_wins / len(long_trades_list) * 100.0) if long_trades_list else 0.0
    short_win_rate_pct = (short_wins / len(short_trades_list) * 100.0) if short_trades_list else 0.0

    # Exit reasons breakdown
    exit_reasons: Dict[str, int] = {}
    for t in trades:
        reason = t.get("exit_reason", "UNKNOWN")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    # Max Drawdown from Equity Curve
    max_drawdown_pct = 0.0
    max_drawdown_usdt = 0.0

    if not equity_curve.empty and "equity" in equity_curve.columns:
        equities = equity_curve["equity"].values
        running_max = np.maximum.accumulate(equities)
        drawdowns_usdt = running_max - equities
        drawdowns_pct = np.where(running_max > 0, drawdowns_usdt / running_max, 0.0) * 100.0

        max_drawdown_pct = float(np.max(drawdowns_pct)) if len(drawdowns_pct) > 0 else 0.0
        max_drawdown_usdt = float(np.max(drawdowns_usdt)) if len(drawdowns_usdt) > 0 else 0.0

    # Annualized Sharpe Ratio from Equity Curve returns
    sharpe_ratio = 0.0
    if not equity_curve.empty and len(equity_curve) > 1 and "equity" in equity_curve.columns:
        equity_series = equity_curve["equity"].astype(float)
        candle_returns = equity_series.pct_change().dropna()

        # Determine annualization factor
        annualization_map = {
            "1m": 525600.0,
            "3m": 175200.0,
            "5m": 105120.0,
            "15m": 35040.0,
            "30m": 17520.0,
            "1h": 8760.0,
            "4h": 2190.0,
            "1d": 365.0,
        }
        factor = annualization_map.get(interval, 525600.0)

        std = float(candle_returns.std())
        if std > 1e-12:
            mean_ret = float(candle_returns.mean())
            sharpe_ratio = float((mean_ret / std) * math.sqrt(factor))

    return_pct = (total_net_pnl / initial_balance) * 100.0 if initial_balance > 0 else 0.0

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate_pct": win_rate_pct,
        "gross_pnl": total_gross_pnl,
        "total_fees": total_fees,
        "total_slippage": total_slippage,
        "fee_drag_pct": fee_drag_pct,
        "net_pnl": total_net_pnl,
        "return_pct": return_pct,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown_pct,
        "max_drawdown_usdt": max_drawdown_usdt,
        "sharpe_ratio": sharpe_ratio,
        "long_trades": len(long_trades_list),
        "short_trades": len(short_trades_list),
        "long_win_rate_pct": long_win_rate_pct,
        "short_win_rate_pct": short_win_rate_pct,
        "exit_reasons": exit_reasons,
        "cb_trigger_count": int(cb_trigger_count),
        "blocked_trend_trades": int(blocked_trend_trades),
        "fast_exit_count": int(fast_exit_count),
        "fast_exit_saved_usdt": float(fast_exit_saved_usdt),
    }


def format_metrics_report(metrics: Dict[str, Any], symbol: str = "SOLUSDT") -> str:
    """Format metrics into a sleek terminal-friendly report."""
    pf_str = f"{metrics['profit_factor']:.2f}" if metrics['profit_factor'] != float("inf") else "∞ (No Losses)"
    reasons_str = ", ".join(f"{k}: {v}" for k, v in metrics.get("exit_reasons", {}).items()) or "None"

    gross_str = f"${metrics['gross_pnl']:+.4f} USDT"
    fees_str = f"${metrics['total_fees']:.4f} USDT"
    slip_str = f"${metrics['total_slippage']:.4f} USDT"
    drag_str = f"{metrics['fee_drag_pct']:.2f}%"
    net_str = f"${metrics['net_pnl']:+.4f} USDT ({metrics['return_pct']:+.2f}%)"

    cb_count = metrics.get("cb_trigger_count", 0)
    blocked_count = metrics.get("blocked_trend_trades", 0)
    fast_count = metrics.get("fast_exit_count", 0)
    saved_usdt = metrics.get("fast_exit_saved_usdt", 0.0)
    saved_str = f"${saved_usdt:+.4f} USDT"

    lines = [
        "=" * 68,
        f"        OPENPMM HISTORICAL BACKTEST REPORT: {symbol}",
        "=" * 68,
        f"  Total Trades:            {metrics['total_trades']:<10} | Win Rate:          {metrics['win_rate_pct']:.2f}%",
        f"  Long Trades:             {metrics['long_trades']:<10} | Long Win Rate:     {metrics['long_win_rate_pct']:.2f}%",
        f"  Short Trades:            {metrics['short_trades']:<10} | Short Win Rate:    {metrics['short_win_rate_pct']:.2f}%",
        "-" * 68,
        f"  Gross Realized PnL:      {gross_str:<18}",
        f"  Total Trading Fees:      {fees_str:<18} (Maker: 0.02%, Taker: 0.05%)",
        f"  Total Slippage Cost:     {slip_str:<18} (SL penalty: 0.05%)",
        f"  Fee Drag Ratio:          {drag_str:<18} (Total Fees / |Gross PnL|)",
        f"  Net Realized PnL:        {net_str:<18}",
        "-" * 68,
        f"  Profit Factor:           {pf_str:<10}",
        f"  Max Drawdown (MDD):      {metrics['max_drawdown_pct']:.2f}% (${metrics['max_drawdown_usdt']:.2f} USDT)",
        f"  Sharpe Ratio (Ann):      {metrics['sharpe_ratio']:.2f}",
        f"  Exit Breakdown:          {reasons_str}",
        "-" * 68,
        f"  DIGITAL TWIN QUANTITATIVE RISK DIAGNOSTICS:",
        f"  - CB 60s Trigger Count:  {cb_count:<10} (Volatility Freeze Periods)",
        f"  - Blocked Trend Trades:  {blocked_count:<10} (Counter-Trend 1h EMA50 Protection)",
        f"  - Fast Exit Count:       {fast_count:<10} (Toxic Flow / Time-Decay SL Exits)",
        f"  - Fast Exit Saved USDT:  {saved_str:<18} (Loss Prevented vs Full Hard SL)",
        "=" * 68,
    ]
    return "\n".join(lines)
