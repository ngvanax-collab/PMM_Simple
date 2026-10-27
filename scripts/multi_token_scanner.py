import os
import sys
import re
import json
import argparse
import requests
import numpy as np
import pandas as pd
from datetime import datetime

# Đảm bảo nhận diện đúng thư mục gốc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtest.data_loader import fetch_binance_futures_klines
from backtest.pmm_simulator import PMMBacktestEngine

DEFAULT_FALLBACK = [
    "AVAXUSDT", "SOLUSDT", "NEARUSDT", "ICPUSDT", "INJUSDT",
    "ADAUSDT", "SUIUSDT", "XRPUSDT", "LINKUSDT", "DOTUSDT"
]

TRADFI_AND_JUNK = {
    "USDCUSDT", "FDUSDUSDT", "BTCDOMUSDT", "XAUUSDT", "XAGUSDT", 
    "CLUSDT", "BZUSDT", "SOXLUSDT", "SPCXUSDT", "SKHYNIXUSDT", 
    "SKHYUSDT", "MUUSDT", "KORUUSDT", "MSTRUSDT", "SNXXUSDT", "SNDKUSDT"
}

def clean_number(val, default=0.0):
    """Chuyển đổi chuỗi có ký tự lạ ($, %, USDT, space) thành số thực an toàn."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        if "∞" in val or "nan" in val.lower():
            return 999.0
        cleaned = re.sub(r"[^\d.-]", "", val)
        try:
            return float(cleaned)
        except ValueError:
            return default
    return default

def robust_extract_metrics(res: dict) -> dict:
    """Trích xuất linh hoạt mọi biến thể key từ engine/metrics."""
    def get_any(keys, default=0.0):
        for k in keys:
            if k in res and res[k] is not None:
                return clean_number(res[k], default)
        return default

    trades = int(get_any(["Total Trades", "total_trades", "trades"], default=0))
    win_rate = get_any(["Win Rate (%)", "Win Rate", "win_rate"], default=0.0)
    net_pnl = get_any(["Net Realized PnL (USDT)", "Net Realized PnL", "Net PnL (USDT)", "net_pnl", "net_profit"], default=0.0)
    gross_pnl = get_any(["Gross Realized PnL (USDT)", "Gross Realized PnL", "Gross PnL (USDT)", "gross_pnl"], default=0.0)
    fees = get_any(["Total Trading Fees (USDT)", "Total Trading Fees", "Total Fee (USDT)", "total_fees", "fees"], default=0.0)
    fee_drag = get_any(["Fee Drag Ratio", "Fee Drag Ratio (%)", "Fee Drag (%)", "fee_drag"], default=100.0)
    sharpe = get_any(["Sharpe Ratio (Ann)", "Sharpe Ratio", "sharpe_ratio", "sharpe"], default=0.0)
    mdd = get_any(["Max Drawdown (MDD %)", "Max Drawdown (MDD)", "Max Drawdown (%)", "max_drawdown", "mdd"], default=0.0)

    # Nếu fee_drag chưa tính được nhưng có fees và gross_pnl
    if fee_drag == 100.0 and gross_pnl != 0.0:
        fee_drag = round((fees / abs(gross_pnl)) * 100.0, 2)

    return {
        "trades": trades,
        "win_rate": win_rate,
        "net_pnl": net_pnl,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "fee_drag": fee_drag,
        "sharpe": sharpe,
        "mdd": mdd
    }

def discover_crypto_futures_tokens(top_n: int = 25, min_volume_usdt: float = 30_000_000.0) -> list:
    """Lọc danh sách Crypto Futures thuần túy, loại bỏ hoàn toàn TradFi, Stock và Pump/Dump."""
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    print(f"[*] Đang quét toàn sàn Binance Futures (Lọc sạch TradFi & Extreme Volatility)...")
    try:
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return DEFAULT_FALLBACK

        data = res.json()
        candidates = []
        for item in data:
            sym = item.get("symbol", "")
            vol = float(item.get("quoteVolume", 0.0))
            price_change = abs(float(item.get("priceChangePercent", 0.0)))

            # 1. Chỉ lấy ký tự ASCII hợp lệ kết thúc bằng USDT
            if not sym.endswith("USDT") or not sym.isascii() or not sym.isalnum():
                continue
            # 2. Loại trừ TradFi, ETF, chỉ số
            if sym in TRADFI_AND_JUNK:
                continue
            # 3. Lọc khối lượng và biên độ biến động 24h <= 10%
            if vol >= min_volume_usdt and price_change <= 10.0:
                candidates.append((sym, vol))

        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [c[0] for c in candidates[:top_n]]
        print(f"[+] Đã chọn {len(selected)} crypto hợp lệ:")
        print("    " + ", ".join(selected))
        return selected
    except Exception as e:
        print(f"[!] Lỗi API: {e}. Dùng danh sách mặc định.")
        return DEFAULT_FALLBACK

def calculate_fitness(trades: int, sharpe: float, fee_drag: float, net_pnl: float, mdd: float, min_trades: int = 10, max_fee_drag: float = 30.0) -> float:
    if trades < min_trades or fee_drag > max_fee_drag or net_pnl <= 0.0 or mdd > 1.50:
        return -999.0
    stat_weight = np.sqrt(min(trades, 40))
    fee_efficiency = max(0.0, 1.0 - (fee_drag / 100.0))
    drawdown_penalty = 1.0 / (1.0 + 2.0 * (mdd / 100.0))
    return float(sharpe * stat_weight * fee_efficiency * drawdown_penalty)

def scan_token_universe():
    parser = argparse.ArgumentParser(description="Multi-Token Quant Grid Scanner & Selector")
    parser.add_argument("--days", type=int, default=14, help="Số ngày dữ liệu backtest")
    parser.add_argument("--interval", type=str, default="1m", help="Khung thời gian nến")
    parser.add_argument("--order-amount", type=float, default=30.0, help="Khối lượng vào lệnh danh nghĩa (USDT)")
    parser.add_argument("--output-config", type=str, default="config/top5_live_presets.json", help="File JSON lưu cấu hình Top 5")
    parser.add_argument("--top-pool", type=int, default=20, help="Số lượng token lọc từ sàn")
    parser.add_argument("--tokens", nargs="+", default=None, help="Danh sách token chỉ định thủ công")
    args = parser.parse_args()

    os.makedirs("config", exist_ok=True)
    os.makedirs("backtest/reports", exist_ok=True)

    if args.tokens:
        target_tokens = args.tokens
    else:
        target_tokens = discover_crypto_futures_tokens(top_n=args.top_pool)

    print("=" * 80)
    print(f" 🚀 OPENPMM MULTI-TOKEN QUANT SCANNER ({len(target_tokens)} TOKENS | {args.days} DAYS)")
    print(f" Tiêu chí: Fee Drag <= 30% | Trades >= 10 | Net PnL > 0 | MDD <= 1.5%")
    print("=" * 80)

    spreads = [0.0050, 0.0065, 0.0080, 0.0095, 0.0110]
    tps = [0.0080, 0.0100, 0.0120, 0.0140]
    sl_multipliers = [1.5, 2.0]

    all_token_bests = []
    full_audit_records = []

    for t_idx, symbol in enumerate(target_tokens, start=1):
        print(f"\n[{t_idx}/{len(target_tokens)}] Đang nạp & phân tích: {symbol}...")
        try:
            df_candles = fetch_binance_futures_klines(symbol, interval=args.interval, days=args.days)
            if df_candles.empty or len(df_candles) < 1000:
                print(f"  [!] Dữ liệu {symbol} không khả dụng. Bỏ qua.")
                continue
        except Exception as e:
            print(f"  [!] Lỗi nạp dữ liệu {symbol}: {e}. Bỏ qua.")
            continue

        best_combo = None
        best_fitness = -9999.0

        for spread in spreads:
            for tp in tps:
                for sl_mult in sl_multipliers:
                    sl = round(tp * sl_mult, 4)
                    trailing_cb = 0.0030

                    engine = PMMBacktestEngine(
                        symbol=symbol,
                        initial_balance=1000.0,
                        order_amount_usdt=args.order_amount,
                        base_spread_pct=spread,
                        take_profit_pct=tp,
                        stop_loss_pct=sl,
                        trailing_callback_pct=trailing_cb
                    )
                    raw_res = engine.run(df_candles)
                    m = robust_extract_metrics(raw_res)

                    fitness = calculate_fitness(m["trades"], m["sharpe"], m["fee_drag"], m["net_pnl"], m["mdd"])

                    record = {
                        "Symbol": symbol,
                        "Spread (%)": round(spread * 100, 2),
                        "TP (%)": round(tp * 100, 2),
                        "SL (%)": round(sl * 100, 2),
                        "Trades": m["trades"],
                        "WinRate (%)": round(m["win_rate"], 2),
                        "Fee Drag (%)": round(m["fee_drag"], 2),
                        "Net PnL (USDT)": round(m["net_pnl"], 4),
                        "MDD (%)": round(m["mdd"], 2),
                        "Sharpe": round(m["sharpe"], 2),
                        "Fitness": round(fitness, 2)
                    }
                    full_audit_records.append(record)

                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_combo = record

        if best_combo and best_fitness > -999.0:
            all_token_bests.append(best_combo)
            print(f"  [✓] Sweet Spot: Spread={best_combo['Spread (%)']}% | TP={best_combo['TP (%)']}% | Net={best_combo['Net PnL (USDT)']} | Sharpe={best_combo['Sharpe']} | Score={best_combo['Fitness']}")
        else:
            print(f"  [✗] Không có tổ hợp nào thỏa mãn tiêu chuẩn an toàn.")

    pd.DataFrame(full_audit_records).to_csv("backtest/reports/multi_token_raw_audit.csv", index=False)

    if not all_token_bests:
        print("\n[!] Không có token nào đạt đủ điều kiện an toàn đưa vào live trade.")
        return

    df_bests = pd.DataFrame(all_token_bests).sort_values(by="Fitness", ascending=False)
    top_5 = df_bests.head(5)

    print("\n" + "=" * 90)
    print(" 🏆 BẢNG XẾP HẠNG TOP 5 TOKEN ĐƯỢC CHỌN ĐƯA VÀO LIVE TRADE")
    print("=" * 90)
    print(top_5.to_string(index=False))
    print("=" * 90)

    live_presets = {}
    for _, row in top_5.iterrows():
        sym_clean = row["Symbol"]
        pair_key = f"{sym_clean.replace('USDT', '')}/USDT:USDT"
        
        live_presets[pair_key] = {
            "symbol": pair_key,
            "enabled": True,
            "leverage": 5,
            "margin_mode": "isolated",
            "allocated_margin_usdt": 50.0,
            "order_amount_usdt": float(args.order_amount),
            "order_levels": 1,
            "order_level_spread": 0.0050,
            "order_level_amount": 0.0,
            "bid_spread": float(row["Spread (%)"]) / 100.0,
            "ask_spread": float(row["Spread (%)"]) / 100.0,
            "take_profit": float(row["TP (%)"]) / 100.0,
            "stop_loss": float(row["SL (%)"]) / 100.0,
            "trailing_tp_enabled": True,
            "trailing_tp_activation_pct": float(row["TP (%)"]) / 100.0,
            "trailing_tp_callback_pct": 0.0030,
            "time_limit_sec": 14400,
            "base_cooldown_sec": 600,
            "max_long_usdt": float(args.order_amount) * 2.5,
            "max_short_usdt": float(args.order_amount) * 2.5,
            "gross_exposure_cap_usdt": 150.0,
            "backtest_score": float(row["Fitness"])
        }

    with open(args.output_config, "w", encoding="utf-8") as f:
        json.dump(live_presets, f, indent=2)

    print(f"\n[+] Đã xuất cấu hình chuẩn hóa Production ra file: {args.output_config}")

if __name__ == "__main__":
    scan_token_universe()
