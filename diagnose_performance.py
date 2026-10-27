import sqlite3
import pandas as pd
from datetime import datetime

conn = sqlite3.connect("file:data/pmm_engine.db?mode=ro", uri=True)

print("=== 1. TỔNG QUAN HIỆU SUẤT THEO CẶP GIAO DỊCH ===")
query_pnl = """
SELECT 
    symbol,
    COUNT(*) as total_fills,
    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as win_count,
    SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as loss_count,
    ROUND(SUM(realized_pnl), 4) as gross_pnl,
    ROUND(SUM(fee), 4) as total_fee,
    ROUND(SUM(realized_pnl - fee), 4) as net_pnl
FROM fills
GROUP BY symbol
"""
try:
    df_pnl = pd.read_sql_query(query_pnl, conn)
    print(df_pnl.to_string(index=False))
except Exception as e:
    print(f"Lỗi truy vấn fills: {e}")

print("\n=== 2. PHÂN TÍCH 10 LỆNH THUA LỖ GẦN NHẤT (ADVERSE SELECTION) ===")
query_losses = """
SELECT 
    symbol, position_side, side, price, amount, realized_pnl, fee,
    datetime(timestamp, 'unixepoch', 'localtime') as fill_time
FROM fills 
WHERE realized_pnl < 0 
ORDER BY timestamp DESC LIMIT 10
"""
try:
    df_loss = pd.read_sql_query(query_losses, conn)
    print(df_loss.to_string(index=False))
except Exception as e:
    print(f"Lỗi truy vấn losses: {e}")

conn.close()
