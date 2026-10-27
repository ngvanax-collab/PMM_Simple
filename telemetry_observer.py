import os
import re
import sqlite3
import subprocess
import time
from datetime import datetime, timezone

DELL_DIR = os.path.expanduser("~/PMM_Sandbox")
SYNC_SCRIPT = os.path.join(DELL_DIR, "sync_from_asus.sh")
DB_PATH = os.path.join(DELL_DIR, "data/pmm_engine.db")
LOG_PATH = os.path.join(DELL_DIR, "logs/runtime.log")
REPORT_LOG = os.path.join(DELL_DIR, "logs/telemetry_events.log")

SYNC_INTERVAL_SEC = 60

PATTERNS = {
    "POST_ONLY_REJECT": re.compile(r"-5022|would immediately match"),
    "STOP_ORDER_LIMIT": re.compile(r"-4045|max stop order limit"),
    "MARGIN_INSUFFICIENT": re.compile(r"-2019|Margin is insufficient"),
    "DISCONNECT_TIMEOUT": re.compile(r"(WebSocketClientDisconnected|TimeoutError|ConnectionResetError)"),
    "CIRCUIT_BREAKER": re.compile(r"(CIRCUIT_BREAKER_TRIPPED|FLASH_SPIKE)"),
    "UNHANDLED_EXCEPTION": re.compile(r"Traceback \(most recent call last\)|ERROR.*Exception"),
}

def sync_data():
    try:
        subprocess.run(["bash", SYNC_SCRIPT], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[Sync Warning] Không thể chạy sync script: {e}")

def audit_pnl_anomalies(hours=2):
    if not os.path.exists(DB_PATH):
        return []

    anomalies = []
    cutoff_ts = time.time() - (hours * 3600)
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Tự động nhận diện tên cột trong pnl_records
        cols = [col[1] for col in cursor.execute("PRAGMA table_info(pnl_records);").fetchall()]
        time_col = next((c for c in ['timestamp', 'ts', 'time', 'closed_at', 'created_at'] if c in cols), cols[0])
        pnl_col = next((c for c in ['realized_pnl', 'pnl', 'net_pnl'] if c in cols), 'realized_pnl')
        side_col = next((c for c in ['position_side', 'side', 'pos_side'] if c in cols), 'position_side')
        sym_col = next((c for c in ['symbol', 'pair'] if c in cols), 'symbol')

        query = f"""
            SELECT {sym_col}, {side_col}, {pnl_col}, {time_col}
            FROM pnl_records
            WHERE {time_col} >= ?
            ORDER BY {time_col} ASC
        """
        cursor.execute(query, (cutoff_ts,))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        streak = 0
        total_pnl = 0.0

        for sym, side, pnl, ts in rows:
            total_pnl += pnl
            if pnl < 0:
                streak += 1
                if streak >= 3:
                    # Hỗ trợ timestamp mili-giây hoặc giây
                    ts_sec = ts / 1000.0 if ts > 1e11 else ts
                    t_str = datetime.fromtimestamp(ts_sec, tz=timezone.utc).strftime("%H:%M:%S")
                    anomalies.append(f"[{t_str}] [LOSS_STREAK] Cặp {sym} ({side}) chuỗi lỗ {streak} lệnh liên tiếp!")
            else:
                streak = 0

        if total_pnl < -10.0:
            anomalies.append(f"[GLOBAL_DRAWDOWN] Net PnL trong {hours}h qua: {total_pnl:.2f} USDT")

    except Exception as err:
        anomalies.append(f"[DB Error] Lỗi phân tích SQLite: {err}")

    return anomalies

def check_log_errors():
    if not os.path.exists(LOG_PATH):
        return []

    detected = []
    try:
        res = subprocess.run(["tail", "-n", "200", LOG_PATH], capture_output=True, text=True)
        for line in res.stdout.splitlines():
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    detected.append(f"[{label}] {line.strip()[:140]}")
    except Exception:
        pass
    return list(set(detected))

def run_observer():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Trạm quan sát Telemetry trên Dell bắt đầu hoạt động...")
    os.makedirs(os.path.join(DELL_DIR, "logs"), exist_ok=True)

    while True:
        sync_data()

        log_errors = check_log_errors()
        pnl_anomalies = audit_pnl_anomalies(hours=2)

        all_events = log_errors + pnl_anomalies
        ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if all_events:
            print(f"\n=== [PHÁT HIỆN BẤT THƯỜNG] {ts_now} ===")
            with open(REPORT_LOG, "a", encoding="utf-8") as f:
                for ev in all_events:
                    print(f" -> {ev}")
                    f.write(f"[{ts_now}] {ev}\n")
        else:
            print(f"[{ts_now}] Sync & Check OK: Không phát hiện lỗi nghiêm trọng.")

        time.sleep(SYNC_INTERVAL_SEC)

if __name__ == "__main__":
    run_observer()
