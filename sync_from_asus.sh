#!/usr/bin/env bash
set -euo pipefail

ASUS_USER="ax99"
ASUS_IP="100.95.85.9"
ASUS_DIR="/home/ax99/PMM Simple"
DELL_DIR="/home/dx99/PMM_Sandbox"

mkdir -p "$DELL_DIR/data" "$DELL_DIR/logs"

# 1. Tạo snapshot nguyên khối từ Asus
ssh "${ASUS_USER}@${ASUS_IP}" "python3 -c \"
import sqlite3, os
snap = '${ASUS_DIR}/data/snapshot.db'
if os.path.exists(snap):
    os.remove(snap)
conn = sqlite3.connect('${ASUS_DIR}/data/pmm_engine.db')
conn.execute(f\\\"VACUUM INTO '{snap}'\\\")
conn.close()
\""

# 2. Xóa sạch file WAL/SHM cũ trên Dell để chống lệch phân trang
rm -f "$DELL_DIR/data/pmm_engine.db-wal" "$DELL_DIR/data/pmm_engine.db-shm"

# 3. Kéo snapshot sạch về Dell
rsync -az "${ASUS_USER}@${ASUS_IP}:${ASUS_DIR}/data/snapshot.db" "$DELL_DIR/data/pmm_engine.db"

# 4. Kéo runtime log về Dell
rsync -az --update "${ASUS_USER}@${ASUS_IP}:${ASUS_DIR}/logs/" "$DELL_DIR/logs/"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Snapshot VACUUM & Sync thành công."
