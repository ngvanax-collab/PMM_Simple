#!/bin/bash
ASUS_HOST="ax99@100.95.85.9"
ASUS_DIR="/home/ax99/PMM Simple"

mkdir -p data logs

echo "Đang đồng bộ toàn bộ thư mục data (kèm file WAL/SHM) từ máy ASUS..."
rsync -avz -s "$ASUS_HOST:$ASUS_DIR/data/" ./data/

echo "Đang kéo 1000 dòng log mới nhất..."
ssh "$ASUS_HOST" "tail -n 1000 '$ASUS_DIR/logs/runtime.log'" > ./logs/asus_runtime.log 2>/dev/null || \
ssh "$ASUS_HOST" "tail -n 1000 \$(ls -t '$ASUS_DIR'/logs/pmm_engine_*.log | head -n 1)" > ./logs/asus_runtime.log

echo "Đồng bộ hoàn tất!"
