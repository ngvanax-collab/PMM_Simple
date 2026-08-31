# OpenPMM-Engine v3 (Futures Hedge Mode Native)

Engine PMM (Pure Market Making) **Futures-only, Hedge Mode native** hỗ trợ 20–30 cặp đồng thời, đầy đủ chức năng tương đương **Hummingbot PMM Simple V2 + Perpetual Market Making**, nhưng đơn giản, hiệu năng cao trên nền tảng single-process asyncio, Web UI NiceGUI (Port 8502), SQLite (WAL mode) và CCXT.pro.

---

## 🌟 Điểm nổi bật & Quy tắc bảo vệ vốn

1. **Hedge Mode Contract Invariant**:
   - Mọi payload lệnh/vị thế đều có `positionSide` (`LONG` hoặc `SHORT`).
   - Tuyệt đối không dùng `reduceOnly=True` hoặc `closePosition=True` (Binance sẽ từ chối trong Hedge Mode).
   - Đóng LONG = SELL + `positionSide='LONG'`, Đóng SHORT = BUY + `positionSide='SHORT'`.
2. **SL Quantity Synchronization**:
   - Lệnh Stop Loss server-side `STOP_MARKET` cho từng `positionSide`.
   - Khi khớp partial TP, Executor lập tức cập nhật lại quantity cho lệnh `STOP_MARKET` đúng bằng size vị thế còn lại, chống lật vị thế ngược.
3. **Triple Barrier Executor per positionSide**:
   - TP đa tầng (`tp_levels`), SL server-side, Trailing Stop (activation price + trailing delta), Time limit exit, và Post-SL cooldown.
4. **Token Bucket Rate Limiter**:
   - Quản lý đồng thời `ORDERS` và `WEIGHT` kèm jitter ngẫu nhiên ±20%.
5. **6-Phase Emergency Kill-All**:
   - Dừng worker $\rightarrow$ Cancel all $\rightarrow$ Fetch positions thật $\rightarrow$ MARKET close theo từng `positionSide` $\rightarrow$ Re-fetch xác nhận 100% Flat $\rightarrow$ Báo cáo chi tiết.
6. **Web Dashboard (NiceGUI Port 8502)**:
   - Quản trị API Key & Secret (mã hóa AES-256 an toàn), kết nối sàn & kiểm tra Hedge mode, chỉnh sửa cấu hình các cặp (SOL, BTC, ETH preset), giám sát lưới 2-slot LONG/SHORT, và nút Emergency Kill-All.

---

## 🚀 Hướng dẫn khởi chạy

### 1. Khởi động Web UI:
```bash
python3 app/main.py
```
Mở trình duyệt truy cập: `http://localhost:8502`

### 2. Vận hành qua Web UI:
1. Chuyển sang Tab **"🔑 API & Exchange Settings"**:
   - Chọn sàn (`binance` hoặc `bybit`).
   - Nhập **API Key** và **API Secret**.
   - Bấm **"Test & Connect (Verify Hedge Mode)"**. Hệ thống sẽ kiểm tra và tự động kích hoạt Dual/Hedge Mode nếu sàn đang ở One-Way.
2. Chuyển sang Tab **"⚙️ Pair Configurations"**:
   - Tùy chỉnh tham số hoặc bấm các nút preset nhanh (SOL, BTC, ETH).
3. Chuyển sang Tab **"⚡ Live Grid Dashboard"**:
   - Bấm **"Start"** trên từng thẻ cặp coin hoặc bấm **"Start All Enabled"** để kích hoạt bot.
4. Trong trường hợp khẩn cấp: Bấm nút đỏ **"🚨 EMERGENCY KILL-ALL"** trên thanh tiêu đề.

### 3. Script Emergency Kill độc lập qua CLI:
Nếu process chính gặp sự cố, bạn có thể đóng toàn bộ vị thế qua terminal:
```bash
python3 scripts/emergency_kill.py --exchange binance
```

---

## 🧪 Chạy Kiểm thử Tự động:
```bash
pytest -v
```
