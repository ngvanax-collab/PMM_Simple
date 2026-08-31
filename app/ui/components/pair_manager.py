"""Pair Management & Config Editor UI Component with Full Add & Delete Capabilities."""
from nicegui import ui
from app.core.manager import bot_manager
from app.models.config import PairConfig
from app.persistence.store import config_store


def render_pair_manager(on_configs_updated=None):
    """Render Pair Manager Tab with Live List, Form Editor, and Delete Support."""

    # Outer container
    with ui.column().classes("w-full max-w-5xl mx-auto gap-6"):

        # ── 1. Danh Sách Các Cặp Đang Lưu (Saved Pairs List & Deletion) ──
        saved_list_container = ui.column().classes("w-full")

        def refresh_saved_list():
            saved_list_container.clear()
            all_configs = config_store.load_all_pair_configs()

            with saved_list_container:
                with ui.card().classes("w-full p-6 bg-slate-900 border border-slate-800 rounded-xl shadow-xl"):
                    with ui.row().classes("w-full justify-between items-center mb-3 pb-2 border-b border-slate-800"):
                        with ui.row().classes("items-center gap-2"):
                            ui.icon("list_alt", size="sm").classes("text-primary")
                            ui.label("📋 Danh Sách Cặp Giao Dịch Đang Lưu").classes("text-xl font-bold text-white")
                            ui.badge(f"{len(all_configs)} Cặp").props("color=slate-700")

                        ui.button(icon="refresh", on_click=refresh_saved_list).props("flat round dense size=sm color=grey").tooltip("Làm mới danh sách")

                    if not all_configs:
                        with ui.column().classes("w-full items-center justify-center py-6 text-slate-500 gap-2"):
                            ui.icon("inbox", size="lg").classes("text-slate-600")
                            ui.label("Chưa có cặp giao dịch nào được lưu trong hệ thống.").classes("text-sm")
                            ui.label("Chọn Preset mẫu hoặc điền thông tin bên dưới để thêm cặp mới!").classes("text-xs text-slate-600")
                    else:
                        with ui.column().classes("w-full gap-3"):
                            for sym, cfg in all_configs.items():
                                with ui.card().classes("w-full p-4 bg-slate-950/70 border border-slate-800 rounded-lg"):
                                    with ui.row().classes("w-full justify-between items-center"):
                                        # Left: Symbol info & parameters
                                        with ui.column().classes("gap-1"):
                                            with ui.row().classes("items-center gap-2"):
                                                ui.label(sym).classes("text-base font-bold text-white")
                                                ui.badge(f"{cfg.leverage}x {cfg.margin_mode.upper()}").props("color=slate-800 text-color=slate-300 text-[10px]")
                                                if cfg.enabled:
                                                    ui.badge("ENABLED").props("color=positive text-[10px]")
                                                else:
                                                    ui.badge("DISABLED").props("color=grey text-[10px]")

                                            with ui.row().classes("text-xs text-slate-400 gap-3 mt-1"):
                                                ui.label(f"Order: ${cfg.order_amount_usdt:.1f}")
                                                ui.label(f"Alloc Margin: {('$' + str(cfg.allocated_margin_usdt)) if cfg.allocated_margin_usdt else 'Tự động'}")
                                                ui.label(f"Spread: {cfg.bid_spread*100:.2f}%/{cfg.ask_spread*100:.2f}%")
                                                ui.label(f"TP: {cfg.take_profit*100:.2f}% | SL: {cfg.stop_loss*100:.2f}%")
                                                ui.label(f"Max Loss: ${getattr(cfg, 'worker_max_loss_usdt', 30.0):.1f}")

                                        # Right: Action Buttons (Edit into form & Delete)
                                        with ui.row().classes("items-center gap-2"):
                                            def load_into_form(target_cfg=cfg):
                                                symbol_input.value = target_cfg.symbol
                                                leverage_input.value = target_cfg.leverage
                                                margin_select.value = target_cfg.margin_mode
                                                order_amt_input.value = target_cfg.order_amount_usdt
                                                bid_spread_input.value = target_cfg.bid_spread
                                                ask_spread_input.value = target_cfg.ask_spread
                                                order_levels_input.value = target_cfg.order_levels
                                                level_spread_input.value = target_cfg.order_level_spread
                                                level_amt_input.value = target_cfg.order_level_amount
                                                max_long_input.value = target_cfg.max_long_usdt
                                                max_short_input.value = target_cfg.max_short_usdt
                                                gross_cap_input.value = target_cfg.gross_exposure_cap_usdt
                                                allocated_margin_input.value = target_cfg.allocated_margin_usdt
                                                tp_input.value = target_cfg.take_profit
                                                sl_input.value = target_cfg.stop_loss
                                                time_limit_input.value = target_cfg.time_limit
                                                base_cooldown_input.value = getattr(target_cfg, "base_cooldown_sec", 900)
                                                cooldown_mult_input.value = getattr(target_cfg, "cooldown_multiplier", 2.0)
                                                max_cooldown_input.value = getattr(target_cfg, "max_cooldown_sec", 86400)
                                                worker_max_loss_input.value = getattr(target_cfg, "worker_max_loss_usdt", 30.0)
                                                worker_max_dd_input.value = getattr(target_cfg, "worker_max_drawdown_usdt", 40.0)
                                                ui.notify(f"Đã nạp cấu hình {target_cfg.symbol} vào Form bên dưới để chỉnh sửa!", type="info")

                                            def confirm_delete_pair(target_sym=sym):
                                                with ui.dialog() as del_dialog, ui.card().classes("bg-slate-900 border border-slate-700 p-6 rounded-xl max-w-md"):
                                                    ui.label("🗑️ XÁC NHẬN XÓA CẶP GIAO DỊCH").classes("text-lg font-bold text-rose-400 mb-2")
                                                    ui.label(
                                                        f"Bạn có chắc chắn muốn XÓA VĨNH VIỄN cặp {target_sym} khỏi hệ thống không? "
                                                        f"Tiến trình bot của cặp này sẽ được dừng và file cấu hình sẽ bị xóa."
                                                    ).classes("text-slate-300 text-sm mb-4")

                                                    async def do_delete():
                                                        del_dialog.close()
                                                        deleted = await bot_manager.delete_pair(target_sym)
                                                        if deleted:
                                                            ui.notify(f"Đã xóa hoàn toàn cặp {target_sym}!", type="positive")
                                                        else:
                                                            ui.notify(f"Xóa cặp {target_sym} thất bại hoặc không tìm thấy file.", type="warning")
                                                        refresh_saved_list()
                                                        if on_configs_updated:
                                                            on_configs_updated()

                                                    with ui.row().classes("w-full justify-end gap-3"):
                                                        ui.button("Hủy", on_click=del_dialog.close).props("flat color=grey")
                                                        ui.button("Xác Nhận Xóa", on_click=do_delete).props("color=negative")

                                                del_dialog.open()

                                            ui.button("✏️ Sửa", on_click=load_into_form).props("size=sm outline color=primary")
                                            ui.button("🗑️ Xóa", on_click=confirm_delete_pair).props("size=sm color=negative")

        # Initial render of saved pairs
        refresh_saved_list()

        # ── 2. Form Thêm / Sửa Cấu Hình Cặp (Form Editor) ──
        with ui.card().classes("w-full p-6 bg-slate-900 border border-slate-800 rounded-xl shadow-2xl"):
            with ui.row().classes("w-full justify-between items-center mb-4 pb-2 border-b border-slate-800"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("tune", size="sm").classes("text-emerald-400")
                    ui.label("➕ Thêm / Chỉnh Sửa Cấu Hình Cặp").classes("text-2xl font-bold text-white")
                ui.label("Tùy chỉnh thông số chi tiết và lưu cấu hình").classes("text-slate-400 text-sm")

            # ── Form Inputs ──
            with ui.grid(columns=3).classes("w-full gap-4 mb-4"):
                symbol_input = ui.input("Symbol (e.g. SOL/USDT:USDT)", value="SOL/USDT:USDT").classes("w-full")
                leverage_input = ui.number("Leverage (x)", value=5, min=1, max=125).classes("w-full")
                margin_select = ui.select(["isolated", "cross"], value="isolated", label="Margin Mode").classes("w-full")

            with ui.grid(columns=3).classes("w-full gap-4 mb-4"):
                order_amt_input = ui.number("Order Amount (USDT)", value=50.0, min=5.0).classes("w-full")
                bid_spread_input = ui.number("Base Bid Spread (0.003 = 0.3%)", value=0.003, min=0.0001, step=0.0005).classes("w-full")
                ask_spread_input = ui.number("Base Ask Spread (0.003 = 0.3%)", value=0.003, min=0.0001, step=0.0005).classes("w-full")

            with ui.grid(columns=3).classes("w-full gap-4 mb-4"):
                order_levels_input = ui.number("Order Levels", value=3, min=1, max=10).classes("w-full")
                level_spread_input = ui.number("Level Spread Step", value=0.002, min=0.0001, step=0.0005).classes("w-full")
                level_amt_input = ui.number("Level Amount Step (USDT)", value=25.0, min=0.0).classes("w-full")

            with ui.grid(columns=4).classes("w-full gap-4 mb-4"):
                max_long_input = ui.number("Max LONG Cap (USDT)", value=300.0, min=10.0).classes("w-full")
                max_short_input = ui.number("Max SHORT Cap (USDT)", value=300.0, min=10.0).classes("w-full")
                gross_cap_input = ui.number("Gross Exposure Cap (USDT)", value=450.0, min=10.0).classes("w-full")
                allocated_margin_input = ui.number("Allocated Margin Cap (USDT)", value=None, min=0.0).props("clearable").classes("w-full")
                ui.tooltip("Hạn mức USDT ký quỹ tối đa cấp riêng cho cặp này (để trống để tự tính từ Gross Cap / Leverage)").bind_visibility_from(allocated_margin_input)

            with ui.grid(columns=3).classes("w-full gap-4 mb-4"):
                tp_input = ui.number("Take Profit % (0.008 = 0.8%)", value=0.008, min=0.001, step=0.001).classes("w-full")
                sl_input = ui.number("Stop Loss % (0.02 = 2.0%)", value=0.02, min=0.001, step=0.001).classes("w-full")
                time_limit_input = ui.number("Time Limit (sec)", value=21600, min=60).classes("w-full")

            # ── 🛡️ Risk Management & Progressive Cooldown ──
            with ui.column().classes("w-full bg-slate-950/40 border border-slate-800/80 rounded-xl p-4 mb-6"):
                with ui.row().classes("items-center gap-2 mb-2"):
                    ui.icon("shield", size="sm").classes("text-amber-400")
                    ui.label("🛡️ Risk Management & Progressive Cooldown").classes("text-sm font-bold text-slate-200")

                with ui.grid(columns=3).classes("w-full gap-4 mb-3"):
                    base_cooldown_input = ui.number("Base Cooldown (giây)", value=900, min=0, step=60).classes("w-full")
                    cooldown_mult_input = ui.number("Cooldown Multiplier (hệ số)", value=2.0, min=1.0, max=10.0, step=0.5).classes("w-full")
                    max_cooldown_input = ui.number("Max Cooldown Cap (giây)", value=86400, min=300, step=3600).classes("w-full")

                with ui.grid(columns=2).classes("w-full gap-4"):
                    worker_max_loss_input = ui.number("Worker Max Loss Limit (USDT)", value=30.0, min=1.0, step=5.0).classes("w-full")
                    worker_max_dd_input = ui.number("Worker Max Drawdown Limit (USDT)", value=40.0, min=1.0, step=5.0).classes("w-full")

            async def save_pair():
                sym = symbol_input.value.strip().upper()
                if not sym:
                    ui.notify("Symbol cannot be empty!", type="warning")
                    return

                alloc_m = float(allocated_margin_input.value) if allocated_margin_input.value is not None and str(allocated_margin_input.value).strip() != "" and float(allocated_margin_input.value) > 0 else None

                existing_cfg = config_store.load_pair_config(sym)
                was_enabled = existing_cfg.enabled if existing_cfg else True

                cfg = PairConfig(
                    symbol=sym,
                    exchange="binance",
                    enabled=was_enabled,
                    leverage=int(leverage_input.value),
                    margin_mode=str(margin_select.value),
                    order_amount_usdt=float(order_amt_input.value),
                    bid_spread=float(bid_spread_input.value),
                    ask_spread=float(ask_spread_input.value),
                    order_levels=int(order_levels_input.value),
                    order_level_spread=float(level_spread_input.value),
                    order_level_amount=float(level_amt_input.value),
                    max_long_usdt=float(max_long_input.value),
                    max_short_usdt=float(max_short_input.value),
                    gross_exposure_cap_usdt=float(gross_cap_input.value),
                    allocated_margin_usdt=alloc_m,
                    take_profit=float(tp_input.value),
                    stop_loss=float(sl_input.value),
                    time_limit=int(time_limit_input.value),
                    base_cooldown_sec=int(base_cooldown_input.value or 900),
                    cooldown_multiplier=float(cooldown_mult_input.value or 2.0),
                    max_cooldown_sec=int(max_cooldown_input.value or 86400),
                    worker_max_loss_usdt=float(worker_max_loss_input.value or 30.0),
                    worker_max_drawdown_usdt=float(worker_max_dd_input.value or 40.0),
                    tp_levels=[[float(tp_input.value), 0.6], [float(tp_input.value) * 1.8, 0.4]],
                )

                config_store.save_pair_config(cfg)

                # Update worker in bot_manager
                if bot_manager.gateway:
                    from app.core.worker import PMMWorker
                    was_running = False
                    if sym in bot_manager.workers:
                        was_running = bot_manager.workers[sym].is_running
                        await bot_manager.workers[sym].stop()
                    new_worker = PMMWorker(cfg, bot_manager.gateway)
                    bot_manager.workers[sym] = new_worker
                    if was_running and cfg.enabled:
                        await new_worker.start()

                ui.notify(f"Đã lưu thành công cấu hình cho {sym}!", type="positive")
                refresh_saved_list()
                if on_configs_updated:
                    on_configs_updated()

            def apply_template(template_name: str):
                if template_name == "SOL":
                    symbol_input.value = "SOL/USDT:USDT"
                    leverage_input.value = 5
                    order_amt_input.value = 50.0
                    bid_spread_input.value = 0.003
                    ask_spread_input.value = 0.003
                    max_long_input.value = 300.0
                    max_short_input.value = 300.0
                    gross_cap_input.value = 450.0
                    tp_input.value = 0.008
                    sl_input.value = 0.02
                elif template_name == "BTC":
                    symbol_input.value = "BTC/USDT:USDT"
                    leverage_input.value = 10
                    order_amt_input.value = 100.0
                    bid_spread_input.value = 0.0015
                    ask_spread_input.value = 0.0015
                    max_long_input.value = 600.0
                    max_short_input.value = 600.0
                    gross_cap_input.value = 900.0
                    tp_input.value = 0.005
                    sl_input.value = 0.015
                elif template_name == "ETH":
                    symbol_input.value = "ETH/USDT:USDT"
                    leverage_input.value = 10
                    order_amt_input.value = 80.0
                    bid_spread_input.value = 0.002
                    ask_spread_input.value = 0.002
                    max_long_input.value = 500.0
                    max_short_input.value = 500.0
                    gross_cap_input.value = 750.0
                    tp_input.value = 0.006
                    sl_input.value = 0.018

            with ui.row().classes("w-full justify-between items-center"):
                with ui.row().classes("gap-2 items-center"):
                    ui.label("Quick Presets:").classes("text-slate-400 text-sm")
                    ui.button("SOL Preset", on_click=lambda: apply_template("SOL")).props("outline size=sm color=primary")
                    ui.button("BTC Preset", on_click=lambda: apply_template("BTC")).props("outline size=sm color=amber")
                    ui.button("ETH Preset", on_click=lambda: apply_template("ETH")).props("outline size=sm color=cyan")

                ui.button("💾 Lưu Cấu Hình Cặp", on_click=save_pair).props("color=positive").classes("font-bold")
