"""Main NiceGUI Dashboard Layout and Assembly."""
import asyncio
from typing import Dict
from nicegui import ui
from loguru import logger

from app.core.manager import bot_manager
from app.ui.components.api_settings import render_api_settings
from app.ui.components.audit_pnl import render_audit_pnl
from app.ui.components.fr_arbitrage_tab import render_fr_arbitrage_tab
from app.ui.components.pair_card import PairCard
from app.ui.components.pair_manager import render_pair_manager
from app.ui.components.screener_tab import render_screener_tab


def create_dashboard():
    """Build NiceGUI Single-Page Application."""
    ui.query("body").classes("bg-slate-950 text-slate-100 font-sans")

    # Storage for active pair card instances
    pair_cards: Dict[str, PairCard] = {}

    # ── Top Navigation Bar ──
    with ui.header().classes("bg-slate-900 border-b border-slate-800 px-6 py-3 items-center justify-between shadow-md"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("candlestick_chart", size="md").classes("text-indigo-400")
            with ui.column().classes("gap-0"):
                ui.label("OpenPMM-Engine v3").classes("text-lg font-black tracking-wide text-white")
                ui.label("Futures Hedge Mode Native").classes("text-xs text-slate-400 font-medium")

        # Live telemetry pills
        with ui.row().classes("items-center gap-3"):
            conn_badge = ui.badge("DISCONNECTED").props("color=grey")
            cb_badge = ui.badge("HEALTHY").props("color=emerald")

            async def reset_circuit_breaker():
                bot_manager.circuit_breaker.reset()
                ui.notify("Circuit Breaker reset! You can now start pair workers.", type="positive", duration=5.0)
                refresh_ui()

            cb_reset_btn = ui.button("🔄 RESET CB", on_click=reset_circuit_breaker).props("size=xs color=warning outline").classes("hidden font-bold")
            worker_count_badge = ui.badge("0/0 BOTS ACTIVE").props("color=slate-700")

            # ── EMERGENCY KILL-ALL BUTTON (6-Phase Native Hedge) ──
            async def trigger_kill_all():
                with ui.dialog() as dialog, ui.card().classes("bg-slate-900 border border-rose-600 p-6 rounded-xl"):
                    ui.label("⚠️ CONFIRM EMERGENCY KILL-ALL").classes("text-xl font-bold text-rose-500 mb-2")
                    ui.label(
                        "This will immediately halt all quoting, cancel all open orders, and MARKET-CLOSE all open LONG and SHORT positions."
                    ).classes("text-slate-300 text-sm mb-4")

                    with ui.row().classes("w-full justify-end gap-3"):
                        ui.button("Cancel", on_click=dialog.close).props("flat color=grey")

                        async def confirm_execute():
                            dialog.close()
                            ui.notify("Executing 6-Phase Emergency Kill-All...", type="warning", duration=5.0)
                            report = await bot_manager.emergency_kill_all()
                            status = report.get("status", "UNKNOWN")
                            if status == "COMPLETED":
                                ui.notify("Emergency Kill-All COMPLETED! All positions 100% Flat.", type="positive", duration=8.0)
                            else:
                                ui.notify(f"Emergency Kill-All status: {status}", type="negative", duration=8.0)

                        ui.button("YES, KILL ALL NOW", on_click=confirm_execute).props("color=negative")
                dialog.open()

            ui.button("🚨 EMERGENCY KILL-ALL", on_click=trigger_kill_all).props("color=negative glossy").classes("font-bold tracking-wider px-4")

    # ── Tabs Container ──
    with ui.column().classes("w-full max-w-7xl mx-auto p-6 gap-6"):
        with ui.tabs().classes("w-full bg-slate-900 border border-slate-800 rounded-lg text-slate-400") as tabs:
            tab_dashboard = ui.tab("⚡ Live Grid Dashboard")
            tab_screener = ui.tab("🎯 Auto Screener & Rebalancer")
            tab_fr_arb = ui.tab("📈 Funding Rate Arbitrage")
            tab_settings = ui.tab("🔑 API & Exchange Settings")
            tab_pairs = ui.tab("⚙️ Pair Configurations")
            tab_audit = ui.tab("📊 Fills & PnL History")

        with ui.tab_panels(tabs, value=tab_dashboard).classes("w-full bg-transparent"):
            # ── Tab 1: Live Grid Dashboard ──
            with ui.tab_panel(tab_dashboard):
                with ui.row().classes("w-full justify-between items-center mb-4"):
                    ui.label("Active Hedge Pairs (2-Slot Realtime View)").classes("text-xl font-bold text-white")

                    with ui.row().classes("gap-2 items-center"):
                        async def start_all():
                            await bot_manager.start_all()
                            ui.notify("Starting all enabled pairs...", type="info")

                        async def stop_all():
                            await bot_manager.stop_all()
                            ui.notify("Stopping all pairs...", type="info")

                        ui.button("Start All Enabled", on_click=start_all).props("size=sm color=positive")
                        ui.button("Stop All", on_click=stop_all).props("size=sm color=grey")
                        ui.button("➕ Bật Lại / Thêm Cặp", on_click=lambda: open_enable_pair_dialog()).props("size=sm color=primary outline")

                grid_container = ui.grid(columns=2).classes("w-full gap-6")

                def open_enable_pair_dialog():
                    from app.persistence.store import config_store
                    from app.models.config import PairConfig
                    from app.core.worker import PMMWorker

                    all_configs = config_store.load_all_pair_configs()
                    with ui.dialog() as dialog, ui.card().classes("bg-slate-900 border border-slate-700 p-6 rounded-xl w-full max-w-lg"):
                        with ui.row().classes("w-full justify-between items-center mb-3"):
                            ui.label("➕ BẬT LẠI / THÊM CẶP GIAO DỊCH").classes("text-base font-bold text-white")
                            ui.button(icon="close", on_click=dialog.close).props("flat round dense size=xs color=grey")

                        ui.label("Danh sách các cặp đã lưu trong hệ thống:").classes("text-xs text-slate-400 mb-2")

                        if not all_configs:
                            ui.label("Chưa có cặp nào được lưu trong hệ thống.").classes("text-xs text-slate-500 italic py-2")

                        with ui.column().classes("w-full gap-2 mb-4 max-h-60 overflow-y-auto"):
                            for sym, cfg in all_configs.items():
                                is_en = cfg.enabled
                                with ui.row().classes("w-full justify-between items-center p-2.5 bg-slate-950/60 rounded-lg border border-slate-800"):
                                    with ui.column().classes("gap-0"):
                                        ui.label(sym).classes("text-sm font-bold text-white")
                                        ui.label(f"{cfg.leverage}x {cfg.margin_mode.upper()} | Base: ${cfg.order_amount_usdt}").classes("text-[11px] text-slate-500")

                                    async def toggle_pair_enabled(target_sym=sym, target_cfg=cfg):
                                        target_cfg.enabled = not target_cfg.enabled
                                        config_store.save_pair_config(target_cfg)
                                        if bot_manager.gateway:
                                            if target_sym not in bot_manager.workers:
                                                bot_manager.workers[target_sym] = PMMWorker(target_cfg, bot_manager.gateway)
                                            else:
                                                bot_manager.workers[target_sym].config = target_cfg
                                        rebuild_cards()
                                        dialog.close()
                                        ui.notify(f"Đã {'bật' if target_cfg.enabled else 'ẩn'} cặp {target_sym}!", type="positive")

                                    async def delete_pair_from_dialog(target_sym=sym):
                                        await bot_manager.delete_pair(target_sym)
                                        rebuild_cards()
                                        dialog.close()
                                        ui.notify(f"Đã xóa hoàn toàn cặp {target_sym} khỏi hệ thống!", type="positive")

                                    status_btn_text = "Ẩn Cặp" if is_en else "Bật Cặp"
                                    status_btn_color = "grey" if is_en else "positive"

                                    with ui.row().classes("items-center gap-1.5"):
                                        ui.button(status_btn_text, on_click=toggle_pair_enabled).props(f"size=xs color={status_btn_color}")
                                        ui.button(icon="delete", on_click=delete_pair_from_dialog).props("size=xs flat round color=negative").tooltip("Xóa vĩnh viễn cặp này")

                        # Quick Add New Symbol
                        ui.label("Hoặc kích hoạt cặp mới nhanh:").classes("text-xs text-slate-400 mb-1")
                        with ui.row().classes("w-full gap-2 items-center"):
                            new_sym_input = ui.input(placeholder="e.g. BTC/USDT:USDT").props("outlined dense dark").classes("flex-1 text-xs")

                            async def add_new_quick():
                                val = new_sym_input.value.strip().upper()
                                if not val:
                                    return
                                new_cfg = config_store.load_pair_config(val) or PairConfig(
                                    symbol=val,
                                    exchange="binance",
                                    enabled=True,
                                    leverage=5,
                                    margin_mode="isolated",
                                    order_amount_usdt=50.0,
                                )
                                new_cfg.enabled = True
                                config_store.save_pair_config(new_cfg)
                                if bot_manager.gateway:
                                    bot_manager.workers[val] = PMMWorker(new_cfg, bot_manager.gateway)
                                rebuild_cards()
                                dialog.close()
                                ui.notify(f"Đã kích hoạt cặp {val} thành công!", type="positive")

                            ui.button("Kích Hoạt", on_click=add_new_quick).props("size=sm color=primary")

                    dialog.open()

                def rebuild_cards():
                    grid_container.clear()
                    pair_cards.clear()
                    active_symbols = [sym for sym, w in bot_manager.workers.items() if w.config.enabled]
                    with grid_container:
                        if not active_symbols:
                            with ui.card().classes("col-span-2 bg-slate-900/50 border border-dashed border-slate-800 rounded-xl p-8 items-center justify-center"):
                                ui.icon("grid_off", size="lg").classes("text-slate-600 mb-2")
                                ui.label("Chưa có cặp giao dịch nào được hiển thị").classes("text-slate-400 font-semibold")
                                ui.label("Nhấn '➕ Bật Lại / Thêm Cặp' ở trên để chọn cặp giao dịch.").classes("text-xs text-slate-600 mb-3")
                                ui.button("➕ Thêm / Bật Cặp Giao Dịch", on_click=open_enable_pair_dialog).props("color=primary size=sm")
                        else:
                            for symbol in active_symbols:
                                pair_cards[symbol] = PairCard(symbol, on_remove=rebuild_cards)

                rebuild_cards()

            # ── Tab 2: Auto Screener & Rebalancer ──
            with ui.tab_panel(tab_screener):
                render_screener_tab()

            # ── Tab 3: Funding Rate Arbitrage ──
            with ui.tab_panel(tab_fr_arb):
                render_fr_arbitrage_tab()

            # ── Tab 4: API Settings ──
            with ui.tab_panel(tab_settings):
                render_api_settings()

            # ── Tab 5: Pair Configurations ──
            with ui.tab_panel(tab_pairs):
                render_pair_manager(on_configs_updated=rebuild_cards)

            # ── Tab 6: Fills & PnL ──
            with ui.tab_panel(tab_audit):
                render_audit_pnl()

    # ── UI Telemetry Periodic Refresh (1 Hz) ──
    def refresh_ui():
        # Update connection badge
        if bot_manager.gateway and bot_manager.gateway._is_connected:
            conn_badge.text = f"CONNECTED ({bot_manager.gateway.exchange_name.upper()})"
            conn_badge.props("color=positive")
        else:
            conn_badge.text = "DISCONNECTED"
            conn_badge.props("color=grey")

        # Update Circuit breaker badge & Reset button
        if bot_manager.circuit_breaker.is_tripped:
            reason = bot_manager.circuit_breaker.trip_reason
            short_reason = reason[:24] + "..." if len(reason) > 24 else reason
            cb_badge.text = f"CB TRIPPED: {short_reason}"
            cb_badge.props("color=negative")
            cb_reset_btn.classes(remove="hidden")
        else:
            cb_badge.text = "CIRCUIT BREAKER: OK"
            cb_badge.props("color=emerald")
            cb_reset_btn.classes("hidden")

        # Update Worker count
        active_cnt = sum(1 for w in bot_manager.workers.values() if w.is_running and not w.is_paused)
        worker_count_badge.text = f"{active_cnt}/{len(bot_manager.workers)} BOTS ACTIVE"

        # Refresh all pair cards
        for card in pair_cards.values():
            card.update_state()

    ui.timer(1.0, refresh_ui)
