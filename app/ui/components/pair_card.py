"""Pair Card UI Component with 2-Slot LONG / SHORT Sub-Panels, Remove Pair Action & Quick Config Accordion."""
from typing import Callable, Optional
from nicegui import ui
from loguru import logger

from app.core.manager import bot_manager
from app.core.worker import PMMWorker
from app.models.state import PositionSide
from app.persistence.store import config_store


class PairCard:
    """Renders real-time 2-slot LONG/SHORT status card with collapsible quick-edit form and remove action."""

    def __init__(self, symbol: str, on_remove: Optional[Callable[[], None]] = None):
        self.symbol = symbol
        self.on_remove = on_remove
        self.container = None
        self._build_card()

    def _build_card(self):
        worker: PMMWorker = bot_manager.workers.get(self.symbol)
        cfg = worker.config if worker else None

        with ui.card().classes("w-full bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg transition-all duration-300") as card:
            self.container = card

            # ── Header ──
            with ui.row().classes("w-full justify-between items-center pb-3 border-b border-slate-800"):
                with ui.row().classes("items-center gap-2"):
                    ui.label(self.symbol).classes("text-lg font-bold text-white")
                    if cfg:
                        self.lev_badge = ui.badge(f"{cfg.leverage}x {cfg.margin_mode.upper()}").props("color=slate-700")

                with ui.row().classes("items-center gap-2"):
                    self.status_badge = ui.badge("STOPPED").props("color=grey")

                    with ui.row().classes("gap-1 items-center"):
                        self.start_btn = ui.button(icon="play_arrow", on_click=self._toggle_start).props("round flat size=sm color=positive").tooltip("Start/Stop Worker")
                        self.pause_btn = ui.button(icon="pause", on_click=self._toggle_pause).props("round flat size=sm color=warning").tooltip("Pause/Resume Quoting")
                        self.remove_btn = ui.button(icon="close", on_click=self._confirm_remove).props("round flat size=xs color=grey").classes("hover:text-rose-400").tooltip("Ẩn / Gỡ cặp khỏi Dashboard")

            # ── Locked / Kill-Switch Warning Banner ──
            with ui.row().classes("w-full bg-rose-950/40 border border-rose-600 rounded-lg p-2.5 items-center justify-between mt-3 hidden") as locked_banner:
                self.locked_banner = locked_banner
                with ui.row().classes("items-center gap-2"):
                    ui.icon("warning", size="xs").classes("text-rose-400")
                    ui.label("🚨 KILLED: MAX DRAWDOWN / LOSS BREACHED").classes("text-xs font-black text-rose-400")
                ui.button("🔓 UNLOCK & RESUME WORKER", on_click=self._unlock_worker).props("size=xs color=warning glossy text-color=black").classes("font-bold")

            # ── 2 Slots: LONG and SHORT Grid ──
            with ui.grid(columns=2).classes("w-full gap-4 mt-3"):
                # ── LONG Slot ──
                with ui.column().classes("bg-emerald-950/20 border border-emerald-900/40 rounded-lg p-3"):
                    with ui.row().classes("w-full justify-between items-center mb-1"):
                        ui.label("🟢 LONG SLOT").classes("text-xs font-bold text-emerald-400 tracking-wider")
                        self.long_cd_badge = ui.badge("COOLDOWN").props("color=amber text-color=black size=xs").classes("hidden font-bold")

                    self.long_pos_label = ui.label("Pos: 0.0000 (0.00 USDT)").classes("text-sm text-slate-300")
                    self.long_entry_label = ui.label("Entry: $0.00 | Mark: $0.00").classes("text-xs text-slate-400")
                    self.long_pnl_label = ui.label("uPnL: $0.00 (0.00%)").classes("text-xs font-semibold text-slate-400")
                    self.long_barrier_label = ui.label("TP: None | SL: None").classes("text-xs text-slate-500 mt-1")

                # ── SHORT Slot ──
                with ui.column().classes("bg-rose-950/20 border border-rose-900/40 rounded-lg p-3"):
                    with ui.row().classes("w-full justify-between items-center mb-1"):
                        ui.label("🔴 SHORT SLOT").classes("text-xs font-bold text-rose-400 tracking-wider")
                        self.short_cd_badge = ui.badge("COOLDOWN").props("color=amber text-color=black size=xs").classes("hidden font-bold")

                    self.short_pos_label = ui.label("Pos: 0.0000 (0.00 USDT)").classes("text-sm text-slate-300")
                    self.short_entry_label = ui.label("Entry: $0.00 | Mark: $0.00").classes("text-xs text-slate-400")
                    self.short_pnl_label = ui.label("uPnL: $0.00 (0.00%)").classes("text-xs font-semibold text-slate-400")
                    self.short_barrier_label = ui.label("TP: None | SL: None").classes("text-xs text-slate-500 mt-1")

            # ── Session PnL & Drawdown Telemetry Footer ──
            with ui.row().classes("w-full justify-between items-center pt-2.5 mt-2 border-t border-slate-800 text-xs"):
                self.pnl_stats_label = ui.label("Session PnL: $0.00 USDT | Peak DD: $0.00 / $40.00 USDT").classes("text-slate-400 font-medium")

            # ── ⚙️ Collapsible Quick View / Edit Config Accordion ──
            if cfg:
                with ui.expansion("⚙️ Xem / Sửa Cấu hình Nhanh", icon="tune").classes("w-full bg-slate-950/40 border border-slate-800/80 rounded-lg text-xs mt-3 text-slate-300"):
                    with ui.column().classes("w-full p-2 gap-2"):
                        # Row 1: Đòn bẩy & Size
                        with ui.grid(columns=3).classes("w-full gap-2"):
                            quick_lev = ui.number("Leverage (x)", value=cfg.leverage, min=1, max=125, step=1).props("outlined dense dark")
                            quick_amt = ui.number("Order Amt ($)", value=cfg.order_amount_usdt, min=5.0, step=5.0).props("outlined dense dark")
                            quick_alloc = ui.number("Alloc Margin ($)", value=cfg.allocated_margin_usdt, min=0.0, step=10.0).props("outlined dense dark clearable")

                        # Row 2: Spreads & Tầng
                        with ui.grid(columns=3).classes("w-full gap-2"):
                            quick_bid_spd = ui.number("Bid Spread (%)", value=round(cfg.bid_spread * 100, 3), min=0.01, step=0.05).props("outlined dense dark")
                            quick_ask_spd = ui.number("Ask Spread (%)", value=round(cfg.ask_spread * 100, 3), min=0.01, step=0.05).props("outlined dense dark")
                            quick_levels = ui.number("Levels", value=cfg.order_levels, min=1, max=10, step=1).props("outlined dense dark")

                        # Row 3: Vốn & Rủi ro
                        with ui.grid(columns=3).classes("w-full gap-2"):
                            quick_max_l = ui.number("Max Long ($)", value=cfg.max_long_usdt, min=10.0, step=25.0).props("outlined dense dark")
                            quick_max_s = ui.number("Max Short ($)", value=cfg.max_short_usdt, min=10.0, step=25.0).props("outlined dense dark")
                            quick_gross = ui.number("Gross Cap ($)", value=cfg.gross_exposure_cap_usdt, min=10.0, step=50.0).props("outlined dense dark")

                        # Row 4: Exit & TP/SL
                        with ui.grid(columns=2).classes("w-full gap-2"):
                            quick_tp = ui.number("Take Profit (%)", value=round(cfg.take_profit * 100, 3), min=0.05, step=0.1).props("outlined dense dark")
                            quick_sl = ui.number("Stop Loss (%)", value=round(cfg.stop_loss * 100, 3), min=0.1, step=0.1).props("outlined dense dark")

                        # Save Button
                        async def save_quick_config():
                            try:
                                alloc_val = float(quick_alloc.value) if quick_alloc.value is not None and str(quick_alloc.value).strip() != "" and float(quick_alloc.value) > 0 else None
                                cfg.leverage = int(quick_lev.value or 5)
                                cfg.order_amount_usdt = float(quick_amt.value or 50.0)
                                cfg.allocated_margin_usdt = alloc_val
                                cfg.bid_spread = float(quick_bid_spd.value or 0.3) / 100.0
                                cfg.ask_spread = float(quick_ask_spd.value or 0.3) / 100.0
                                cfg.order_levels = int(quick_levels.value or 3)
                                cfg.max_long_usdt = float(quick_max_l.value or 300.0)
                                cfg.max_short_usdt = float(quick_max_s.value or 300.0)
                                cfg.gross_exposure_cap_usdt = float(quick_gross.value or 450.0)
                                cfg.take_profit = float(quick_tp.value or 0.8) / 100.0
                                cfg.stop_loss = float(quick_sl.value or 2.0) / 100.0
                                cfg.tp_levels = [[cfg.take_profit, 0.6], [cfg.take_profit * 1.8, 0.4]]

                                config_store.save_pair_config(cfg)

                                # Hot update worker quoter parameters
                                if worker:
                                    worker.quoter.config = cfg
                                    worker.config = cfg

                                if hasattr(self, "lev_badge"):
                                    self.lev_badge.text = f"{cfg.leverage}x {cfg.margin_mode.upper()}"

                                ui.notify(f"Đã lưu cấu hình nhanh cho {self.symbol} thành công!", type="positive")
                                self.update_state()
                            except Exception as e:
                                logger.error(f"Error saving quick config for {self.symbol}: {e}")
                                ui.notify(f"Lỗi lưu cấu hình: {e}", type="negative")

                        ui.button("💾 Lưu Cấu Hình Nhanh", on_click=save_quick_config).props("size=sm color=primary").classes("w-full mt-1 font-bold")

            self.update_state()

    def update_state(self):
        """Update live values from worker state."""
        worker: PMMWorker = bot_manager.workers.get(self.symbol)
        if not worker:
            return

        risk_stats = worker.get_risk_stats()
        is_locked = risk_stats.get("is_locked", False)

        # Status badge & buttons
        if is_locked:
            self.status_badge.text = "LOCKED"
            self.status_badge.props("color=negative")
            self.locked_banner.classes(remove="hidden")
            self.container.classes("border-rose-600 shadow-rose-950/50", remove="border-slate-800")
            self.start_btn.props("disable")
            self.pause_btn.props("disable")
        else:
            self.locked_banner.classes("hidden")
            self.container.classes("border-slate-800", remove="border-rose-600 shadow-rose-950/50")
            self.start_btn.props(remove="disable")
            self.pause_btn.props(remove="disable")

            if worker.is_running:
                if worker.is_paused:
                    self.status_badge.text = "PAUSED"
                    self.status_badge.props("color=warning")
                    self.pause_btn.props("icon=play_arrow color=positive")
                else:
                    self.status_badge.text = "RUNNING"
                    self.status_badge.props("color=positive")
                    self.pause_btn.props("icon=pause color=warning")
                self.start_btn.props("icon=stop color=negative")
            else:
                self.status_badge.text = "STOPPED"
                self.status_badge.props("color=grey")
                self.start_btn.props("icon=play_arrow color=positive")

        # Session PnL and Peak Drawdown Footer
        curr_pnl = risk_stats["current_pnl"]
        peak_dd = risk_stats["drawdown"]
        max_dd = risk_stats["max_drawdown_usdt"]
        pnl_color = "text-emerald-400" if curr_pnl >= 0 else "text-rose-400"
        self.pnl_stats_label.text = f"Session PnL: {curr_pnl:+.2f} USDT | Peak DD: {peak_dd:.2f} / {max_dd:.2f} USDT"
        self.pnl_stats_label.classes(f"font-semibold {pnl_color}")

        # LONG Slot
        long_s = worker.tracker.long_pos
        self.long_pos_label.text = f"Pos: {long_s.amount:.4f} ({long_s.notional:.2f} USDT)"
        self.long_entry_label.text = f"Entry: ${long_s.entry_price:.2f} | Mark: ${long_s.current_price:.2f}"

        pnl_pct = (long_s.unrealized_pnl / max(1.0, long_s.initial_margin) * 100) if long_s.initial_margin > 0 else 0.0
        self.long_pnl_label.text = f"uPnL: ${long_s.unrealized_pnl:+.2f} ({pnl_pct:+.2f}%)"
        if long_s.unrealized_pnl > 0.01:
            self.long_pnl_label.classes("text-emerald-400", remove="text-rose-400 text-slate-400")
        elif long_s.unrealized_pnl < -0.01:
            self.long_pnl_label.classes("text-rose-400", remove="text-emerald-400 text-slate-400")
        else:
            self.long_pnl_label.classes("text-slate-400", remove="text-emerald-400 text-rose-400")

        # LONG Barrier
        b_long = worker.executor_long.state
        tp_str = f"{len(b_long.tp_orders)} orders" if b_long.tp_orders else "None"
        sl_str = f"${b_long.sl_price:.2f}" if b_long.sl_order_id else "None"
        self.long_barrier_label.text = f"TP: {tp_str} | SL: {sl_str}"

        # Progressive Cooldown check on LONG
        cd_l_act, cd_l_lvl, cd_l_rem = worker.tracker.get_cooldown_info(PositionSide.LONG)
        if cd_l_act:
            self.long_cd_badge.text = f"⏳ Cooldown Lvl {cd_l_lvl} ({cd_l_rem}s)"
            self.long_cd_badge.classes(remove="hidden")
        else:
            self.long_cd_badge.classes("hidden")

        # SHORT Slot
        short_s = worker.tracker.short_pos
        self.short_pos_label.text = f"Pos: {short_s.amount:.4f} ({short_s.notional:.2f} USDT)"
        self.short_entry_label.text = f"Entry: ${short_s.entry_price:.2f} | Mark: ${short_s.current_price:.2f}"

        s_pnl_pct = (short_s.unrealized_pnl / max(1.0, short_s.initial_margin) * 100) if short_s.initial_margin > 0 else 0.0
        self.short_pnl_label.text = f"uPnL: ${short_s.unrealized_pnl:+.2f} ({s_pnl_pct:+.2f}%)"
        if short_s.unrealized_pnl > 0.01:
            self.short_pnl_label.classes("text-emerald-400", remove="text-rose-400 text-slate-400")
        elif short_s.unrealized_pnl < -0.01:
            self.short_pnl_label.classes("text-rose-400", remove="text-emerald-400 text-slate-400")
        else:
            self.short_pnl_label.classes("text-slate-400", remove="text-emerald-400 text-rose-400")

        # SHORT Barrier
        b_short = worker.executor_short.state
        s_tp_str = f"{len(b_short.tp_orders)} orders" if b_short.tp_orders else "None"
        s_sl_str = f"${b_short.sl_price:.2f}" if b_short.sl_order_id else "None"
        self.short_barrier_label.text = f"TP: {s_tp_str} | SL: {s_sl_str}"

        # Progressive Cooldown check on SHORT
        cd_s_act, cd_s_lvl, cd_s_rem = worker.tracker.get_cooldown_info(PositionSide.SHORT)
        if cd_s_act:
            self.short_cd_badge.text = f"⏳ Cooldown Lvl {cd_s_lvl} ({cd_s_rem}s)"
            self.short_cd_badge.classes(remove="hidden")
        else:
            self.short_cd_badge.classes("hidden")

    async def _toggle_start(self):
        worker = bot_manager.workers.get(self.symbol)
        if worker and worker.is_running:
            await bot_manager.stop_pair(self.symbol)
            ui.notify(f"Stopped {self.symbol}", type="info")
        else:
            success = await bot_manager.start_pair(self.symbol)
            if success:
                ui.notify(f"Started {self.symbol}", type="positive")
            else:
                ui.notify(f"Failed to start {self.symbol}! Check API connection, Hedge mode or Locked status.", type="negative")
        self.update_state()

    async def _toggle_pause(self):
        worker = bot_manager.workers.get(self.symbol)
        if worker:
            if worker.is_paused:
                await bot_manager.resume_pair(self.symbol)
                ui.notify(f"Resumed {self.symbol}", type="positive")
            else:
                await bot_manager.pause_pair(self.symbol)
                ui.notify(f"Paused {self.symbol}", type="warning")
        self.update_state()

    async def _unlock_worker(self):
        """Unlock and resume worker from isolated kill state."""
        ui.notify(f"Unlocking and resuming {self.symbol}...", type="info")
        success = await bot_manager.unlock_pair(self.symbol)
        if success:
            ui.notify(f"Worker {self.symbol} unlocked and restarted successfully!", type="positive")
        else:
            ui.notify(f"Failed to unlock {self.symbol}.", type="negative")
        self.update_state()

    def _confirm_remove(self):
        """Show confirmation dialog to hide / remove pair from dashboard."""
        worker = bot_manager.workers.get(self.symbol)
        if worker and worker.is_running:
            ui.notify(f"Vui lòng Stop cặp {self.symbol} trước khi gỡ/xóa khỏi hệ thống!", type="warning", duration=4.0)
            return

        with ui.dialog() as dialog, ui.card().classes("bg-slate-900 border border-slate-700 p-6 rounded-xl max-w-md"):
            ui.label("⚠️ QUẢN LÝ CẶP GIAO DỊCH").classes("text-lg font-bold text-amber-400 mb-2")
            ui.label(
                f"Bạn muốn thực hiện thao tác nào với cặp {self.symbol}?"
            ).classes("text-slate-300 text-sm mb-4")

            async def execute_hide():
                dialog.close()
                if worker:
                    worker.config.enabled = False
                    config_store.save_pair_config(worker.config)
                ui.notify(f"Đã ẩn cặp {self.symbol} khỏi bảng theo dõi.", type="info")
                if self.on_remove:
                    self.on_remove()

            async def execute_delete():
                dialog.close()
                deleted = await bot_manager.delete_pair(self.symbol)
                if deleted:
                    ui.notify(f"Đã xóa hoàn toàn cấu hình {self.symbol} khỏi bot!", type="positive")
                else:
                    ui.notify(f"Không tìm thấy file cấu hình {self.symbol} để xóa.", type="warning")
                if self.on_remove:
                    self.on_remove()

            with ui.column().classes("w-full gap-2 mb-2"):
                ui.button("👁️ Ẩn Khỏi Dashboard (Tắt)", on_click=execute_hide).props("outline color=warning").classes("w-full text-xs")
                ui.button("🗑️ Xóa Vĩnh Viễn Khỏi Bot", on_click=execute_delete).props("color=negative").classes("w-full text-xs font-bold")

            with ui.row().classes("w-full justify-end mt-2"):
                ui.button("Đóng", on_click=dialog.close).props("flat size=sm color=grey")

        dialog.open()

