"""UI Component: Quantitative Screener Leaderboard & Dynamic Rebalancer Control."""
import asyncio
import time
from typing import Any, Dict, List
from nicegui import ui
from loguru import logger

from app.core.manager import bot_manager
from app.core.pair_rebalancer import rebalancer_service
from app.core.screener import screener_engine


def render_screener_tab():
    """Render Auto Pair Selector, Screener Leaderboard, and Rebalance Control Tab."""
    with ui.column().classes("w-full gap-6"):
        # ── Top Header & Telemetry Cards ──
        with ui.row().classes("w-full justify-between items-center"):
            with ui.column().classes("gap-1"):
                ui.label("🎯 Quantitative Screener & Dynamic Rebalancer").classes("text-xl font-bold text-white tracking-wide")
                ui.label(
                    "Scans USDT Perpetual Futures, evaluates Hurst Mean-Reversion, NATR Volatility & Depth with Pump-Dump Filters to optimize Top 5 Hedge Pairs."
                ).classes("text-xs text-slate-400")

            with ui.row().classes("gap-3 items-center"):
                async def trigger_manual_scan():
                    scan_btn.props("loading")
                    try:
                        ui.notify("Starting full market quantitative scan & rebalance...", type="info")
                        summary = await rebalancer_service.execute_rebalance_cycle(bot_manager)
                        added = len(summary.get("added_pairs", []))
                        draining = len(summary.get("draining_pairs", []))
                        retired = len(summary.get("retired_pairs", []))
                        retained = len(summary.get("retained_pairs", []))
                        ui.notify(
                            f"Scan complete! Retained: {retained}, Added: {added}, Draining: {draining}, Retired: {retired}",
                            type="positive",
                            duration=6.0
                        )
                    except Exception as e:
                        logger.error(f"Manual scan error: {e}")
                        ui.notify(f"Scan error: {e}", type="negative")
                    finally:
                        scan_btn.props(remove="loading")
                        refresh_screener_data()

                scan_btn = ui.button("🔍 SCAN & REBALANCE NOW", on_click=trigger_manual_scan).props(
                    "color=primary glossy size=sm"
                ).classes("font-bold px-4 shadow-lg")

        # ── Top Metric Cards Grid ──
        with ui.grid(columns=4).classes("w-full gap-4"):
            # Card 1: Screener Status
            with ui.card().classes("bg-slate-900 border border-slate-800 p-4 rounded-xl shadow-sm"):
                with ui.row().classes("items-center justify-between w-full mb-1"):
                    ui.label("SCREENER STATUS").classes("text-[11px] font-semibold text-slate-400 uppercase tracking-wider")
                    ui.icon("radar", size="xs").classes("text-indigo-400")
                status_label = ui.label("IDLE").classes("text-lg font-black text-white")
                status_sub = ui.label("Auto-scan enabled").classes("text-[11px] text-slate-500")

            # Card 2: Last Scan Time
            with ui.card().classes("bg-slate-900 border border-slate-800 p-4 rounded-xl shadow-sm"):
                with ui.row().classes("items-center justify-between w-full mb-1"):
                    ui.label("LAST SCAN TIME").classes("text-[11px] font-semibold text-slate-400 uppercase tracking-wider")
                    ui.icon("schedule", size="xs").classes("text-sky-400")
                last_scan_label = ui.label("Never").classes("text-lg font-black text-sky-400")
                total_cand_label = ui.label("0 candidates evaluated").classes("text-[11px] text-slate-500")

            # Card 3: Active Portfolio Slots
            with ui.card().classes("bg-slate-900 border border-slate-800 p-4 rounded-xl shadow-sm"):
                with ui.row().classes("items-center justify-between w-full mb-1"):
                    ui.label("ACTIVE PORTFOLIO SLOTS").classes("text-[11px] font-semibold text-slate-400 uppercase tracking-wider")
                    ui.icon("view_carousel", size="xs").classes("text-emerald-400")
                slots_label = ui.label("0 / 5 SLOTS").classes("text-lg font-black text-emerald-400")
                draining_sub = ui.label("0 draining workers").classes("text-[11px] text-slate-500")

            # Card 4: Total Portfolio Budget
            with ui.card().classes("bg-slate-900 border border-slate-800 p-4 rounded-xl shadow-sm"):
                with ui.row().classes("items-center justify-between w-full mb-1"):
                    ui.label("MARGIN BUDGET QUOTA").classes("text-[11px] font-semibold text-slate-400 uppercase tracking-wider")
                    ui.icon("account_balance_wallet", size="xs").classes("text-amber-400")
                budget_label = ui.label(f"${rebalancer_service.config.total_margin_budget_usdt:,.0f} USDT").classes("text-lg font-black text-amber-400")
                per_slot_label = ui.label(
                    f"${rebalancer_service.config.total_margin_budget_usdt / max(1, rebalancer_service.config.max_active_pairs):,.0f} / slot"
                ).classes("text-[11px] text-slate-500")

        # ── Middle Row: Rebalancer Control Config & Live Leaderboard ──
        with ui.row().classes("w-full gap-6 items-start"):
            # Left Column: Rebalancer Parameters
            with ui.card().classes("w-1/3 bg-slate-900 border border-slate-800 p-5 rounded-xl gap-4"):
                ui.label("⚙️ Screener & Rebalancer Controls").classes("text-base font-bold text-white mb-2")

                auto_rebalance_switch = ui.switch(
                    "Auto Rebalance Enabled",
                    value=rebalancer_service.config.enabled
                ).classes("text-sm text-slate-200")

                with ui.row().classes("w-full gap-3"):
                    max_pairs_input = ui.number(
                        label="Max Active Pairs",
                        value=rebalancer_service.config.max_active_pairs,
                        min=1, max=10, step=1
                    ).classes("flex-1 text-xs").props("outlined dense dark")

                    interval_input = ui.number(
                        label="Scan Interval (min)",
                        value=rebalancer_service.config.scan_interval_minutes,
                        min=5, max=1440, step=5
                    ).classes("flex-1 text-xs").props("outlined dense dark")

                # Quantitative Volatility & Anti-Pump Filter Inputs
                ui.label("🛡️ Volatility Ceiling & Pump Filters").classes("text-xs font-semibold text-sky-400 mt-1")
                with ui.row().classes("w-full gap-3"):
                    max_natr_input = ui.number(
                        label="Max NATR (%)",
                        value=screener_engine.config.max_natr_pct,
                        min=1.0, max=10.0, step=0.1
                    ).classes("flex-1 text-xs").props("outlined dense dark")

                    max_change_input = ui.number(
                        label="Max 24h Change (±%)",
                        value=screener_engine.config.max_24h_price_change_pct,
                        min=1.0, max=50.0, step=1.0
                    ).classes("flex-1 text-xs").props("outlined dense dark")

                with ui.row().classes("w-full gap-3"):
                    max_hurst_input = ui.number(
                        label="Max Hurst (H)",
                        value=screener_engine.config.max_hurst_exponent,
                        min=0.30, max=0.70, step=0.01
                    ).classes("flex-1 text-xs").props("outlined dense dark")

                    vol_ratio_input = ui.number(
                        label="Max Vol Ratio (σ7d/σ30d)",
                        value=screener_engine.config.max_volatility_ratio,
                        min=1.0, max=3.0, step=0.05
                    ).classes("flex-1 text-xs").props("outlined dense dark")

                with ui.row().classes("w-full gap-3 mt-1"):
                    delta_input = ui.number(
                        label="Score Delta Threshold (%)",
                        value=rebalancer_service.config.score_delta_threshold_pct * 100,
                        min=1.0, max=50.0, step=1.0
                    ).classes("flex-1 text-xs").props("outlined dense dark")

                    rank_buf_input = ui.number(
                        label="Rank Evict Threshold",
                        value=rebalancer_service.config.rank_threshold,
                        min=1, max=20, step=1
                    ).classes("flex-1 text-xs").props("outlined dense dark")

                total_budget_input = ui.number(
                    label="Total Portfolio Margin Budget (USDT)",
                    value=rebalancer_service.config.total_margin_budget_usdt,
                    min=100.0, step=50.0
                ).classes("w-full text-xs").props("outlined dense dark")

                async def save_rebalance_config():
                    rebalancer_service.config.enabled = bool(auto_rebalance_switch.value)
                    rebalancer_service.config.max_active_pairs = int(max_pairs_input.value or 5)
                    rebalancer_service.config.scan_interval_minutes = int(interval_input.value or 60)
                    rebalancer_service.config.score_delta_threshold_pct = float(delta_input.value or 10.0) / 100.0
                    rebalancer_service.config.rank_threshold = int(rank_buf_input.value or 7)
                    rebalancer_service.config.total_margin_budget_usdt = float(total_budget_input.value or 400.0)

                    # Update screener volatility barrier parameters
                    screener_engine.config.max_natr_pct = float(max_natr_input.value or 1.8)
                    screener_engine.config.max_24h_price_change_pct = float(max_change_input.value or 6.0)
                    screener_engine.config.max_hurst_exponent = float(max_hurst_input.value or 0.46)
                    screener_engine.config.max_volatility_ratio = float(vol_ratio_input.value or 1.25)

                    ui.notify("Screener & Rebalancer configuration updated successfully!", type="positive")
                    refresh_screener_data()

                ui.button("💾 SAVE CONFIGURATION", on_click=save_rebalance_config).props(
                    "color=positive outline size=sm"
                ).classes("w-full font-bold mt-2")

            # Right Column: Leaderboard Table
            with ui.card().classes("flex-1 bg-slate-900 border border-slate-800 p-5 rounded-xl gap-3"):
                with ui.row().classes("w-full justify-between items-center mb-1"):
                    ui.label("📊 Quantitative Screener Leaderboard").classes("text-base font-bold text-white")
                    ui.label("Filtered by Volatility Ceiling & Ranked by PMM Composite Score").classes("text-xs text-slate-500 italic")

                columns = [
                    {"name": "rank", "label": "Rank", "field": "rank", "align": "center", "sortable": True},
                    {"name": "symbol", "label": "Symbol", "field": "symbol", "align": "left", "sortable": True},
                    {"name": "pmm_score", "label": "PMM Score", "field": "pmm_score", "align": "center", "sortable": True},
                    {"name": "hurst", "label": "Hurst (H)", "field": "hurst", "align": "center", "sortable": True},
                    {"name": "natr", "label": "NATR 14 (%)", "field": "natr", "align": "center", "sortable": True},
                    {"name": "change_24h", "label": "24h Chg (%)", "field": "change_24h", "align": "center", "sortable": True},
                    {"name": "vol_ratio", "label": "Vol Ratio", "field": "vol_ratio", "align": "center", "sortable": True},
                    {"name": "vol_24h", "label": "24h Vol (USDT)", "field": "vol_24h", "align": "right", "sortable": True},
                    {"name": "funding", "label": "Funding Rate", "field": "funding", "align": "center", "sortable": True},
                    {"name": "status", "label": "Status", "field": "status", "align": "center"},
                ]

                leaderboard_table = ui.table(
                    columns=columns,
                    rows=[],
                    row_key="symbol",
                    pagination={"rowsPerPage": 10}
                ).classes("w-full text-xs text-slate-200")

        # ── Bottom Section: Rebalance Activity & Audit Log ──
        with ui.card().classes("w-full bg-slate-900 border border-slate-800 p-5 rounded-xl gap-3"):
            with ui.row().classes("w-full justify-between items-center mb-1"):
                ui.label("📜 Rebalance Activity & Transition Log").classes("text-base font-bold text-white")
                ui.label("Audit trail of promoted, draining, and retired pairs").classes("text-xs text-slate-500")

            event_columns = [
                {"name": "time", "label": "Time", "field": "time", "align": "left"},
                {"name": "action", "label": "Action", "field": "action", "align": "center"},
                {"name": "symbol", "label": "Symbol", "field": "symbol", "align": "left"},
                {"name": "score", "label": "Score", "field": "score", "align": "center"},
                {"name": "rank", "label": "Rank", "field": "rank", "align": "center"},
                {"name": "reason", "label": "Reason", "field": "reason", "align": "left"},
            ]

            events_table = ui.table(
                columns=event_columns,
                rows=[],
                row_key="time",
                pagination={"rowsPerPage": 8}
            ).classes("w-full text-xs text-slate-200")

    def refresh_screener_data():
        """Update reactive labels and table rows."""
        # Screener status card
        if screener_engine.is_scanning:
            status_label.text = "SCANNING..."
            status_label.classes(replace="text-lg font-black text-amber-400")
        elif rebalancer_service.config.enabled:
            status_label.text = "ACTIVE"
            status_label.classes(replace="text-lg font-black text-emerald-400")
        else:
            status_label.text = "PAUSED"
            status_label.classes(replace="text-lg font-black text-slate-400")

        # Last scan time
        if screener_engine.last_scan_time > 0:
            diff_min = int((time.time() - screener_engine.last_scan_time) / 60)
            last_scan_label.text = f"{diff_min}m ago" if diff_min > 0 else "Just now"
        else:
            last_scan_label.text = "Never"
        total_cand_label.text = f"{len(screener_engine.last_metrics)} candidate pairs evaluated"

        # Active Slots
        running_cnt = sum(
            1 for w in bot_manager.workers.values()
            if w.config.enabled and not getattr(w, "is_draining", False)
        )
        draining_cnt = sum(
            1 for w in bot_manager.workers.values()
            if getattr(w, "is_draining", False)
        )
        max_slots = rebalancer_service.config.max_active_pairs
        slots_label.text = f"{running_cnt} / {max_slots} SLOTS"
        draining_sub.text = f"{draining_cnt} draining workers"

        # Margin Budget
        budget_label.text = f"${rebalancer_service.config.total_margin_budget_usdt:,.0f} USDT"
        per_slot_label.text = f"${rebalancer_service.config.total_margin_budget_usdt / max(1, max_slots):,.0f} / slot"

        # Leaderboard Table Rows
        sorted_candidates = sorted(screener_engine.last_metrics.values(), key=lambda m: m.rank)
        rows = []
        for m in sorted_candidates[:25]:
            status_tag = "CANDIDATE"
            if m.symbol in bot_manager.workers:
                w = bot_manager.workers[m.symbol]
                if getattr(w, "is_draining", False):
                    status_tag = "DRAINING"
                elif w.config.enabled:
                    status_tag = "ACTIVE"

            # Formatting with warning highlights near filter thresholds
            hurst_str = f"⚠️ {m.hurst:.3f}" if m.hurst >= 0.44 else f"{m.hurst:.3f}"
            natr_str = f"⚠️ {m.natr_14:.2f}%" if m.natr_14 >= 2.0 else f"{m.natr_14:.2f}%"
            chg_str = f"⚠️ {m.price_change_24h:+.1f}%" if abs(m.price_change_24h) >= 8.0 else f"{m.price_change_24h:+.1f}%"

            rows.append({
                "rank": f"#{m.rank}",
                "symbol": m.symbol,
                "pmm_score": f"{m.pmm_score:.1f}",
                "hurst": hurst_str,
                "natr": natr_str,
                "change_24h": chg_str,
                "vol_ratio": f"{m.volatility_ratio:.2f}",
                "vol_24h": f"${m.volume_24h:,.0f}",
                "funding": f"{m.funding_rate*100:+.4f}%",
                "status": status_tag,
            })
        leaderboard_table.rows = rows

        # Events Table Rows
        ev_rows = []
        for e in reversed(rebalancer_service.events[-50:]):
            t_str = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
            ev_rows.append({
                "time": t_str,
                "action": e.action,
                "symbol": e.symbol,
                "score": f"{e.score:.1f}" if e.score > 0 else "-",
                "rank": f"#{e.rank}" if e.rank > 0 else "-",
                "reason": e.reason,
            })
        events_table.rows = ev_rows

    # Periodic UI Refresh (1 Hz)
    ui.timer(1.0, refresh_screener_data)
