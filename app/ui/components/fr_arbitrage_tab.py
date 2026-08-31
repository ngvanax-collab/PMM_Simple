"""Funding Rate Arbitrage Management & Telemetry UI Tab for NiceGUI."""
import asyncio
import time
from typing import Any, Dict, List
from nicegui import ui
from loguru import logger

from app.core.fr_execution.manager import fr_manager
from app.core.fr_execution.models import FRRiskConfig


def render_fr_arbitrage_tab():
    """Render the full Funding Rate Arbitrage tab interface on NiceGUI."""

    # ── 1. Top Summary Banner (Metrics) ──
    with ui.grid(columns=4).classes("w-full gap-4 mb-6"):
        # Metric 1: Realized Funding PnL
        with ui.card().classes("bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm"):
            ui.label("Realized Funding PnL").classes("text-xs text-slate-400 font-medium")
            metric_funding_pnl = ui.label("+$0.0000").classes("text-2xl font-black text-emerald-400 tracking-tight")
            ui.label("Cumulative 8h/4h funding payments").classes("text-[10px] text-slate-500")

        # Metric 2: Net Arbitrage APR
        with ui.card().classes("bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm"):
            ui.label("Net Arbitrage APR").classes("text-xs text-slate-400 font-medium")
            metric_apr = ui.label("0.0%").classes("text-2xl font-black text-indigo-400 tracking-tight")
            ui.label("Annualized net spread yield").classes("text-[10px] text-slate-500")

        # Metric 3: Active Arb Pairs
        with ui.card().classes("bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm"):
            ui.label("Active Arb Pairs").classes("text-xs text-slate-400 font-medium")
            metric_active_pairs = ui.label("0 Pairs").classes("text-2xl font-black text-sky-400 tracking-tight")
            ui.label("Delta-neutral hedged positions").classes("text-[10px] text-slate-500")

        # Metric 4: Free Collateral Margins
        with ui.card().classes("bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm"):
            ui.label("Exchange Free Margins").classes("text-xs text-slate-400 font-medium")
            metric_margins = ui.label("Binance: $0 | Bybit: $0").classes("text-base font-bold text-slate-200 tracking-tight mt-1")
            ui.label("Available collateral for new legs").classes("text-[10px] text-slate-500")

    # ── 2. Active Dual-Leg Position Cards Grid ──
    with ui.column().classes("w-full gap-3 mb-6"):
        with ui.row().classes("w-full justify-between items-center"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("sync_alt", size="sm").classes("text-emerald-400")
                ui.label("Active Dual-Leg Arbitrage Pairs").classes("text-lg font-bold text-white")
            active_count_badge = ui.badge("0 ACTIVE").props("color=emerald")

        cards_container = ui.grid(columns=2).classes("w-full gap-4")

    # ── 3. Decision Feed View (Opportunities & Policies Table) ──
    with ui.column().classes("w-full bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm mb-6"):
        with ui.row().classes("w-full justify-between items-center mb-3"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("dynamic_feed", size="sm").classes("text-indigo-400")
                ui.label("Decision Feed (SIGNAL_CONTRACT_V2)").classes("text-base font-bold text-white")
                ui.label("Live stream from Decision Layer :8102").classes("text-xs text-slate-400")

            async def trigger_poll_now():
                ui.notify("Polling Decision Layer for fresh policies...", type="info")
                results = await fr_manager.engine.fetch_and_execute_policies()
                ui.notify(f"Polled {len(results)} policies.", type="positive" if results else "info")
                update_tab_ui()

            ui.button("⚡ Poll Policies Now", on_click=trigger_poll_now).props("size=sm color=indigo outline")

        feed_table = ui.table(
            columns=[
                {"name": "symbol", "label": "Symbol", "field": "symbol", "sortable": True, "align": "left"},
                {"name": "exchange_long", "label": "Long Exchange", "field": "exchange_long", "align": "center"},
                {"name": "exchange_short", "label": "Short Exchange", "field": "exchange_short", "align": "center"},
                {"name": "score_7d", "label": "7d Score", "field": "score_7d", "sortable": True, "align": "center"},
                {"name": "expected_net_edge_bps", "label": "Net Spread (bps)", "field": "expected_net_edge_bps", "sortable": True, "align": "right"},
                {"name": "action", "label": "Action", "field": "action", "sortable": True, "align": "center"},
                {"name": "target_notional_usdt", "label": "Target Size ($)", "field": "target_notional_usdt", "align": "right"},
            ],
            rows=[],
            row_key="symbol",
            pagination=10,
        ).classes("w-full bg-slate-950 text-slate-200 border border-slate-800 rounded-lg")

    # ── 4. Arbitrage Config & Risk Management Form ──
    with ui.column().classes("w-full bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm mb-6"):
        with ui.row().classes("items-center gap-2 mb-3"):
            ui.icon("security", size="sm").classes("text-amber-400")
            ui.label("Arbitrage Risk & Execution Parameters").classes("text-base font-bold text-white")

        with ui.grid(columns=3).classes("w-full gap-4"):
            leverage_input = ui.number(
                label="Max Leverage Cap (≤5x)",
                value=fr_manager.risk_config.max_leverage,
                min=1,
                max=10,
                step=1,
            ).props("outlined dark dense")

            margin_input = ui.number(
                label="Allocated Margin / Pair (USDT)",
                value=fr_manager.risk_config.allocated_margin_per_pair,
                min=10.0,
                step=10.0,
            ).props("outlined dark dense")

            edge_input = ui.number(
                label="Min Expected Edge (bps)",
                value=fr_manager.risk_config.min_expected_edge_bps,
                min=1.0,
                step=1.0,
            ).props("outlined dark dense")

            stop_loss_input = ui.number(
                label="Max Loss Stop-Loss ($)",
                value=fr_manager.risk_config.max_loss_usd,
                min=5.0,
                step=5.0,
            ).props("outlined dark dense")

            decision_url_input = ui.input(
                label="Decision Layer URL",
                value=fr_manager.risk_config.decision_layer_url,
            ).props("outlined dark dense")

            with ui.row().classes("items-center gap-3 mt-2"):
                auto_exec_switch = ui.switch("Auto Policy Execution", value=fr_manager.risk_config.auto_execution_enabled).props("color=positive")

        async def save_config():
            new_cfg = FRRiskConfig(
                max_leverage=int(leverage_input.value or 5),
                allocated_margin_per_pair=float(margin_input.value or 200.0),
                min_expected_edge_bps=float(edge_input.value or 15.0),
                max_loss_usd=float(stop_loss_input.value or 50.0),
                decision_layer_url=str(decision_url_input.value or "http://localhost:8102"),
                auto_execution_enabled=bool(auto_exec_switch.value),
            )
            fr_manager.save_risk_config(new_cfg)
            ui.notify("Funding Rate Arbitrage risk configuration saved!", type="positive")

        ui.button("💾 Save Risk Parameters", on_click=save_config).props("color=primary").classes("mt-4 self-end")

    # ── 5. ARBITRAGE EMERGENCY KILL-ALL ──
    with ui.card().classes("w-full bg-rose-950/20 border border-rose-900/50 rounded-xl p-5 shadow-md items-center justify-between"):
        with ui.row().classes("w-full justify-between items-center"):
            with ui.column().classes("gap-1"):
                ui.label("🚨 ARBITRAGE EMERGENCY KILL-ALL (6-Phase Dual-Exchange)").classes("text-lg font-black text-rose-400 tracking-wide")
                ui.label(
                    "Immediately trips global kill-switch, cancels all pending orders, and MARKET-CLOSES all open LONG and SHORT legs on Binance & Bybit."
                ).classes("text-xs text-rose-200/80")

            async def trigger_fr_kill_all():
                with ui.dialog() as dialog, ui.card().classes("bg-slate-900 border border-rose-600 p-6 rounded-xl"):
                    ui.label("⚠️ CONFIRM ARBITRAGE EMERGENCY KILL-ALL").classes("text-xl font-bold text-rose-500 mb-2")
                    ui.label(
                        "Are you sure you want to execute 6-Phase Kill-All across Binance Futures and Bybit Linear? All arbitrage positions will be market-closed immediately."
                    ).classes("text-slate-300 text-sm mb-4")

                    with ui.row().classes("w-full justify-end gap-3"):
                        ui.button("Cancel", on_click=dialog.close).props("flat color=grey")

                        async def confirm_kill_all():
                            dialog.close()
                            ui.notify("Executing 6-Phase Dual-Exchange Kill-All...", type="warning", duration=5.0)
                            report = await fr_manager.trigger_emergency_kill_all()
                            status = report.get("status", "UNKNOWN")
                            if status == "COMPLETED":
                                ui.notify("Arbitrage Kill-All COMPLETED! All exchange positions are 100% Flat.", type="positive", duration=8.0)
                            else:
                                ui.notify(f"Arbitrage Kill-All result: {status}", type="negative", duration=8.0)
                            update_tab_ui()

                        ui.button("YES, KILL ALL NOW", on_click=confirm_kill_all).props("color=negative")
                dialog.open()

            ui.button("🚨 ARBITRAGE EMERGENCY KILL-ALL", on_click=trigger_fr_kill_all).props("color=negative glossy size=md").classes("font-black px-6 tracking-wider")

    # ── Realtime Dynamic Refresh Routine ──
    async def update_tab_ui():
        # Update metrics banner
        metrics = await fr_manager.get_summary_metrics()
        
        pnl_val = metrics.total_realized_funding_pnl
        pnl_prefix = "+" if pnl_val >= 0 else ""
        metric_funding_pnl.text = f"{pnl_prefix}${pnl_val:.4f}"
        metric_funding_pnl.classes(replace="text-emerald-400" if pnl_val >= 0 else "text-rose-400")

        metric_apr.text = f"{metrics.net_arbitrage_apr:.1f}%"
        metric_active_pairs.text = f"{metrics.active_arb_pairs} Pairs"
        metric_margins.text = f"Binance: ${metrics.binance_free_margin:.1f} | Bybit: ${metrics.bybit_free_margin:.1f}"

        # Update Decision Feed table
        policies = fr_manager.get_recent_policies()
        feed_rows = []
        for p in policies:
            feed_rows.append({
                "symbol": p.symbol,
                "exchange_long": p.exchange_long.upper(),
                "exchange_short": p.exchange_short.upper(),
                "score_7d": f"{p.score_7d:.2f}" if p.score_7d is not None else "--",
                "expected_net_edge_bps": f"+{p.expected_net_edge_bps:.1f}" if p.expected_net_edge_bps > 0 else f"{p.expected_net_edge_bps:.1f}",
                "action": p.action.value,
                "target_notional_usdt": f"${p.target_notional_usdt:.1f}",
            })
        feed_table.rows = feed_rows

        # Update Dual Position Cards
        active_positions = fr_manager.get_active_positions()
        active_count_badge.text = f"{len(active_positions)} ACTIVE"
        
        cards_container.clear()
        with cards_container:
            if not active_positions:
                with ui.card().classes("col-span-2 bg-slate-900/50 border border-dashed border-slate-800 rounded-xl p-8 items-center justify-center"):
                    ui.icon("hourglass_empty", size="lg").classes("text-slate-600 mb-2")
                    ui.label("No Active Arbitrage Pairs").classes("text-slate-400 font-semibold")
                    ui.label("Positions opened by Decision Layer will appear here in real-time.").classes("text-xs text-slate-600")
            else:
                for pos in active_positions:
                    pos.recalculate()
                    _build_single_dual_card(pos, update_tab_ui)

    # Initial load and 2.0s periodic refresh
    ui.timer(2.0, update_tab_ui)
    asyncio.create_task(update_tab_ui())


def _build_single_dual_card(pos, refresh_callback):
    """Build a single 2-column dual-leg position card."""
    with ui.card().classes("w-full bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg"):
        # Header
        with ui.row().classes("w-full justify-between items-center pb-3 border-b border-slate-800"):
            with ui.row().classes("items-center gap-2"):
                ui.label(pos.symbol).classes("text-lg font-black text-white")
                status_color = "positive" if pos.status == "OPEN" else ("warning" if pos.status == "OPENING" else "grey")
                ui.badge(pos.status).props(f"color={status_color}")
                if pos.holding_duration_hours > 0:
                    ui.badge(f"⏱️ {pos.holding_duration_hours:.1f}h").props("color=slate-800 text-color=slate-300")

            with ui.row().classes("gap-2 items-center"):
                async def pause_pair():
                    is_paused = fr_manager.toggle_pause_symbol(pos.symbol)
                    ui.notify(f"Pair {pos.symbol} {'PAUSED' if is_paused else 'RESUMED'}", type="info")
                    await refresh_callback()

                pause_text = "▶️ Resume" if pos.is_paused else "⏸️ Pause"
                ui.button(pause_text, on_click=pause_pair).props("size=xs color=warning outline")

                async def close_pair():
                    ui.notify(f"Closing dual-leg position for {pos.symbol}...", type="warning")
                    res = await fr_manager.manual_close_pair(pos.symbol)
                    if res.get("status") in ("EXIT_COMPLETED", "ALREADY_FLAT"):
                        ui.notify(f"Pair {pos.symbol} successfully closed to Flat.", type="positive")
                    else:
                        ui.notify(f"Pair close response: {res.get('status')}", type="info")
                    await refresh_callback()

                ui.button("⚡ Close Pair", on_click=close_pair).props("size=xs color=negative outline")

        # 2-Column Dual Leg Grid
        with ui.grid(columns=2).classes("w-full gap-4 my-3"):
            # Left: LONG Leg
            with ui.column().classes("bg-emerald-950/20 border border-emerald-900/40 rounded-lg p-3"):
                with ui.row().classes("w-full justify-between items-center mb-1"):
                    ui.label("🟢 LONG LEG").classes("text-xs font-bold text-emerald-400 tracking-wider")
                    ui.badge(pos.long_leg.exchange.upper()).props("color=emerald size=xs")

                ui.label(f"Size: {pos.long_leg.size} (${pos.long_leg.notional:.2f})").classes("text-sm text-slate-200 font-semibold")
                ui.label(f"Entry: ${pos.long_leg.entry_price:.4f} | Mark: ${pos.long_leg.mark_price:.4f}").classes("text-xs text-slate-400")
                
                long_upnl_color = "text-emerald-400" if pos.long_leg.unrealized_pnl >= 0 else "text-rose-400"
                ui.label(f"uPnL: ${pos.long_leg.unrealized_pnl:+.4f}").classes(f"text-xs font-bold {long_upnl_color}")
                ui.label(f"Funding Accrued: ${pos.long_leg.funding_accrued:+.4f}").classes("text-xs text-sky-400")

            # Right: SHORT Leg
            with ui.column().classes("bg-rose-950/20 border border-rose-900/40 rounded-lg p-3"):
                with ui.row().classes("w-full justify-between items-center mb-1"):
                    ui.label("🔴 SHORT LEG").classes("text-xs font-bold text-rose-400 tracking-wider")
                    ui.badge(pos.short_leg.exchange.upper()).props("color=rose size=xs")

                ui.label(f"Size: {pos.short_leg.size} (${pos.short_leg.notional:.2f})").classes("text-sm text-slate-200 font-semibold")
                ui.label(f"Entry: ${pos.short_leg.entry_price:.4f} | Mark: ${pos.short_leg.mark_price:.4f}").classes("text-xs text-slate-400")
                
                short_upnl_color = "text-emerald-400" if pos.short_leg.unrealized_pnl >= 0 else "text-rose-400"
                ui.label(f"uPnL: ${pos.short_leg.unrealized_pnl:+.4f}").classes(f"text-xs font-bold {short_upnl_color}")
                ui.label(f"Funding Accrued: ${pos.short_leg.funding_accrued:+.4f}").classes("text-xs text-sky-400")

        # Footer
        net_color = "text-emerald-400" if pos.net_pnl >= 0 else "text-rose-400"
        with ui.row().classes("w-full justify-between items-center pt-2 border-t border-slate-800/80"):
            with ui.row().classes("items-center gap-3"):
                ui.label("Net Pair PnL:").classes("text-xs text-slate-400")
                ui.label(f"${pos.net_pnl:+.4f} USDT").classes(f"text-sm font-black {net_color}")
            ui.label(f"Total Funding: ${pos.total_funding_accrued:+.4f}").classes("text-xs font-semibold text-sky-400")
