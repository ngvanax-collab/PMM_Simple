"""Audit & Realtime PnL UI Component."""
import time
from nicegui import ui
from app.persistence.db import db


def render_audit_pnl():
    """Render PnL and Fills History Tab."""
    with ui.card().classes("w-full max-w-6xl mx-auto p-6 bg-slate-900 border border-slate-800 rounded-xl shadow-2xl"):
        ui.label("📊 Performance & Fills History").classes("text-2xl font-bold text-white mb-4")

        # ── KPI Cards ──
        with ui.grid(columns=3).classes("w-full gap-4 mb-6"):
            with ui.card().classes("bg-slate-800/60 p-4 border border-slate-700 rounded-lg"):
                ui.label("Total Realized PnL").classes("text-xs text-slate-400 font-semibold")
                pnl_label = ui.label("$0.00 USDT").classes("text-xl font-bold text-emerald-400 mt-1")

            with ui.card().classes("bg-slate-800/60 p-4 border border-slate-700 rounded-lg"):
                ui.label("Total Fees Paid").classes("text-xs text-slate-400 font-semibold")
                fee_label = ui.label("$0.00 USDT").classes("text-xl font-bold text-slate-300 mt-1")

            with ui.card().classes("bg-slate-800/60 p-4 border border-slate-700 rounded-lg"):
                ui.label("Net Realized PnL").classes("text-xs text-slate-400 font-semibold")
                net_label = ui.label("$0.00 USDT").classes("text-xl font-bold text-emerald-400 mt-1")

        # ── Fills Table ──
        ui.label("Recent Fills (Source of Truth)").classes("text-lg font-bold text-white mb-2")

        columns = [
            {"name": "time", "label": "Time", "field": "time", "align": "left"},
            {"name": "symbol", "label": "Symbol", "field": "symbol", "align": "left"},
            {"name": "side", "label": "Side", "field": "side", "align": "center"},
            {"name": "pos_side", "label": "Position Side", "field": "pos_side", "align": "center"},
            {"name": "price", "label": "Price", "field": "price", "align": "right"},
            {"name": "amount", "label": "Amount", "field": "amount", "align": "right"},
            {"name": "fee", "label": "Fee", "field": "fee", "align": "right"},
            {"name": "pnl", "label": "Realized PnL", "field": "pnl", "align": "right"},
        ]

        table = ui.table(columns=columns, rows=[], row_key="id").classes("w-full bg-slate-800/40 rounded-lg text-slate-200")
        table.add_slot(
            "body-cell-pnl",
            """
            <q-td :props="props" :class="props.row.pnl_raw > 0 ? 'text-emerald-400 font-bold' : (props.row.pnl_raw < 0 ? 'text-rose-400 font-bold' : 'text-slate-400')">
                {{ props.value }}
            </q-td>
            """
        )
        table.add_slot(
            "body-cell-side",
            """
            <q-td :props="props" :class="props.value === 'BUY' ? 'text-emerald-400 font-semibold' : 'text-rose-400 font-semibold'">
                {{ props.value }}
            </q-td>
            """
        )
        table.add_slot(
            "body-cell-pos_side",
            """
            <q-td :props="props" :class="props.value === 'LONG' ? 'text-emerald-300' : 'text-rose-300'">
                {{ props.value }}
            </q-td>
            """
        )

        async def refresh_data():
            summary = await db.get_pnl_summary()
            total_pnl = summary.get("total_realized_pnl", 0.0)
            total_fee = summary.get("total_fee", 0.0)
            net_pnl = summary.get("total_net_pnl", 0.0)

            pnl_label.text = f"${total_pnl:+.2f} USDT"
            fee_label.text = f"${total_fee:.2f} USDT"
            net_label.text = f"${net_pnl:+.2f} USDT"

            if total_pnl >= 0:
                pnl_label.classes("text-emerald-400", remove="text-rose-400")
            else:
                pnl_label.classes("text-rose-400", remove="text-emerald-400")

            if net_pnl >= 0:
                net_label.classes("text-emerald-400", remove="text-rose-400")
            else:
                net_label.classes("text-rose-400", remove="text-emerald-400")

            fills = await db.get_recent_fills(limit=30)
            rows = []
            for f in fills:
                time_str = time.strftime("%H:%M:%S", time.localtime(f.timestamp))
                rows.append({
                    "id": f.id,
                    "time": time_str,
                    "symbol": f.symbol,
                    "side": f.side.value if hasattr(f.side, 'value') else str(f.side),
                    "pos_side": f.position_side.value if hasattr(f.position_side, 'value') else str(f.position_side),
                    "price": f"${f.price:.2f}",
                    "amount": f"{f.amount:.4f}",
                    "fee": f"${f.fee:.4f}",
                    "pnl": f"${f.realized_pnl:+.2f}",
                    "pnl_raw": f.realized_pnl,
                })
            table.rows = rows

        ui.button("Refresh History", on_click=refresh_data).props("outline size=sm color=primary").classes("mt-4")
        ui.timer(5.0, refresh_data)
