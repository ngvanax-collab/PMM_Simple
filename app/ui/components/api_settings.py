"""API Key & Exchange Settings UI Component."""
import asyncio
from nicegui import ui
from app.core.manager import bot_manager
from app.models.config import ExchangeCredentials
from app.persistence.store import credential_store


def render_api_settings():
    """Render Exchange & API Credentials configuration tab."""
    with ui.card().classes("w-full max-w-4xl p-6 mx-auto bg-slate-900 border border-slate-800 rounded-xl shadow-2xl"):
        ui.label("🔑 Exchange & API Settings").classes("text-2xl font-bold text-white mb-2")
        ui.label(
            "Configure exchange API credentials. Your keys will be securely encrypted at rest using AES-256. Hedge Mode is automatically verified upon connection."
        ).classes("text-slate-400 text-sm mb-6")

        with ui.grid(columns=2).classes("w-full gap-4 mb-4"):
            exchange_select = ui.select(
                options=["binance", "bybit"],
                value="binance",
                label="Target Exchange"
            ).classes("w-full")

            testnet_switch = ui.switch("Use Testnet / Demo Trading", value=False).classes("mt-4 text-white")

        api_key_input = ui.input(
            label="API Key",
            placeholder="Enter Binance Futures or Bybit API Key",
            password=True,
            password_toggle_button=True
        ).classes("w-full mb-3")

        api_secret_input = ui.input(
            label="API Secret",
            placeholder="Enter API Secret",
            password=True,
            password_toggle_button=True
        ).classes("w-full mb-3")

        passphrase_input = ui.input(
            label="API Passphrase (Optional, Bybit V5)",
            placeholder="Enter Passphrase if required",
            password=True,
            password_toggle_button=True
        ).classes("w-full mb-6")

        status_container = ui.row().classes("w-full items-center justify-between")

        with status_container:
            status_label = ui.label("Status: Ready to configure").classes("text-sm text-slate-400")

            async def load_existing():
                creds = await credential_store.load_credentials(exchange_select.value)
                if creds:
                    api_key_input.value = creds.api_key
                    api_secret_input.value = creds.api_secret
                    testnet_switch.value = creds.testnet
                    if creds.passphrase:
                        passphrase_input.value = creds.passphrase
                    status_label.text = f"Status: Loaded saved {creds.exchange} credentials"
                    status_label.classes("text-emerald-400")

            async def save_and_connect():
                if not api_key_input.value or not api_secret_input.value:
                    ui.notify("Please enter both API Key and Secret!", type="warning")
                    return

                status_label.text = "Connecting & Verifying Hedge Mode..."
                status_label.classes("text-yellow-400")

                creds = ExchangeCredentials(
                    exchange=exchange_select.value,
                    api_key=api_key_input.value.strip(),
                    api_secret=api_secret_input.value.strip(),
                    passphrase=passphrase_input.value.strip() if passphrase_input.value else None,
                    testnet=testnet_switch.value,
                )

                # Save encrypted credentials
                await credential_store.save_credentials(creds)

                # Initialize Gateway & Verify Hedge Mode
                success = await bot_manager.start_gateway(creds)
                if success:
                    status_label.text = "CONNECTED: Hedge Mode Verified & Active!"
                    status_label.classes("text-emerald-400 font-bold")
                    ui.notify("Successfully connected to exchange in Hedge Mode!", type="positive")
                else:
                    status_label.text = "ERROR: Failed to connect or Hedge Mode rejected by exchange!"
                    status_label.classes("text-rose-400 font-bold")
                    ui.notify("Connection failed! Check API permissions or ensure no One-Way positions exist.", type="negative")

            with ui.row().classes("gap-3"):
                ui.button("Load Saved", on_click=load_existing).props("outline color=grey")
                ui.button("Test & Connect (Verify Hedge Mode)", on_click=save_and_connect).props("color=primary")

        # Initial auto-load
        ui.timer(0.5, load_existing, once=True)
