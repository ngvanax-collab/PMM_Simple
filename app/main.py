"""Main Application Entry Point combining FastAPI and NiceGUI on port 8502."""
import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from nicegui import app, ui
from loguru import logger

from app.api.router import router as api_router
from app.config import settings
from app.core.fr_execution import fr_manager
from app.core.manager import bot_manager
from app.ui.dashboard import create_dashboard


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Application startup and graceful shutdown management."""
    logger.info("Initializing OpenPMM-Engine v3 & Funding Rate Arbitrage Engine...")
    await bot_manager.initialize()
    await fr_manager.initialize()
    logger.info(f"OpenPMM-Engine started. Web UI ready on http://{settings.web_host}:{settings.web_port}")
    yield
    logger.info("Shutting down OpenPMM-Engine & FR Arbitrage Engine...")
    await fr_manager.shutdown()
    await bot_manager.shutdown()
    logger.info("Shutdown complete.")


# Register NiceGUI lifecycle hooks
app.include_router(api_router)

@app.on_startup
async def on_app_startup():
    logger.info("Initializing OpenPMM-Engine v3 & Funding Rate Arbitrage Engine...")
    await bot_manager.initialize()
    await fr_manager.initialize()
    logger.info(f"OpenPMM-Engine started. Web UI ready on http://{settings.web_host}:{settings.web_port}")

@app.on_shutdown
async def on_app_shutdown():
    logger.info("Shutting down OpenPMM-Engine & FR Arbitrage Engine...")
    await fr_manager.shutdown()
    await bot_manager.shutdown()
    logger.info("Shutdown complete.")


# Register NiceGUI Page
@ui.page("/")
def main_page():
    create_dashboard()


def handle_exit_signal(sig, frame):
    logger.warning(f"Received exit signal {sig}. Initiating graceful shutdown...")
    asyncio.create_task(fr_manager.shutdown())
    asyncio.create_task(bot_manager.shutdown())
    sys.exit(0)


def start_server():
    """Start the NiceGUI and FastAPI server."""
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)

    ui.run(
        title="OpenPMM-Engine v3 (Hedge Mode)",
        host=settings.web_host,
        port=settings.web_port,
        reload=False,
        show=False,
        dark=True,
    )


if __name__ in {"__main__", "__mp_main__"}:
    start_server()
