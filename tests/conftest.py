"""Pytest configuration and global async cleanup fixtures."""
import asyncio
import pytest
from app.persistence.db import db


@pytest.fixture(autouse=True)
async def cleanup_db_and_tasks():
    """
    Ensure all DB connections and background tasks are cleanly closed
    before the test's event loop is closed, preventing PytestUnhandledThreadExceptionWarning.
    """
    yield

    # 1. Cancel and await any pending tasks on current event loop before closing DB
    loop = asyncio.get_running_loop()
    if not loop.is_closed():
        pending = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
        if pending:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    # 2. Close global DB connection if any test opened it
    if db._conn is not None:
        await db.close()
