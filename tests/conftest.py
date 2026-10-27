"""Pytest configuration and global async cleanup fixtures."""
import asyncio
import site
import sys
from pathlib import Path

# Add project root and .venv site-packages to sys.path
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

for sp in root_dir.glob(".venv/lib/python*/site-packages"):
    site.addsitedir(str(sp))

pytest_plugins = ["pytest_asyncio"]

import pytest
from app.persistence.db import db


@pytest.fixture(autouse=True)
def cleanup_db_and_tasks(request):
    """
    Ensure all DB connections and background tasks are cleanly closed
    before the test's event loop is closed, preventing PytestUnhandledThreadExceptionWarning.
    """
    yield

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            pending = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
            for t in pending:
                t.cancel()
            if getattr(db, "_conn", None) is not None:
                loop.create_task(db.close())
        elif not loop.is_closed():
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            if getattr(db, "_conn", None) is not None:
                loop.run_until_complete(db.close())
    except Exception:
        pass
