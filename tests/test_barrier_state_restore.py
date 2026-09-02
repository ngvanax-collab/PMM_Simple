"""Tests for Barrier State Persistence, DDL Migration, and Reconcile Hardening (TASK H-6)."""
import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
import aiosqlite
import pytest

from app.core.executor import TripleBarrierExecutor
from app.core.position_tracker import PositionTracker
from app.core.quoter import PMMQuoter
from app.models.config import PairConfig
from app.models.state import ExecutorBarrierState, PositionSide
from app.persistence.db import Database, db


@pytest.fixture(autouse=True)
def mock_db_ops():
    with patch("app.core.worker.db.save_fill", new_callable=AsyncMock), \
         patch("app.core.worker.db.record_pnl", new_callable=AsyncMock), \
         patch("app.core.worker.db.save_order", new_callable=AsyncMock):
        yield


@pytest.mark.asyncio
async def test_restart_preserves_guaranteed_sl():
    """
    Verify that upon bot restart:
    1. Saved Guaranteed SL state (is_guaranteed_sl_locked=1, sl_price=100.45, pyramid_filled_count=1)
       is restored from SQLite barrier_state table.
    2. Executor state reflects restored values rather than recalculating a looser baseline SL.
    """
    cfg = PairConfig(symbol="SOL/USDT:USDT", leverage=5, stop_loss=0.02)
    mock_gw = MagicMock()
    mock_gw.get_market_precision.return_value = (2, 3, 0.01, 0.001)
    mock_gw.get_market_limits.return_value = (0.0, 5.0)
    mock_gw.create_exit_order = AsyncMock(return_value={"id": "exit_1", "status": "closed"})
    mock_gw.cancel_order = AsyncMock()

    tracker = PositionTracker(cfg, mock_gw)
    tracker.long_pos.amount = 1.5
    tracker.long_pos.entry_price = 100.40

    quoter = PMMQuoter(
        config=cfg,
        price_precision=2,
        amount_precision=3,
        tick_size=0.01,
        step_size=0.001,
        min_amount=0.001,
        min_notional=5.0,
    )

    executor = TripleBarrierExecutor(
        config=cfg,
        position_side=PositionSide.LONG,
        gateway=mock_gw,
        position_tracker=tracker,
        quoter=quoter,
    )

    fake_saved_state = {
        "symbol": "SOL/USDT:USDT",
        "position_side": "LONG",
        "sl_price": 100.45,
        "is_guaranteed_sl_locked": 1,
        "pyramid_filled_count": 1,
        "peak_price": 101.20,
        "trough_price": 0.0,
        "trailing_tp_active": 0,
        "updated_at": 1000.0,
    }

    with patch.object(db, "load_barrier_state", new_callable=AsyncMock, return_value=fake_saved_state):
        await executor.reconcile_barrier(current_mark_price=101.0)

    assert executor.state.active is True
    assert executor.state.is_guaranteed_sl_locked is True
    assert executor.state.sl_price == 100.45
    assert executor.state.pyramid_filled_count == 1
    assert executor._trailing_cb_override is not None


@pytest.mark.asyncio
async def test_reconcile_barrier_alerts_on_zero_entry_price():
    """
    Fail-closed guardrail:
    If exchange reports open position (amount > 0) but entry_price <= 0:
    - Log CRITICAL
    - Do NOT arm barrier
    - Trigger isolated kill callback
    """
    cfg = PairConfig(symbol="SOL/USDT:USDT", leverage=5)
    mock_gw = MagicMock()
    mock_gw.get_market_precision.return_value = (2, 3, 0.01, 0.001)
    mock_gw.get_market_limits.return_value = (0.0, 5.0)

    tracker = PositionTracker(cfg, mock_gw)
    tracker.long_pos.amount = 1.0
    tracker.long_pos.entry_price = 0.0  # Invalid entry price!

    quoter = PMMQuoter(
        config=cfg,
        price_precision=2,
        amount_precision=3,
        tick_size=0.01,
        step_size=0.001,
        min_amount=0.001,
        min_notional=5.0,
    )

    mock_kill_cb = AsyncMock()
    executor = TripleBarrierExecutor(
        config=cfg,
        position_side=PositionSide.LONG,
        gateway=mock_gw,
        position_tracker=tracker,
        quoter=quoter,
        on_isolated_kill=mock_kill_cb,
    )

    await executor.reconcile_barrier(current_mark_price=100.0)

    # Barrier must NOT be armed
    assert executor.state.active is False
    # Isolated kill must have been triggered
    mock_kill_cb.assert_awaited_once()


@pytest.mark.asyncio
async def test_barrier_state_table_migrates_on_existing_db():
    """
    Verify SQLite schema DDL migration:
    - Pre-create database without barrier_state table.
    - Call Database.connect() -> barrier_state table is created idempotently.
    - Insert and load barrier state succeeds.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        # Pre-create old DB schema without barrier_state
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("CREATE TABLE old_table (id INTEGER PRIMARY KEY);")
            await conn.commit()

        # Connect using our Database class
        test_db = Database(db_path=db_path)
        await test_db.connect()

        # Check that barrier_state table exists
        async with test_db._lock:
            async with test_db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='barrier_state';"
            ) as cur:
                row = await cur.fetchone()
                assert row is not None
                assert row[0] == "barrier_state"

        # Test save and load
        test_state = ExecutorBarrierState(
            symbol="SOL/USDT:USDT",
            position_side=PositionSide.LONG,
            sl_price=100.55,
            is_guaranteed_sl_locked=True,
            pyramid_filled_count=1,
            peak_price=102.0,
        )
        await test_db.save_barrier_state(test_state)

        loaded = await test_db.load_barrier_state("SOL/USDT:USDT", PositionSide.LONG)
        assert loaded is not None
        assert loaded["sl_price"] == pytest.approx(100.55, abs=1e-4)
        assert loaded["is_guaranteed_sl_locked"] == 1
        assert loaded["pyramid_filled_count"] == 1

        # Test delete
        await test_db.delete_barrier_state("SOL/USDT:USDT", PositionSide.LONG)
        loaded_after = await test_db.load_barrier_state("SOL/USDT:USDT", PositionSide.LONG)
        assert loaded_after is None

        await test_db.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
