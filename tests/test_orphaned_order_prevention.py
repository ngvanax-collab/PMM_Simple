"""Unit Tests for Orphaned Order Prevention (TASK RO-1)."""
import time
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.core.worker import PMMWorker
from app.models.config import PairConfig
from app.models.state import (
    OrderPurpose,
    OrderRecord,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
)


@pytest.fixture
def mock_worker():
    cfg = PairConfig(
        symbol="AAVE/USDT:USDT",
        requote_threshold_pct=0.002,
        order_refresh_time=30,
        order_levels=1,
    )
    mock_gw = MagicMock()
    mock_gw.get_market_precision.return_value = (2, 2, 0.01, 0.01)
    mock_gw.fetch_free_balance = AsyncMock(return_value=500.0)
    mock_gw.create_quote_order = AsyncMock(return_value={"id": "new_ord_1"})
    mock_gw.cancel_order = AsyncMock(return_value=True)
    mock_gw.fetch_open_orders = AsyncMock(return_value=[])

    worker = PMMWorker(cfg, mock_gw)
    worker.market_state.best_bid = 129.50
    worker.market_state.best_ask = 129.60
    worker.market_state.smoothed_mid = 129.55
    worker._running = True
    worker._paused = False
    worker.tracker.reconcile_with_exchange = AsyncMock(return_value=True)
    return worker


@pytest.mark.asyncio
async def test_cancel_failure_does_not_remove_from_tracking(mock_worker):
    """When gateway.cancel_order returns False, order must remain in _active_quote_orders."""
    # Existing order drifted from target
    old_ord = OrderRecord(
        id="drifted_ord_1",
        client_order_id="q_sell_0_123",
        symbol="AAVE/USDT:USDT",
        side=OrderSide.SELL,
        position_side=PositionSide.SHORT,
        order_type=OrderType.LIMIT_MAKER,
        price=135.0,  # Far away from mid
        amount=0.4,
        remaining_amount=0.4,
        status=OrderStatus.NEW,
        purpose=OrderPurpose.ENTRY_QUOTE,
        created_at=time.time() - 20.0,
        updated_at=time.time() - 20.0,
    )
    mock_worker._active_quote_orders = {"drifted_ord_1": old_ord}

    # Mock cancel_order failure (e.g. -2015 REST error) - order remains open on exchange
    mock_worker.gateway.cancel_order = AsyncMock(return_value=False)
    mock_worker.gateway.fetch_open_orders = AsyncMock(return_value=[old_ord])

    await mock_worker._requote()

    # Crucial assertion: drifted_ord_1 is NOT removed from _active_quote_orders
    assert "drifted_ord_1" in mock_worker._active_quote_orders
    assert mock_worker._active_quote_orders["drifted_ord_1"] == old_ord
    assert getattr(mock_worker, "_pending_cancel_failures", 0) >= 1


@pytest.mark.asyncio
async def test_cancel_failure_triggers_reconcile(mock_worker):
    """When cancel_order fails, emergency reconciliation background tasks must be scheduled."""
    old_ord = OrderRecord(
        id="drifted_ord_2",
        client_order_id="q_sell_0_456",
        symbol="AAVE/USDT:USDT",
        side=OrderSide.SELL,
        position_side=PositionSide.SHORT,
        order_type=OrderType.LIMIT_MAKER,
        price=135.0,
        amount=0.4,
        remaining_amount=0.4,
        status=OrderStatus.NEW,
        purpose=OrderPurpose.ENTRY_QUOTE,
        created_at=time.time() - 20.0,
        updated_at=time.time() - 20.0,
    )
    mock_worker._active_quote_orders = {"drifted_ord_2": old_ord}
    mock_worker.gateway.cancel_order = AsyncMock(return_value=False)

    bg_tasks = []
    def capture_bg(coro):
        bg_tasks.append(coro)
    mock_worker._create_background_task = MagicMock(side_effect=capture_bg)

    await mock_worker._requote()

    # Must schedule reconcile_with_exchange and reconcile_open_orders
    assert mock_worker._create_background_task.call_count >= 2


@pytest.mark.asyncio
async def test_cancel_success_removes_from_tracking(mock_worker):
    """When cancel_order succeeds (returns True), order is normally popped."""
    old_ord = OrderRecord(
        id="drifted_ord_3",
        client_order_id="q_sell_0_789",
        symbol="AAVE/USDT:USDT",
        side=OrderSide.SELL,
        position_side=PositionSide.SHORT,
        order_type=OrderType.LIMIT_MAKER,
        price=135.0,
        amount=0.4,
        remaining_amount=0.4,
        status=OrderStatus.NEW,
        purpose=OrderPurpose.ENTRY_QUOTE,
        created_at=time.time() - 20.0,
        updated_at=time.time() - 20.0,
    )
    mock_worker._active_quote_orders = {"drifted_ord_3": old_ord}
    mock_worker.gateway.cancel_order = AsyncMock(return_value=True)

    await mock_worker._requote()

    # Successfully cancelled order must be removed
    assert "drifted_ord_3" not in mock_worker._active_quote_orders


@pytest.mark.asyncio
async def test_reconcile_open_orders_syncs_exchange_ground_truth(mock_worker):
    """reconcile_open_orders pulls active quotes from exchange and updates tracking."""
    # Stale local state has ord_ghost
    ghost_ord = OrderRecord(
        id="ord_ghost",
        client_order_id="q_buy_0_999",
        symbol="AAVE/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        order_type=OrderType.LIMIT_MAKER,
        price=125.0,
        amount=0.4,
        remaining_amount=0.4,
        status=OrderStatus.NEW,
        purpose=OrderPurpose.ENTRY_QUOTE,
        created_at=time.time() - 100.0,
        updated_at=time.time() - 100.0,
    )
    mock_worker._active_quote_orders = {"ord_ghost": ghost_ord}

    # Exchange actually has ord_real_1 (entry quote) and ord_tp (tp exit, not quote)
    real_quote = OrderRecord(
        id="ord_real_1",
        client_order_id="q_buy_0_1788560000",
        exchange_order_id="ord_real_1",
        symbol="AAVE/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        order_type=OrderType.LIMIT_MAKER,
        price=129.40,
        amount=0.4,
        remaining_amount=0.4,
        status=OrderStatus.NEW,
        purpose=OrderPurpose.ENTRY_QUOTE,
        created_at=time.time(),
        updated_at=time.time(),
    )
    real_tp = OrderRecord(
        id="ord_tp_1",
        client_order_id="tp_long_0_1788560000",
        exchange_order_id="ord_tp_1",
        symbol="AAVE/USDT:USDT",
        side=OrderSide.SELL,
        position_side=PositionSide.LONG,
        order_type=OrderType.LIMIT_MAKER,
        price=131.00,
        amount=0.2,
        remaining_amount=0.2,
        status=OrderStatus.NEW,
        purpose=OrderPurpose.TAKE_PROFIT,
        created_at=time.time(),
        updated_at=time.time(),
    )
    mock_worker.gateway.fetch_open_orders = AsyncMock(return_value=[real_quote, real_tp])

    await mock_worker.reconcile_open_orders()

    # Ghost order removed, real quote order synced, non-quote TP ignored
    assert "ord_ghost" not in mock_worker._active_quote_orders
    assert "ord_real_1" in mock_worker._active_quote_orders
    assert "ord_tp_1" not in mock_worker._active_quote_orders
