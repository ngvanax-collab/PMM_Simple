"""Unit Tests for Event-Driven Requote Triggers & Hanging Orders."""
import time
from unittest.mock import MagicMock
import pytest

from app.core.worker import PMMWorker
from app.models.config import PairConfig
from app.models.state import OrderPurpose, OrderRecord, OrderSide, OrderStatus, OrderType, PositionSide


@pytest.fixture
def mock_worker():
    cfg = PairConfig(
        symbol="SOL/USDT:USDT",
        requote_threshold_pct=0.002,  # 0.2%
        order_refresh_time=30,
        hanging_orders_enabled=True,
        hanging_orders_cancel_pct=0.015,  # 1.5%
        ping_pong_enabled=True,
    )
    mock_gw = MagicMock()
    mock_gw.get_market_precision.return_value = (2, 3, 0.01, 0.001)

    worker = PMMWorker(cfg, mock_gw)
    worker.market_state.best_bid = 99.8
    worker.market_state.best_ask = 100.2
    worker.market_state.smoothed_mid = 100.0
    worker._running = True
    worker._paused = False
    return worker


def test_requote_trigger_on_mid_move(mock_worker):
    """Test requote triggers when mid price moves > requote_threshold_pct after min lifespan."""
    mock_worker.market_state.smoothed_mid = 100.0
    mock_worker._last_quote_mid = 100.0
    mock_worker._last_quote_time = time.time() - 10.0  # 10s ago (> 5s min lifespan)
    mock_worker._active_quote_orders = {
        "ord_1": OrderRecord(
            id="ord_1",
            client_order_id="q_1",
            symbol="SOL/USDT:USDT",
            side=OrderSide.BUY,
            position_side=PositionSide.LONG,
            order_type=OrderType.LIMIT_MAKER,
            price=99.7,
            amount=1.0,
            remaining_amount=1.0,
            status=OrderStatus.NEW,
            purpose=OrderPurpose.ENTRY_QUOTE,
            created_at=time.time() - 10.0,
            updated_at=time.time() - 10.0,
        )
    }

    # No price move -> No requote
    should_req, _ = mock_worker._check_should_requote()
    assert should_req is False

    # Price moves to 100.30 (> 0.2% = 0.002) -> Should requote
    mock_worker.market_state.smoothed_mid = 100.30
    should_req, reason = mock_worker._check_should_requote()
    assert should_req is True
    assert "Mid moved" in reason


def test_min_quote_lifespan_throttles_rapid_cancellations(mock_worker):
    """Verify that minor mid moves within min_quote_lifespan_sec do NOT cancel/requote."""
    mock_worker.market_state.smoothed_mid = 100.0
    mock_worker._last_quote_mid = 100.0
    mock_worker._last_quote_time = time.time() - 1.0  # Only 1 second ago (< 5s min lifespan)
    mock_worker._active_quote_orders = {
        "ord_1": OrderRecord(
            id="ord_1",
            client_order_id="q_1",
            symbol="SOL/USDT:USDT",
            side=OrderSide.BUY,
            position_side=PositionSide.LONG,
            order_type=OrderType.LIMIT_MAKER,
            price=99.7,
            amount=1.0,
            remaining_amount=1.0,
            status=OrderStatus.NEW,
            purpose=OrderPurpose.ENTRY_QUOTE,
            created_at=time.time() - 1.0,
            updated_at=time.time() - 1.0,
        )
    }

    # Small mid move (0.3% > 0.2% threshold, but < 2.5x threshold): Throttled to prevent fast cancelling
    mock_worker.market_state.smoothed_mid = 100.30
    should_req, reason = mock_worker._check_should_requote()
    assert should_req is False

    # Large sudden mid move (0.6% >= 2.5x threshold): Bypasses throttle for safety
    mock_worker.market_state.smoothed_mid = 100.60
    should_req, reason = mock_worker._check_should_requote()
    assert should_req is True


def test_requote_trigger_on_timeout(mock_worker):
    """Test requote triggers on timeout fallback (order_refresh_time)."""
    mock_worker.market_state.smoothed_mid = 100.0
    mock_worker._last_quote_mid = 100.0
    mock_worker._last_quote_time = time.time() - 35.0  # 35s ago (timeout is 30s)
    mock_worker._active_quote_orders = {
        "ord_1": OrderRecord(
            id="ord_1",
            client_order_id="q_1",
            symbol="SOL/USDT:USDT",
            side=OrderSide.BUY,
            position_side=PositionSide.LONG,
            order_type=OrderType.LIMIT_MAKER,
            price=99.7,
            amount=1.0,
            remaining_amount=1.0,
            status=OrderStatus.NEW,
            purpose=OrderPurpose.ENTRY_QUOTE,
            created_at=time.time(),
            updated_at=time.time(),
        )
    }

    should_req, reason = mock_worker._check_should_requote()
    assert should_req is True
    assert "Timeout fallback" in reason


def test_requote_trigger_on_hanging_order_distance(mock_worker):
    """Test requote triggers when a hanging order exceeds hanging_orders_cancel_pct."""
    mock_worker.market_state.smoothed_mid = 100.0
    mock_worker._last_quote_mid = 100.0
    mock_worker._last_quote_time = time.time()

    # Active bid order at 98.0 (distance = (100 - 98) / 100 = 2% >= 1.5% cancel pct)
    mock_order = OrderRecord(
        id="ord_hanging_1",
        client_order_id="q_bid_0_1",
        symbol="SOL/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        order_type=OrderType.LIMIT_MAKER,
        price=98.0,
        amount=1.0,
        remaining_amount=1.0,
        status=OrderStatus.NEW,
        purpose=OrderPurpose.ENTRY_QUOTE,
        created_at=time.time(),
        updated_at=time.time(),
    )
    mock_worker._active_quote_orders = {"ord_hanging_1": mock_order}

    should_req, reason = mock_worker._check_should_requote()
    assert should_req is True
    assert "Hanging order distance" in reason


def test_ping_pong_per_side_quoting(mock_worker):
    """Verify ping-pong locks ONLY the side with active position, leaving other side quoting."""
    # Give LONG a position under gross cap (50 USDT < 105 USDT)
    mock_worker.tracker.long_pos.amount = 0.5
    mock_worker.tracker.long_pos.entry_price = 100.0
    mock_worker.tracker.long_pos.notional = 50.0

    # SHORT has 0 position
    mock_worker.tracker.short_pos.amount = 0.0
    mock_worker.tracker.short_pos.notional = 0.0

    bids, asks = mock_worker.quoter.calculate_quotes(
        smoothed_mid=100.0,
        long_value_usdt=50.0,
        short_value_usdt=0.0,
        pause_long_entry=True,  # Ping-pong paused LONG side
        pause_short_entry=False,
    )

    # LONG entry quotes (Bids) must be empty
    assert len(bids) == 0, "Ping-pong must pause BID side when holding LONG"
    # SHORT entry quotes (Asks) must still be active!
    assert len(asks) > 0, "Ping-pong must NOT lock ASK side when holding LONG"
