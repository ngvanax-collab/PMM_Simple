"""Unit Tests for 6-Phase Emergency Kill-All (Hedge Mode Native)."""
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.core.manager import BotManager
from app.core.worker import PMMWorker
from app.models.config import PairConfig
from app.models.state import OrderPurpose, OrderSide, OrderType, PositionSide, SidePositionState
from app.persistence.db import db


@pytest.mark.asyncio
async def test_emergency_kill_all_six_phases():
    await db.connect()
    mgr = BotManager()

    # Mock gateway
    mock_gateway = MagicMock()
    mock_gateway._is_connected = True
    mock_gateway.cancel_all_symbol_orders = AsyncMock(return_value=True)
    mock_gateway.create_exit_order = AsyncMock(return_value={"id": "kill_order_1"})
    mock_gateway.get_market_precision.return_value = (2, 3, 0.01, 0.001)

    # Initial positions: LONG 5.0 and SHORT 3.0
    long_pos = SidePositionState(symbol="SOL/USDT:USDT", position_side=PositionSide.LONG, amount=5.0)
    short_pos = SidePositionState(symbol="SOL/USDT:USDT", position_side=PositionSide.SHORT, amount=3.0)

    # Re-fetch after kill returns 0.0
    flat_long = SidePositionState(symbol="SOL/USDT:USDT", position_side=PositionSide.LONG, amount=0.0)
    flat_short = SidePositionState(symbol="SOL/USDT:USDT", position_side=PositionSide.SHORT, amount=0.0)

    mock_gateway.fetch_positions_hedge = AsyncMock(
        side_effect=[
            (long_pos, short_pos),  # Phase 3 fetch
            (flat_long, flat_short),  # Phase 5 confirmation fetch
            (flat_long, flat_short),  # Final check
        ]
    )

    mgr.gateway = mock_gateway
    cfg = PairConfig(symbol="SOL/USDT:USDT")
    worker = PMMWorker(cfg, mock_gateway)
    mgr.workers["SOL/USDT:USDT"] = worker

    # Execute 6-Phase Kill-All
    report = await mgr.emergency_kill_all()

    # Assertions
    assert report["phase1_stopped_workers"] == 1
    assert report["phase2_cancelled_orders"] == 1
    assert len(report["phase3_positions_found"]) == 2  # 1 LONG and 1 SHORT found
    assert len(report["phase4_market_exits_placed"]) == 2
    assert report["phase5_confirmation"]["all_flat"] is True
    assert report["status"] == "COMPLETED"

    # Verify exit calls:
    # 1. Closing LONG 5.0 -> side=SELL, position_side=LONG
    # 2. Closing SHORT 3.0 -> side=BUY, position_side=SHORT
    calls = mock_gateway.create_exit_order.call_args_list
    assert len(calls) == 2

    call_long = next(c for c in calls if c.kwargs["position_side"] == PositionSide.LONG)
    assert call_long.kwargs["side"] == OrderSide.SELL
    assert call_long.kwargs["amount"] == 5.0
    assert call_long.kwargs["order_type"] == OrderType.MARKET

    call_short = next(c for c in calls if c.kwargs["position_side"] == PositionSide.SHORT)
    assert call_short.kwargs["side"] == OrderSide.BUY
    assert call_short.kwargs["amount"] == 3.0
    assert call_short.kwargs["order_type"] == OrderType.MARKET

    await db.close()
