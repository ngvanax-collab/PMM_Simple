"""Unit Tests for 2-Slot Hedge Reconcile & Desync Recovery."""
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.core.position_tracker import PositionTracker
from app.models.config import PairConfig
from app.models.state import PositionSide, SidePositionState


@pytest.mark.asyncio
async def test_reconcile_dual_side_positions():
    """Verify that PositionTracker correctly synchronizes when both LONG and SHORT exist."""
    config = PairConfig(symbol="SOL/USDT:USDT")
    mock_gateway = MagicMock()

    # Mock exchange returning both a LONG position of 15 SOL and a SHORT position of 10 SOL
    real_long = SidePositionState(
        symbol="SOL/USDT:USDT",
        position_side=PositionSide.LONG,
        amount=15.0,
        entry_price=102.50,
        current_price=103.00,
        notional=1545.0,
        unrealized_pnl=7.50,
    )
    real_short = SidePositionState(
        symbol="SOL/USDT:USDT",
        position_side=PositionSide.SHORT,
        amount=10.0,
        entry_price=104.00,
        current_price=103.00,
        notional=1030.0,
        unrealized_pnl=10.00,
    )
    mock_gateway.fetch_positions_hedge = AsyncMock(return_value=(real_long, real_short))

    tracker = PositionTracker(config, mock_gateway)
    assert tracker.long_pos.amount == 0.0
    assert tracker.short_pos.amount == 0.0

    # Perform Reconcile
    success = await tracker.reconcile_with_exchange()
    assert success is True
    assert tracker.long_pos.amount == 15.0
    assert tracker.long_pos.entry_price == 102.50
    assert tracker.short_pos.amount == 10.0
    assert tracker.short_pos.entry_price == 104.00
    assert tracker.gross_exposure_usdt == 2575.0
