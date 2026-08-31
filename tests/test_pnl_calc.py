"""Unit Tests for Realized PnL Calculation & Fill Tracking."""
import time
import pytest
from unittest.mock import MagicMock

from app.core.position_tracker import PositionTracker
from app.models.config import PairConfig
from app.models.state import FillRecord, OrderSide, PositionSide
from app.persistence.db import Database


@pytest.fixture
def position_tracker():
    cfg = PairConfig(
        symbol="ZEC/USDT:USDT",
        leverage=5,
    )
    mock_gw = MagicMock()
    return PositionTracker(cfg, mock_gw)


@pytest.mark.asyncio
async def test_long_exit_positive_realized_pnl(position_tracker):
    """LONG: BUY 1 ZEC @ 500, SELL 1 ZEC @ 510 -> Realized PnL = +10.0"""
    # 1. Entry fill: BUY 1 ZEC @ 500
    entry_fill = FillRecord(
        id="fill_1",
        order_id="ord_1",
        symbol="ZEC/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=500.0,
        amount=1.0,
        quote_amount=500.0,
        fee=0.1,
        timestamp=time.time(),
        realized_pnl=0.0,
    )
    await position_tracker.on_fill(entry_fill)
    assert position_tracker.long_pos.amount == 1.0
    assert position_tracker.long_pos.entry_price == 500.0

    # 2. Exit fill: SELL 1 ZEC @ 510
    exit_fill = FillRecord(
        id="fill_2",
        order_id="ord_2",
        symbol="ZEC/USDT:USDT",
        side=OrderSide.SELL,
        position_side=PositionSide.LONG,
        price=510.0,
        amount=1.0,
        quote_amount=510.0,
        fee=0.1,
        timestamp=time.time(),
        realized_pnl=0.0,
    )
    await position_tracker.on_fill(exit_fill)

    # Realized PnL = (510 - 500) * 1.0 = +10.0
    assert exit_fill.realized_pnl == pytest.approx(10.0, abs=1e-4)
    assert position_tracker.long_pos.amount == 0.0


@pytest.mark.asyncio
async def test_long_exit_negative_realized_pnl(position_tracker):
    """LONG: BUY 1 ZEC @ 504.95, SELL 1 ZEC @ 501.69 -> Realized PnL = -3.26"""
    # 1. Entry fill: BUY 1 ZEC @ 504.95
    entry_fill = FillRecord(
        id="fill_1",
        order_id="ord_1",
        symbol="ZEC/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=504.95,
        amount=1.0,
        quote_amount=504.95,
        fee=0.1,
        timestamp=time.time(),
        realized_pnl=0.0,
    )
    await position_tracker.on_fill(entry_fill)

    # 2. Exit fill: SELL 1 ZEC @ 501.69
    exit_fill = FillRecord(
        id="fill_2",
        order_id="ord_2",
        symbol="ZEC/USDT:USDT",
        side=OrderSide.SELL,
        position_side=PositionSide.LONG,
        price=501.69,
        amount=1.0,
        quote_amount=501.69,
        fee=0.1,
        timestamp=time.time(),
        realized_pnl=0.0,
    )
    await position_tracker.on_fill(exit_fill)

    # Realized PnL = (501.69 - 504.95) * 1.0 = -3.26
    assert exit_fill.realized_pnl == pytest.approx(-3.26, abs=1e-4)
    assert position_tracker.long_pos.amount == 0.0


@pytest.mark.asyncio
async def test_short_exit_positive_and_negative_realized_pnl(position_tracker):
    """SHORT: SELL 1 ZEC @ 500, BUY 1 ZEC @ 490 -> +10.0; SELL 1 ZEC @ 500, BUY 1 ZEC @ 510 -> -10.0"""
    # Case A: Profitable Short
    entry_short = FillRecord(
        id="fill_s1",
        order_id="ord_s1",
        symbol="ZEC/USDT:USDT",
        side=OrderSide.SELL,
        position_side=PositionSide.SHORT,
        price=500.0,
        amount=1.0,
        quote_amount=500.0,
        fee=0.1,
        timestamp=time.time(),
    )
    await position_tracker.on_fill(entry_short)

    exit_short_profit = FillRecord(
        id="fill_s2",
        order_id="ord_s2",
        symbol="ZEC/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.SHORT,
        price=490.0,
        amount=1.0,
        quote_amount=490.0,
        fee=0.1,
        timestamp=time.time(),
    )
    await position_tracker.on_fill(exit_short_profit)
    assert exit_short_profit.realized_pnl == pytest.approx(10.0, abs=1e-4)

    # Case B: Loss-making Short
    entry_short_2 = FillRecord(
        id="fill_s3",
        order_id="ord_s3",
        symbol="ZEC/USDT:USDT",
        side=OrderSide.SELL,
        position_side=PositionSide.SHORT,
        price=500.0,
        amount=1.0,
        quote_amount=500.0,
        fee=0.1,
        timestamp=time.time(),
    )
    await position_tracker.on_fill(entry_short_2)

    exit_short_loss = FillRecord(
        id="fill_s4",
        order_id="ord_s4",
        symbol="ZEC/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.SHORT,
        price=510.0,
        amount=1.0,
        quote_amount=510.0,
        fee=0.1,
        timestamp=time.time(),
    )
    await position_tracker.on_fill(exit_short_loss)
    assert exit_short_loss.realized_pnl == pytest.approx(-10.0, abs=1e-4)


@pytest.mark.asyncio
async def test_db_get_pnl_summary_from_fills(tmp_path):
    """Verify Database.get_pnl_summary calculates total_realized_pnl, total_fee, and total_net_pnl from fills."""
    db_file = str(tmp_path / "test_pnl.db")
    test_db = Database(db_path=db_file)
    await test_db.connect()

    # Save entry and exit fills
    fill_1 = FillRecord(
        id="f1",
        order_id="o1",
        symbol="ZEC/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=504.95,
        amount=1.0,
        quote_amount=504.95,
        fee=0.05,
        timestamp=100.0,
        realized_pnl=0.0,
    )
    fill_2 = FillRecord(
        id="f2",
        order_id="o2",
        symbol="ZEC/USDT:USDT",
        side=OrderSide.SELL,
        position_side=PositionSide.LONG,
        price=501.69,
        amount=1.0,
        quote_amount=501.69,
        fee=0.05,
        timestamp=105.0,
        realized_pnl=-3.26,
    )
    await test_db.save_fill(fill_1)
    await test_db.save_fill(fill_2)

    summary = await test_db.get_pnl_summary()
    assert summary["total_realized_pnl"] == pytest.approx(-3.26, abs=1e-4)
    assert summary["total_fee"] == pytest.approx(0.10, abs=1e-4)
    assert summary["total_net_pnl"] == pytest.approx(-3.36, abs=1e-4)

    await test_db.close()
