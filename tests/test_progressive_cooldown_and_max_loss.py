"""Comprehensive Unit Tests for Progressive SL Cooldown, Worker Max Loss & Drawdown Isolated Kill Switch."""
import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.gateway import ExchangeGateway
from app.core.position_tracker import PositionTracker
from app.core.worker import PMMWorker
from app.models.config import ExchangeCredentials, PairConfig
from app.models.state import FillRecord, OrderPurpose, OrderRecord, OrderSide, OrderStatus, OrderType, PositionSide
from app.persistence.store import config_store


@pytest.fixture
def pair_config():
    return PairConfig(
        symbol="SOL/USDT:USDT",
        exchange="binance",
        enabled=True,
        leverage=5,
        margin_mode="isolated",
        order_amount_usdt=50.0,
        base_cooldown_sec=900,
        cooldown_multiplier=2.0,
        max_cooldown_sec=86400,
        worker_max_loss_usdt=30.0,
        worker_max_drawdown_usdt=40.0,
        is_locked=False,
    )


@pytest.fixture
def mock_gateway():
    gateway = ExchangeGateway(
        credentials=ExchangeCredentials(exchange="binance", api_key="k1", api_secret="s1")
    )
    gateway._is_connected = True
    gateway.get_market_precision = MagicMock(return_value=(2, 2, 0.01, 0.01))
    gateway.setup_symbol = AsyncMock(return_value=True)
    gateway.fetch_positions_hedge = AsyncMock(return_value=(
        MagicMock(amount=0.0, entry_price=0.0, current_price=100.0),
        MagicMock(amount=0.0, entry_price=0.0, current_price=100.0)
    ))
    gateway.fetch_ticker_and_mark = AsyncMock(return_value={"bid": 99.9, "ask": 100.1, "mark": 100.0})
    gateway.create_exit_order = AsyncMock(return_value={"id": "exit_123"})
    gateway.cancel_order = AsyncMock(return_value=True)
    return gateway


@pytest.fixture
def position_tracker(pair_config, mock_gateway):
    return PositionTracker(pair_config, mock_gateway)


@pytest.fixture(autouse=True)
def mock_db_ops():
    with patch("app.core.worker.db.save_fill", new_callable=AsyncMock), \
         patch("app.core.worker.db.record_pnl", new_callable=AsyncMock), \
         patch("app.core.executor.db.save_fill", new_callable=AsyncMock), \
         patch("app.core.executor.db.record_pnl", new_callable=AsyncMock):
        yield


@pytest.fixture
def worker(pair_config, mock_gateway):
    return PMMWorker(pair_config, mock_gateway)


# ── Test 1: Progressive Cooldown Exponential Growth ──

def test_progressive_cooldown_exponential_growth(position_tracker, pair_config):
    """
    SL #1 -> 900s (15m)
    SL #2 -> 1800s (30m)
    SL #3 -> 3600s (60m)
    SL #4 -> 7200s (120m)
    """
    now = time.time()
    with patch("time.time", return_value=now):
        # 1st SL
        position_tracker.set_sl_cooldown(PositionSide.LONG)
        is_cd, lvl, rem = position_tracker.get_cooldown_info(PositionSide.LONG)
        assert is_cd is True
        assert lvl == 1
        assert rem == 900  # 900 * (2^0)

        # 2nd SL
        position_tracker.set_sl_cooldown(PositionSide.LONG)
        is_cd, lvl, rem = position_tracker.get_cooldown_info(PositionSide.LONG)
        assert is_cd is True
        assert lvl == 2
        assert rem == 1800  # 900 * (2^1)

        # 3rd SL
        position_tracker.set_sl_cooldown(PositionSide.LONG)
        is_cd, lvl, rem = position_tracker.get_cooldown_info(PositionSide.LONG)
        assert is_cd is True
        assert lvl == 3
        assert rem == 3600  # 900 * (2^2)

        # SHORT side should remain untouched
        is_cd_s, lvl_s, _ = position_tracker.get_cooldown_info(PositionSide.SHORT)
        assert is_cd_s is False
        assert lvl_s == 0


# ── Test 2: Take Profit Resets Progressive Cooldown ──

@pytest.mark.asyncio
async def test_take_profit_resets_progressive_cooldown(worker):
    """
    When in Cooldown Lvl 3, a profitable TP execution resets consecutive_sl_count = 0
    and cooldown_until = 0.0.
    """
    now = time.time()
    with patch("time.time", return_value=now):
        # Hit SL 3 times
        worker.tracker.set_sl_cooldown(PositionSide.LONG)
        worker.tracker.set_sl_cooldown(PositionSide.LONG)
        worker.tracker.set_sl_cooldown(PositionSide.LONG)

        is_cd, lvl, rem = worker.tracker.get_cooldown_info(PositionSide.LONG)
        assert is_cd is True
        assert lvl == 3
        assert rem == 3600

        # Simulate profitable Take Profit fill
        tp_fill = FillRecord(
            id="fill_tp_1",
            order_id="tp_order_123",
            client_order_id="tp_long_0_12345",
            symbol="SOL/USDT:USDT",
            side=OrderSide.SELL,
            position_side=PositionSide.LONG,
            price=105.0,
            amount=1.0,
            quote_amount=105.0,
            fee=0.05,
            realized_pnl=5.0,  # Profitable!
            timestamp=now,
        )

        await worker.on_fill(tp_fill)

        # Cooldown should be completely reset
        is_cd, lvl, rem = worker.tracker.get_cooldown_info(PositionSide.LONG)
        assert is_cd is False
        assert lvl == 0
        assert rem == 0


# ── Test 3: Worker Max Loss Triggers Isolated Kill Switch ──

@pytest.mark.asyncio
async def test_worker_max_loss_triggers_isolated_kill(worker, mock_gateway):
    """
    When cumulative session losses exceed worker_max_loss_usdt (-30.0 USDT):
    - Worker triggers isolated kill switch.
    - Active quotes are cancelled.
    - Open positions are market closed.
    - Worker config is marked is_locked = True.
    """
    # Seed an open position
    worker.tracker.long_pos.amount = 2.0
    worker.tracker.long_pos.entry_price = 100.0
    worker.tracker.long_pos.current_price = 85.0
    worker.tracker.long_pos.unrealized_pnl = -30.0  # -30.0 USDT loss

    worker.session_realized_pnl = -5.0  # Total = -35.0 <= -30.0
    worker.quoter.quantize_amount = MagicMock(return_value=2.0)

    await worker._check_isolated_risk_breach()

    assert worker.is_locked_killed is True
    assert worker.config.is_locked is True
    assert worker.config.enabled is False
    assert worker.is_running is False

    # Verify market exit order was dispatched
    mock_gateway.create_exit_order.assert_called_once()
    call_kwargs = mock_gateway.create_exit_order.call_args.kwargs
    assert call_kwargs["side"] == OrderSide.SELL
    assert call_kwargs["position_side"] == PositionSide.LONG
    assert call_kwargs["order_type"] == OrderType.MARKET
    assert call_kwargs["purpose"] == OrderPurpose.CIRCUIT_BREAKER_EXIT


# ── Test 4: Worker Max Drawdown Triggers Isolated Kill Switch ──

@pytest.mark.asyncio
async def test_worker_max_drawdown_triggers_isolated_kill(worker, mock_gateway):
    """
    When PnL drops from peak by >= worker_max_drawdown_usdt (40.0 USDT):
    Peak PnL = +50.0, Current PnL = +5.0 -> Drawdown = 45.0 >= 40.0.
    Triggers isolated kill switch.
    """
    worker.peak_pnl = 50.0
    worker.session_realized_pnl = 5.0
    worker.tracker.long_pos.unrealized_pnl = 0.0
    worker.tracker.short_pos.unrealized_pnl = 0.0

    worker.tracker.short_pos.amount = 3.0
    worker.quoter.quantize_amount = MagicMock(return_value=3.0)

    await worker._check_isolated_risk_breach()

    assert worker.is_locked_killed is True
    assert worker.config.is_locked is True
    assert worker.is_running is False

    # Verify market close for SHORT position
    mock_gateway.create_exit_order.assert_called_once()
    call_kwargs = mock_gateway.create_exit_order.call_args.kwargs
    assert call_kwargs["side"] == OrderSide.BUY
    assert call_kwargs["position_side"] == PositionSide.SHORT
    assert call_kwargs["purpose"] == OrderPurpose.CIRCUIT_BREAKER_EXIT


# ── Test 5: Config Store & Hot Reload ──

def test_config_store_and_ui_hot_reload(pair_config):
    """
    Verify that saving progressive cooldown and worker loss limits persists correctly
    and reloads cleanly into PairConfig.
    """
    pair_config.base_cooldown_sec = 300
    pair_config.cooldown_multiplier = 3.0
    pair_config.max_cooldown_sec = 43200
    pair_config.worker_max_loss_usdt = 25.0
    pair_config.worker_max_drawdown_usdt = 35.0

    try:
        config_store.save_pair_config(pair_config)
        loaded = config_store.load_pair_config(pair_config.symbol)

        assert loaded is not None
        assert loaded.base_cooldown_sec == 300
        assert loaded.cooldown_multiplier == 3.0
        assert loaded.max_cooldown_sec == 43200
        assert loaded.worker_max_loss_usdt == 25.0
        assert loaded.worker_max_drawdown_usdt == 35.0
    finally:
        config_store.delete_pair_config(pair_config.symbol)



# ── Test 6: Manual Unlock Restores Worker ──

@pytest.mark.asyncio
async def test_manual_unlock_restores_worker(worker):
    """
    Calling unlock() clears is_locked flag, resets session PnL & peak PnL,
    clears SL cooldowns, and allows the worker to run again.
    """
    # Put worker in locked state
    worker.config.is_locked = True
    worker.is_locked_killed = True
    worker.session_realized_pnl = -50.0
    worker.peak_pnl = 10.0
    worker.tracker.set_sl_cooldown(PositionSide.LONG)

    # Calling should_requote is blocked
    can_quote, reason = worker._check_should_requote()
    assert can_quote is False
    assert "Locked" in reason

    # Unlock worker
    await worker.unlock()

    assert worker.config.is_locked is False
    assert worker.is_locked_killed is False
    assert worker.session_realized_pnl == 0.0
    assert worker.peak_pnl == 0.0

    # Cooldown is cleared
    is_cd, lvl, _ = worker.tracker.get_cooldown_info(PositionSide.LONG)
    assert is_cd is False
    assert lvl == 0
