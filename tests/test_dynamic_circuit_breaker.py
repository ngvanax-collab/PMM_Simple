"""Test suite for Dynamic NATR-Anchored Volatility Circuit Breaker (3-Tier Architecture)."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.core.gateway import ExchangeGateway
from app.core.market_state import MarketState
from app.core.position_tracker import PositionTracker
from app.core.quoter import PMMQuoter
from app.core.worker import PMMWorker
from app.models.config import PairConfig
from app.models.state import FillRecord, OrderPurpose, OrderRecord, OrderSide, OrderStatus, OrderType, PositionSide


def _create_pair_config(
    symbol: str,
    natr_mult: float = 1.0,
    min_thresh: float = 0.008,
    pause_sec: int = 60,
    lookback_sec: int = 60,
) -> PairConfig:
    return PairConfig(
        symbol=symbol,
        exchange="binance",
        enabled=True,
        leverage=5,
        margin_mode="isolated",
        order_amount_usdt=30.0,
        bid_spread=0.003,
        ask_spread=0.003,
        minimum_spread=0.001,
        order_levels=3,
        order_level_spread=0.003,
        order_level_amount=0.0,
        level_cooldown_sec=1800,
        order_refresh_time=45,
        requote_threshold_pct=0.001,
        min_holding_sec=10.0,
        inventory_skew_enabled=True,
        allocated_margin_usdt=50.0,
        max_long_usdt=150.0,
        max_short_usdt=150.0,
        gross_exposure_cap_usdt=250.0,
        take_profit=0.006,
        take_profit_order_type="LIMIT_MAKER",
        trailing_tp_enabled=True,
        trailing_tp_activation_pct=0.008,
        trailing_tp_callback_pct=0.002,
        stop_loss=0.015,
        stop_loss_order_type="MARKET",
        circuit_breaker_enabled=True,
        circuit_breaker_lookback_sec=lookback_sec,
        circuit_breaker_natr_multiplier=natr_mult,
        circuit_breaker_min_threshold_pct=min_thresh,
        circuit_breaker_pause_sec=pause_sec,
    )


@pytest.mark.asyncio
async def test_dynamic_threshold_scales_with_natr():
    """
    Test 1: Dynamic Threshold scales with NATR_15m and respects min threshold floor clamp.
    - NATR = 1.5% -> Threshold = 1.5% (0.015)
    - NATR = 0.5% -> Threshold clamped at min floor 0.8% (0.008)
    - NATR = 2.4% -> Threshold = 2.4% (0.024)
    """
    config = _create_pair_config("XPL/USDT:USDT", natr_mult=1.0, min_thresh=0.008)
    ms = MarketState(config)

    # 1. NATR = 1.5% (0.015)
    ms.update_natr_15m(1.5)  # 1.5%
    assert pytest.approx(ms.current_natr_15m, 1e-5) == 0.015
    assert pytest.approx(ms.get_circuit_breaker_threshold(), 1e-5) == 0.015

    # 2. NATR = 0.5% (0.005) -> clamped at floor 0.008
    ms.update_natr_15m(0.5)  # 0.5%
    assert pytest.approx(ms.current_natr_15m, 1e-5) == 0.005
    assert pytest.approx(ms.get_circuit_breaker_threshold(), 1e-5) == 0.008

    # 3. NATR = 2.4% (0.024)
    ms.update_natr_15m(2.4)
    assert pytest.approx(ms.get_circuit_breaker_threshold(), 1e-5) == 0.024


@pytest.mark.asyncio
async def test_circuit_breaker_trips_only_target_pair():
    """
    Test 2: Circuit breaker trips locally on XPL without affecting ZEC or INJ workers.
    """
    cfg_xpl = _create_pair_config("XPL/USDT:USDT")
    cfg_zec = _create_pair_config("ZEC/USDT:USDT")
    cfg_inj = _create_pair_config("INJ/USDT:USDT")

    gw = MagicMock(spec=ExchangeGateway)
    gw.get_market_precision.return_value = (4, 4, 0.0001, 0.0001)
    gw.fetch_free_balance.return_value = 1000.0
    gw.create_quote_order = AsyncMock(return_value={"id": "ord_1", "status": "open"})
    gw.cancel_order = AsyncMock(return_value=True)

    w_xpl = PMMWorker(cfg_xpl, gw)
    w_zec = PMMWorker(cfg_zec, gw)
    w_inj = PMMWorker(cfg_inj, gw)

    now = 1000.0
    with patch("time.time", return_value=now):
        # Initialize normal price history
        w_xpl.market_state.price_history_60s.append((now - 30, 1.00))
        w_zec.market_state.price_history_60s.append((now - 30, 100.00))
        w_inj.market_state.price_history_60s.append((now - 30, 25.00))

        # Set NATR 15m
        w_xpl.market_state.update_natr_15m(1.0)  # threshold = 1.0% (0.010)
        w_zec.market_state.update_natr_15m(1.0)
        w_inj.market_state.update_natr_15m(1.0)

        # XPL experiences 2.5% price spike
        w_xpl.market_state.price_history_60s.append((now, 1.025))
        # ZEC and INJ experience minimal normal price fluctuation (0.1%)
        w_zec.market_state.price_history_60s.append((now, 100.10))
        w_inj.market_state.price_history_60s.append((now, 25.02))

        # Run CB check on all three
        await w_xpl._check_dynamic_circuit_breaker()
        await w_zec._check_dynamic_circuit_breaker()
        await w_inj._check_dynamic_circuit_breaker()

        # XPL should be tripped and paused
        assert w_xpl.market_state.is_circuit_breaker_active(now) is True
        assert w_xpl.market_state.circuit_breaker_paused_until == now + 60

        # ZEC and INJ must NOT be tripped
        assert w_zec.market_state.is_circuit_breaker_active(now) is False
        assert w_inj.market_state.is_circuit_breaker_active(now) is False

        # Verify XPL sanity fails, but ZEC and INJ sanity passes
        w_xpl.market_state.best_bid = 1.00
        w_xpl.market_state.best_ask = 1.01
        w_xpl.market_state.smoothed_mid = 1.005

        w_zec.market_state.best_bid = 99.9
        w_zec.market_state.best_ask = 100.1
        w_zec.market_state.smoothed_mid = 100.0

        is_sane_xpl, reason_xpl = w_xpl.market_state.check_market_sanity()
        is_sane_zec, reason_zec = w_zec.market_state.check_market_sanity()

        assert is_sane_xpl is False
        assert "Circuit Breaker" in reason_xpl
        assert is_sane_zec is True


@pytest.mark.asyncio
async def test_circuit_breaker_cancels_entries_preserves_tp_sl():
    """
    Test 3: When CB trips, active Entry Quote Orders are cancelled,
    while Trailing TP and Virtual Local SL remain 100% active and responsive.
    """
    cfg = _create_pair_config("XPL/USDT:USDT")
    gw = MagicMock(spec=ExchangeGateway)
    gw.get_market_precision.return_value = (4, 4, 0.0001, 0.0001)
    gw.fetch_free_balance.return_value = 1000.0
    gw.create_quote_order = AsyncMock(return_value={"id": "q_buy_1", "status": "open"})
    gw.cancel_order = AsyncMock(return_value=True)
    gw.create_exit_order = AsyncMock(return_value={"id": "sl_exit_1", "status": "closed"})

    worker = PMMWorker(cfg, gw)

    # Setup active quotes in worker
    worker._active_quote_orders["q_buy_1"] = OrderRecord(
        id="q_buy_1",
        client_order_id="q_buy_0_123",
        exchange_order_id="q_buy_1",
        symbol=cfg.symbol,
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        order_type=OrderType.LIMIT,
        price=1.00,
        amount=30.0,
        filled_amount=0.0,
        remaining_amount=30.0,
        status=OrderStatus.NEW,
        purpose=OrderPurpose.ENTRY_QUOTE,
        created_at=time.time(),
        updated_at=time.time(),
    )

    # Setup active LONG position with Virtual SL at 0.985 (1.5% SL from 1.00)
    entry_fill = FillRecord(
        id="fill_1",
        order_id="q_buy_1",
        client_order_id="q_buy_0_123",
        symbol=cfg.symbol,
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=1.00,
        amount=30.0,
        quote_amount=30.0,
        fee=0.01,
        fee_currency="USDT",
        is_maker=True,
        timestamp=time.time() - 30.0,
        realized_pnl=0.0,
    )
    now = 1000.0
    with patch("app.core.worker.db.save_fill", new_callable=AsyncMock):
        await worker.on_fill(entry_fill)
        worker.executor_long.state.entry_timestamp = time.time() - 30.0  # past min_holding_sec

        with patch("time.time", return_value=now):
            # 1. Price spike from 1.00 to 1.02 (+2.0% spike, exceeds NATR 1.2%)
            worker.market_state.price_history_60s.append((now - 10, 1.00))
            worker.market_state.price_history_60s.append((now, 1.02))

            # Check circuit breaker
            await worker._check_dynamic_circuit_breaker()

            # Verify CB tripped
            assert worker.market_state.is_circuit_breaker_active(now) is True

            # Verify active entry quote order was cancelled
            gw.cancel_order.assert_awaited_once_with(cfg.symbol, "q_buy_1")
            assert "q_buy_1" not in worker._active_quote_orders

            # 2. Verify Virtual SL is STILL 100% active and triggers when price dumps to 0.980 (breaching SL 0.985)
            # Even during CB active period, on_ticker_update executes check_runtime_barriers
            await worker.on_ticker_update(bid=0.979, ask=0.981, mark=0.980)

            # Verify Virtual SL executed MARKET exit
            assert gw.create_exit_order.call_count >= 1
            exit_call = gw.create_exit_order.call_args[1]
            assert exit_call["symbol"] == cfg.symbol
            assert exit_call["side"] == OrderSide.SELL
            assert exit_call["position_side"] == PositionSide.LONG
            assert exit_call["order_type"] == OrderType.MARKET
            assert exit_call["purpose"] == OrderPurpose.STOP_LOSS


@pytest.mark.asyncio
async def test_auto_resume_after_cooldown_window():
    """
    Test 4: After 60s pause window and delta_p_60s drops below threshold,
    the worker automatically resets pause flag and resumes normal Level 0 quoting.
    """
    cfg = _create_pair_config("XPL/USDT:USDT", pause_sec=60, lookback_sec=60)
    gw = MagicMock(spec=ExchangeGateway)
    gw.get_market_precision.return_value = (4, 4, 0.0001, 0.0001)
    gw.fetch_free_balance.return_value = 1000.0
    gw.create_quote_order = AsyncMock(return_value={"id": "q_resumed_1", "status": "open"})
    gw.cancel_order = AsyncMock(return_value=True)

    worker = PMMWorker(cfg, gw)
    worker.market_state.best_bid = 1.0000
    worker.market_state.best_ask = 1.0020
    worker.market_state.smoothed_mid = 1.0010
    worker.market_state.mark_price = 1.0010
    worker.market_state.update_natr_15m(1.2)  # threshold = 1.2% (0.012)

    # 1. Trip the CB at t = 1000
    t0 = 1000.0
    worker.market_state.price_history_60s.append((t0 - 20, 1.00))
    worker.market_state.price_history_60s.append((t0, 1.025))  # +2.5% spike

    with patch("time.time", return_value=t0):
        await worker._check_dynamic_circuit_breaker()
        assert worker.market_state.is_circuit_breaker_active(t0) is True
        assert worker.market_state.circuit_breaker_paused_until == t0 + 60  # 1060

    # 2. Advance time to t = 1030 (30s later - still within 60s window)
    t1 = 1030.0
    with patch("time.time", return_value=t1):
        assert worker.market_state.is_circuit_breaker_active(t1) is True

    # 3. Advance time to t = 1065 (65s later) with calm price history (delta = 0.1%)
    t2 = 1065.0
    worker.market_state.price_history_60s.clear()
    worker.market_state.price_history_60s.append((t2 - 30, 1.0010))
    worker.market_state.price_history_60s.append((t2, 1.0015))  # +0.05% calm

    with patch("time.time", return_value=t2):
        await worker._check_dynamic_circuit_breaker()

        # Paused flag must be reset to 0.0
        assert worker.market_state.is_circuit_breaker_active(t2) is False
        assert worker.market_state.circuit_breaker_paused_until == 0.0

        # Market sanity passes
        is_sane, reason = worker.market_state.check_market_sanity()
        assert is_sane is True
