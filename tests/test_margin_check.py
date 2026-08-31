"""Unit Tests for Isolated Per-Pair Margin Quota, Free Margin Pre-Check & Auto-Scaling."""
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.core.gateway import ExchangeGateway
from app.core.quoter import PMMQuoter
from app.core.worker import PMMWorker
from app.models.config import ExchangeCredentials, PairConfig
from app.models.state import OrderRecord, OrderSide, OrderStatus, OrderType, PositionSide


# ── 1. Test PairConfig effective_margin_cap & Fallback ──

def test_pair_effective_margin_cap_fallback():
    """Verify effective_margin_cap uses allocated_margin_usdt if set, or falls back to gross_exposure_cap / leverage."""
    # Case A: allocated_margin_usdt is None -> Fallback to gross_exposure_cap / leverage
    cfg_default = PairConfig(
        symbol="SOL/USDT:USDT",
        leverage=5,
        gross_exposure_cap_usdt=500.0,
        allocated_margin_usdt=None,
    )
    assert cfg_default.effective_margin_cap == 100.0  # 500 / 5 = 100.0

    # Case B: allocated_margin_usdt is explicitly configured
    cfg_explicit = PairConfig(
        symbol="SOL/USDT:USDT",
        leverage=5,
        gross_exposure_cap_usdt=500.0,
        allocated_margin_usdt=35.0,
    )
    assert cfg_explicit.effective_margin_cap == 35.0


# ── 2. Test Quoter Margin Estimation ──

def test_estimate_minimum_margin_required():
    """Verify estimate_minimum_margin_required calculates correct margin for active sides."""
    cfg = PairConfig(
        symbol="SOL/USDT:USDT",
        leverage=5,
        order_amount_usdt=50.0,
        order_levels=3,
    )
    quoter = PMMQuoter(config=cfg, min_notional=5.0)

    # Both sides active (5.0 / 5 * 2 = 2.0 USDT)
    req_both = quoter.estimate_minimum_margin_required(active_long=True, active_short=True)
    assert req_both == pytest.approx(2.0, rel=1e-3)

    # Only 1 side active (5.0 / 5 * 1 = 1.0 USDT)
    req_one = quoter.estimate_minimum_margin_required(active_long=True, active_short=False)
    assert req_one == pytest.approx(1.0, rel=1e-3)

    # 0 sides active
    req_none = quoter.estimate_minimum_margin_required(active_long=False, active_short=False)
    assert req_none == 0.0


# ── 3. Test calculate_quotes with Available Margin Budgeting & Auto-Scale Down ──

def test_quoter_auto_scales_and_prunes_levels():
    """Verify quoter prunes levels when margin is exhausted and scales down level amount for partial margin."""
    cfg = PairConfig(
        symbol="SOL/USDT:USDT",
        leverage=5,
        order_amount_usdt=50.0,
        order_level_amount=25.0,  # nominal: L0=50, L1=75, L2=100
        order_levels=3,
        max_long_usdt=500.0,
        max_short_usdt=500.0,
        gross_exposure_cap_usdt=1000.0,
    )
    quoter = PMMQuoter(config=cfg, price_precision=2, amount_precision=3, tick_size=0.01, step_size=0.001)

    # 30 USDT margin at 5x = 150 USDT total notional -> 75 USDT per side.
    # Level 0 = 50 USDT -> remaining budget = 25 USDT.
    # Nominal Level 1 is 75 USDT, scaled down to 25 USDT.
    # Level 2 omitted since remaining budget becomes 0 (< min_notional 5.0).
    bids, asks = quoter.calculate_quotes(
        smoothed_mid=100.0,
        long_value_usdt=0.0,
        short_value_usdt=0.0,
        available_margin=30.0,
    )

    assert len(bids) == 2
    assert len(asks) == 2

    assert bids[0].level == 0
    assert bids[0].notional == pytest.approx(50.0, abs=0.5)

    assert bids[1].level == 1
    assert bids[1].notional == pytest.approx(25.0, abs=1.0)


def test_calculate_quotes_with_ample_margin():
    """When free margin is plenty, all levels are quoted at full nominal amounts."""
    cfg = PairConfig(
        symbol="SOL/USDT:USDT",
        leverage=5,
        order_amount_usdt=50.0,
        order_level_amount=25.0,
        order_levels=3,
        max_long_usdt=500.0,
        max_short_usdt=500.0,
        gross_exposure_cap_usdt=1000.0,
    )
    quoter = PMMQuoter(config=cfg, price_precision=2, amount_precision=3, tick_size=0.01, step_size=0.001)

    bids, asks = quoter.calculate_quotes(
        smoothed_mid=100.0,
        long_value_usdt=0.0,
        short_value_usdt=0.0,
        available_margin=200.0,
    )

    assert len(bids) == 3
    assert len(asks) == 3
    assert bids[0].notional == pytest.approx(50.0, abs=0.5)
    assert bids[1].notional == pytest.approx(75.0, abs=0.5)
    assert bids[2].notional == pytest.approx(100.0, abs=0.5)


def test_calculate_quotes_zero_or_insufficient_margin():
    """When available margin is 0 or less than min_notional / leverage, no quotes are generated."""
    cfg = PairConfig(
        symbol="SOL/USDT:USDT",
        leverage=5,
        order_amount_usdt=50.0,
        order_levels=3,
    )
    quoter = PMMQuoter(config=cfg, min_notional=5.0)

    # 0 margin
    bids, asks = quoter.calculate_quotes(
        smoothed_mid=100.0,
        long_value_usdt=0.0,
        short_value_usdt=0.0,
        available_margin=0.0,
    )
    assert bids == []
    assert asks == []

    # Insufficient margin (0.5 USDT at 5x = 2.5 USDT total -> 1.25 USDT per side < min_notional 5.0)
    bids_low, asks_low = quoter.calculate_quotes(
        smoothed_mid=100.0,
        long_value_usdt=0.0,
        short_value_usdt=0.0,
        available_margin=0.5,
    )
    assert bids_low == []
    assert asks_low == []


# ── 4. Test Worker Isolated Per-Pair Margin Quota & Account Free Balance ──

@pytest.mark.asyncio
async def test_worker_respects_pair_margin_quota_independent_of_large_account_balance():
    """
    Even if account has 10,000 USDT free balance, worker strictly enforces the pair's allocated margin quota.
    """
    cfg = PairConfig(
        symbol="SOL/USDT:USDT",
        leverage=5,
        order_amount_usdt=50.0,
        order_levels=3,
        allocated_margin_usdt=10.0,  # Cap is only 10 USDT (50 USDT total notional -> 25 USDT per side)
    )
    mock_gw = MagicMock()
    mock_gw.get_market_precision.return_value = (2, 3, 0.01, 0.001)
    mock_gw.get_market_limits.return_value = (0.0, 5.0)
    # Huge account free balance
    mock_gw.fetch_free_balance = AsyncMock(return_value=10000.0)
    mock_gw.create_quote_order = AsyncMock(return_value={"id": "ord_1", "status": "open"})
    mock_gw.cancel_order = AsyncMock()

    worker = PMMWorker(cfg, mock_gw)
    worker.market_state.best_bid = 99.8
    worker.market_state.best_ask = 100.2
    worker.market_state.smoothed_mid = 100.0
    worker._running = True

    with patch("app.persistence.db.db.save_order", new_callable=AsyncMock):
        await worker._requote()

    # Total notional placed across bids and asks must not exceed 10 USDT * 5 = 50 USDT
    assert mock_gw.create_quote_order.call_count >= 1
    total_placed_notional = sum(
        call.kwargs["price"] * call.kwargs["amount"]
        for call in mock_gw.create_quote_order.call_args_list
    )
    assert total_placed_notional <= 50.0 + 1.0  # within 50 USDT (+ small quantization tolerance)


@pytest.mark.asyncio
async def test_worker_respects_low_account_free_balance():
    """
    When pair cap is high (100 USDT) but account free balance is very low (e.g. 2.0 USDT),
    worker pauses quoting because 2.0 * 0.95 = 1.9 USDT is insufficient for min_notional (5.0 * 2 / 5 = 2.0 USDT).
    """
    cfg = PairConfig(
        symbol="SOL/USDT:USDT",
        leverage=5,
        order_amount_usdt=50.0,
        order_levels=3,
        allocated_margin_usdt=100.0,
    )
    mock_gw = MagicMock()
    mock_gw.get_market_precision.return_value = (2, 3, 0.01, 0.001)
    mock_gw.get_market_limits.return_value = (0.0, 5.0)
    # Account has only 2.0 USDT free -> 2.0 * 0.95 = 1.9 USDT < 2.0 required
    mock_gw.fetch_free_balance = AsyncMock(return_value=2.0)
    mock_gw.create_quote_order = AsyncMock()

    worker = PMMWorker(cfg, mock_gw)
    worker.market_state.best_bid = 99.8
    worker.market_state.best_ask = 100.2
    worker.market_state.smoothed_mid = 100.0
    worker._running = True

    await worker._requote()

    # Quoting paused due to low account balance
    assert mock_gw.create_quote_order.call_count == 0
    assert worker._last_margin_warning_time > 0


@pytest.mark.asyncio
async def test_worker_accounts_for_pair_used_margin():
    """
    Worker subtracts existing position notional and active quote order notional from pair margin cap.
    """
    cfg = PairConfig(
        symbol="SOL/USDT:USDT",
        leverage=5,
        order_amount_usdt=50.0,
        order_levels=1,
        allocated_margin_usdt=20.0,  # 20 USDT margin cap (100 USDT notional cap)
    )
    mock_gw = MagicMock()
    mock_gw.get_market_precision.return_value = (2, 3, 0.01, 0.001)
    mock_gw.get_market_limits.return_value = (0.0, 5.0)
    mock_gw.fetch_free_balance = AsyncMock(return_value=1000.0)
    mock_gw.create_quote_order = AsyncMock()

    worker = PMMWorker(cfg, mock_gw)
    worker.market_state.best_bid = 99.8
    worker.market_state.best_ask = 100.2
    worker.market_state.smoothed_mid = 100.0
    worker._running = True

    # Existing LONG position of 0.95 SOL @ 100 = 95 USDT notional (uses 19 USDT margin out of 20)
    worker.tracker.long_pos.amount = 0.95
    worker.tracker.long_pos.entry_price = 100.0
    worker.tracker.long_pos.notional = 95.0

    # Remaining pair margin = 20 - 19 = 1 USDT (< min_req 2.0 USDT)
    await worker._requote()

    # Quoting must be paused
    assert mock_gw.create_quote_order.call_count == 0


# ── 5. Test Gateway Cache TTL & -2019 Error Handling ──

@pytest.mark.asyncio
async def test_gateway_fetch_free_balance_cache_ttl():
    """Verify fetch_free_balance caches result for 1.5s to reduce API rate limit consumption."""
    creds = ExchangeCredentials(exchange="binance", api_key="k", api_secret="s")
    gw = ExchangeGateway(creds)
    mock_exchange = MagicMock()
    gw._exchange = mock_exchange

    mock_exchange.fetch_balance = AsyncMock(return_value={
        "free": {"USDT": 500.0},
    })

    # First call - queries exchange
    val1 = await gw.fetch_free_balance("USDT")
    assert val1 == 500.0
    assert mock_exchange.fetch_balance.call_count == 1

    # Second immediate call - hits 1.5s cache
    val2 = await gw.fetch_free_balance("USDT")
    assert val2 == 500.0
    assert mock_exchange.fetch_balance.call_count == 1


@pytest.mark.asyncio
async def test_gateway_create_quote_order_handles_2019_insufficient_margin():
    """Verify code -2019 logs gracefully without error exception."""
    creds = ExchangeCredentials(exchange="binance", api_key="k", api_secret="s")
    gw = ExchangeGateway(creds)
    mock_exchange = MagicMock()
    gw._exchange = mock_exchange

    mock_exchange.create_order = AsyncMock(
        side_effect=Exception('{"code": -2019, "msg": "Margin is insufficient."}')
    )

    res = await gw.create_quote_order(
        symbol="SOL/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=100.0,
        amount=1.0,
        client_order_id="q_bid_0_1",
    )

    assert res is None
