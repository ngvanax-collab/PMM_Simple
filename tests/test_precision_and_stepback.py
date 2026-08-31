"""Unit & Integration Tests for Market Precision Engine, Auto Step-Back, and Min Notional/Qty Clamping."""
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.core.gateway import ExchangeGateway
from app.core.quoter import PMMQuoter
from app.models.config import ExchangeCredentials, PairConfig
from app.models.state import OrderSide, PositionSide


# ── 1. Test Step-Back on Reject (-5022) for Low-Priced & Varied Precision Coins ──

@pytest.mark.asyncio
async def test_xrp_post_only_step_back_decreases_price_accurately():
    """Verify XRP (0.9894) BUY quote order steps down to 0.9893 (1 tick = 0.0001) instead of jumping to 1.0."""
    creds = ExchangeCredentials(exchange="binance", api_key="k", api_secret="s")
    gw = ExchangeGateway(creds)

    mock_exchange = MagicMock()
    mock_exchange.price_to_precision = MagicMock(side_effect=lambda sym, p: f"{p:.4f}")
    mock_exchange.amount_to_precision = MagicMock(side_effect=lambda sym, a: f"{a:.1f}")
    gw._exchange = mock_exchange
    gw._market_info = {
        "XRP/USDT:USDT": {
            "precision": {"price": 4, "amount": 1},
            "limits": {"price": {"min": 0.0001}, "amount": {"min": 0.1}, "cost": {"min": 5.0}},
            "info": {
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.0001"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.1", "minQty": "0.1"},
                ]
            }
        }
    }

    # First attempt raises -5022 maker rejection, second succeeds
    mock_exchange.create_order = AsyncMock(
        side_effect=[
            Exception('{"code": -5022, "msg": "Due to the order could not be executed as maker, the Post Only order will be rejected."}'),
            {"id": "ord_xrp_stepped", "status": "open", "price": 0.9893},
        ]
    )

    order = await gw.create_quote_order(
        symbol="XRP/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=0.9894,
        amount=100.0,
        client_order_id="q_bid_xrp_1",
    )

    assert order is not None
    assert mock_exchange.create_order.call_count == 2
    # Verify the second call received 0.9893 (0.9894 - 0.0001)
    stepped_price = mock_exchange.create_order.call_args_list[1].kwargs["price"]
    assert stepped_price == 0.9893, f"Expected 0.9893 but got {stepped_price}"


@pytest.mark.asyncio
async def test_trx_post_only_step_back_decreases_price_accurately():
    """Verify TRX (0.33164) BUY quote order steps down to 0.33163 (1 tick = 0.00001)."""
    creds = ExchangeCredentials(exchange="binance", api_key="k", api_secret="s")
    gw = ExchangeGateway(creds)

    mock_exchange = MagicMock()
    mock_exchange.price_to_precision = MagicMock(side_effect=lambda sym, p: f"{p:.5f}")
    mock_exchange.amount_to_precision = MagicMock(side_effect=lambda sym, a: f"{a:.0f}")
    gw._exchange = mock_exchange
    gw._market_info = {
        "TRX/USDT:USDT": {
            "precision": {"price": 5, "amount": 0},
            "limits": {"price": {"min": 0.00001}, "amount": {"min": 1.0}, "cost": {"min": 5.0}},
            "info": {
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.00001"},
                    {"filterType": "LOT_SIZE", "stepSize": "1.0", "minQty": "1.0"},
                ]
            }
        }
    }

    mock_exchange.create_order = AsyncMock(
        side_effect=[
            Exception('{"code": -5022, "msg": "Due to the order could not be executed as maker, the Post Only order will be rejected."}'),
            {"id": "ord_trx_stepped", "status": "open", "price": 0.33163},
        ]
    )

    order = await gw.create_quote_order(
        symbol="TRX/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=0.33164,
        amount=500.0,
        client_order_id="q_bid_trx_1",
    )

    assert order is not None
    assert mock_exchange.create_order.call_count == 2
    stepped_price = mock_exchange.create_order.call_args_list[1].kwargs["price"]
    assert stepped_price == 0.33163, f"Expected 0.33163 but got {stepped_price}"


@pytest.mark.asyncio
async def test_zec_post_only_step_back_decreases_price_accurately():
    """Verify ZEC (507.81) BUY quote order steps down to 507.80 (1 tick = 0.01)."""
    creds = ExchangeCredentials(exchange="binance", api_key="k", api_secret="s")
    gw = ExchangeGateway(creds)

    mock_exchange = MagicMock()
    mock_exchange.price_to_precision = MagicMock(side_effect=lambda sym, p: f"{p:.2f}")
    mock_exchange.amount_to_precision = MagicMock(side_effect=lambda sym, a: f"{a:.3f}")
    gw._exchange = mock_exchange
    gw._market_info = {
        "ZEC/USDT:USDT": {
            "precision": {"price": 2, "amount": 3},
            "limits": {"price": {"min": 0.01}, "amount": {"min": 0.001}, "cost": {"min": 5.0}},
            "info": {
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                ]
            }
        }
    }

    mock_exchange.create_order = AsyncMock(
        side_effect=[
            Exception('{"code": -5022, "msg": "Due to the order could not be executed as maker, the Post Only order will be rejected."}'),
            {"id": "ord_zec_stepped", "status": "open", "price": 507.80},
        ]
    )

    order = await gw.create_quote_order(
        symbol="ZEC/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=507.81,
        amount=0.1,
        client_order_id="q_bid_zec_1",
    )

    assert order is not None
    assert mock_exchange.create_order.call_count == 2
    stepped_price = mock_exchange.create_order.call_args_list[1].kwargs["price"]
    assert stepped_price == 507.80, f"Expected 507.80 but got {stepped_price}"


# ── 2. Test Precision and Formatters for BTC, ETH, SOL, XRP, TRX ──

def test_price_and_amount_precision_across_all_five_pairs():
    """Verify precision extraction and formatting for BTC, ETH, SOL, XRP, and TRX."""
    creds = ExchangeCredentials(exchange="binance", api_key="k", api_secret="s")
    gw = ExchangeGateway(creds)
    gw._market_info = {
        "BTC/USDT:USDT": {
            "precision": {"price": 1, "amount": 3},
            "limits": {"price": {"min": 0.1}, "amount": {"min": 0.001}, "cost": {"min": 5.0}},
            "info": {"filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.1"}, {"filterType": "LOT_SIZE", "stepSize": "0.001"}]},
        },
        "ETH/USDT:USDT": {
            "precision": {"price": 2, "amount": 3},
            "limits": {"price": {"min": 0.01}, "amount": {"min": 0.001}, "cost": {"min": 5.0}},
            "info": {"filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01"}, {"filterType": "LOT_SIZE", "stepSize": "0.001"}]},
        },
        "SOL/USDT:USDT": {
            "precision": {"price": 2, "amount": 3},
            "limits": {"price": {"min": 0.01}, "amount": {"min": 0.001}, "cost": {"min": 5.0}},
            "info": {"filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01"}, {"filterType": "LOT_SIZE", "stepSize": "0.001"}]},
        },
        "XRP/USDT:USDT": {
            "precision": {"price": 4, "amount": 1},
            "limits": {"price": {"min": 0.0001}, "amount": {"min": 0.1}, "cost": {"min": 5.0}},
            "info": {"filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.0001"}, {"filterType": "LOT_SIZE", "stepSize": "0.1"}]},
        },
        "TRX/USDT:USDT": {
            "precision": {"price": 5, "amount": 0},
            "limits": {"price": {"min": 0.00001}, "amount": {"min": 1.0}, "cost": {"min": 5.0}},
            "info": {"filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.00001"}, {"filterType": "LOT_SIZE", "stepSize": "1.0"}]},
        },
    }

    # BTC
    p_prec, a_prec, tick_sz, step_sz = gw.get_market_precision("BTC/USDT:USDT")
    assert (p_prec, a_prec, tick_sz, step_sz) == (1, 3, 0.1, 0.001)

    # ETH
    p_prec, a_prec, tick_sz, step_sz = gw.get_market_precision("ETH/USDT:USDT")
    assert (p_prec, a_prec, tick_sz, step_sz) == (2, 3, 0.01, 0.001)

    # SOL
    p_prec, a_prec, tick_sz, step_sz = gw.get_market_precision("SOL/USDT:USDT")
    assert (p_prec, a_prec, tick_sz, step_sz) == (2, 3, 0.01, 0.001)

    # XRP
    p_prec, a_prec, tick_sz, step_sz = gw.get_market_precision("XRP/USDT:USDT")
    assert (p_prec, a_prec, tick_sz, step_sz) == (4, 1, 0.0001, 0.1)

    # TRX
    p_prec, a_prec, tick_sz, step_sz = gw.get_market_precision("TRX/USDT:USDT")
    assert (p_prec, a_prec, tick_sz, step_sz) == (5, 0, 0.00001, 1.0)


# ── 3. Test Quoter Min Notional and Min Qty Clamping ──

def test_quoter_enforces_min_notional_and_min_qty():
    """Verify Quoter automatically adjusts quantity for high-price coins (e.g. BTC) to satisfy min_notional."""
    cfg = PairConfig(
        symbol="BTC/USDT:USDT",
        order_amount_usdt=2.0,  # Below 5 USDT min notional
        max_long_usdt=500.0,
        max_short_usdt=500.0,
        gross_exposure_cap_usdt=1000.0,
    )
    quoter = PMMQuoter(
        config=cfg,
        price_precision=1,
        amount_precision=3,
        tick_size=0.1,
        step_size=0.001,
        min_amount=0.001,
        min_notional=5.0,
    )

    bids, asks = quoter.calculate_quotes(
        smoothed_mid=60000.0,
        long_value_usdt=0.0,
        short_value_usdt=0.0,
    )

    assert len(bids) > 0
    assert len(asks) > 0
    # Every level must meet or exceed min_notional (5.0 USDT)
    for b in bids:
        assert (b.amount * b.price) >= 5.0, f"Bid notional {b.amount * b.price} < 5.0"
        assert b.amount >= 0.001, f"Bid amount {b.amount} < min_qty 0.001"

    for a in asks:
        assert (a.amount * a.price) >= 5.0, f"Ask notional {a.amount * a.price} < 5.0"
        assert a.amount >= 0.001, f"Ask amount {a.amount} < min_qty 0.001"
