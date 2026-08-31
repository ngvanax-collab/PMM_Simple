"""Unit & Integration Tests for Startup Hedge Mode Verification & Abort Safety."""
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.core.gateway import ExchangeGateway
from app.models.config import ExchangeCredentials


@pytest.mark.asyncio
async def test_binance_already_hedge_mode():
    """Verify gateway succeeds when Binance is already in Dual (Hedge) position mode."""
    creds = ExchangeCredentials(exchange="binance", api_key="k", api_secret="s")
    gw = ExchangeGateway(creds)

    mock_exchange = MagicMock()
    mock_exchange.load_markets = AsyncMock(return_value={"SOL/USDT:USDT": {}})
    mock_exchange.fapiPrivateGetPositionSideDual = AsyncMock(return_value={"dualSidePosition": True})
    mock_exchange.close = AsyncMock()

    gw._exchange = mock_exchange
    success = await gw.verify_and_set_hedge_mode()
    assert success is True


@pytest.mark.asyncio
async def test_binance_switch_to_hedge_mode_success():
    """Verify gateway switches Binance to Hedge mode when currently One-Way."""
    creds = ExchangeCredentials(exchange="binance", api_key="k", api_secret="s")
    gw = ExchangeGateway(creds)

    mock_exchange = MagicMock()
    mock_exchange.fapiPrivateGetPositionSideDual = AsyncMock(return_value={"dualSidePosition": False})
    mock_exchange.fapiPrivatePostPositionSideDual = AsyncMock(return_value={"code": 200, "msg": "success"})

    gw._exchange = mock_exchange
    success = await gw.verify_and_set_hedge_mode()
    assert success is True
    assert mock_exchange.fapiPrivatePostPositionSideDual.called


@pytest.mark.asyncio
async def test_binance_switch_to_hedge_mode_failure_aborts():
    """
    CRITICAL SAFETY TEST:
    Verify gateway aborts (returns False) when Binance is One-Way and switch fails
    (e.g., due to existing open positions/orders).
    """
    creds = ExchangeCredentials(exchange="binance", api_key="k", api_secret="s")
    gw = ExchangeGateway(creds)

    mock_exchange = MagicMock()
    mock_exchange.fapiPrivateGetPositionSideDual = AsyncMock(return_value={"dualSidePosition": False})
    mock_exchange.fapiPrivatePostPositionSideDual = AsyncMock(side_effect=Exception("Position side cannot be changed with open orders/positions"))

    gw._exchange = mock_exchange
    success = await gw.verify_and_set_hedge_mode()
    assert success is False


@pytest.mark.asyncio
async def test_bybit_hedge_mode_verification():
    """Verify Bybit position mode setup."""
    creds = ExchangeCredentials(exchange="bybit", api_key="k", api_secret="s")
    gw = ExchangeGateway(creds)

    mock_exchange = MagicMock()
    mock_exchange.set_position_mode = AsyncMock(return_value={"retCode": 0, "retMsg": "OK"})

    gw._exchange = mock_exchange
    success = await gw.verify_and_set_hedge_mode()
    assert success is True


@pytest.mark.asyncio
async def test_post_only_5022_step_back_retry():
    """Verify that when Binance returns code -5022, gateway steps back price by 1 tick and retries."""
    from app.models.state import OrderSide, PositionSide

    creds = ExchangeCredentials(exchange="binance", api_key="k", api_secret="s")
    gw = ExchangeGateway(creds)

    mock_exchange = MagicMock()
    gw._exchange = mock_exchange
    gw._market_info = {
        "XRP/USDT:USDT": {
            "precision": {"price": 4, "amount": 1},
            "limits": {"price": {"min": 0.0001}, "amount": {"min": 0.1}},
        }
    }

    # First attempt raises -5022, second attempt succeeds
    mock_exchange.create_order = AsyncMock(
        side_effect=[
            Exception('{"code": -5022, "msg": "Due to the order could not be executed as maker, the Post Only order will be rejected."}'),
            {"id": "ord_xrp_1", "status": "open", "price": 0.5233},
        ]
    )

    order = await gw.create_quote_order(
        symbol="XRP/USDT:USDT",
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        price=0.5234,
        amount=100.0,
        client_order_id="q_bid_0_test",
    )

    assert order is not None
    assert order["id"] == "ord_xrp_1"
    assert mock_exchange.create_order.call_count == 2
    # Verify second call stepped price down by 1 tick (0.5234 - 0.0001 = 0.5233)
    second_call_price = mock_exchange.create_order.call_args_list[1].kwargs["price"]
    assert second_call_price == 0.5233
