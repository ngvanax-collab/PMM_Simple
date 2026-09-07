"""Unit Tests for -2022 ReduceOnly Reconciliation & Retry Fail-Safe (TASK RO-4)."""
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.core.executor import TripleBarrierExecutor
from app.models.config import PairConfig
from app.models.state import (
    OrderPurpose,
    OrderSide,
    OrderType,
    PositionSide,
    SidePositionState,
)


@pytest.fixture
def mock_executor():
    cfg = PairConfig(symbol="AAVE/USDT:USDT")
    gw = MagicMock()
    gw._last_exit_error = ""
    gw.create_exit_order = AsyncMock(return_value={"id": "exit_1", "amount": 0.4})
    gw.cancel_order = AsyncMock(return_value=True)

    tracker = MagicMock()
    tracker.short_pos = SidePositionState(
        symbol="AAVE/USDT:USDT",
        position_side=PositionSide.SHORT,
        amount=0.4,
        entry_price=130.0,
    )
    tracker.reconcile_with_exchange = AsyncMock(return_value=True)

    quoter = MagicMock()
    quoter.quantize_amount = lambda a: round(a, 4)
    quoter.calculate_spread_floor.return_value = 0.001
    quoter.quantize_price = lambda p, is_bid: round(p, 2)

    executor = TripleBarrierExecutor(
        config=cfg,
        position_side=PositionSide.SHORT,
        gateway=gw,
        position_tracker=tracker,
        quoter=quoter,
    )
    executor.state.remaining_qty = 0.4
    executor.state.active = True
    return executor


@pytest.mark.asyncio
async def test_exit_order_2022_triggers_reconcile_and_retries_with_updated_qty(mock_executor):
    """When create_exit_order receives -2022, reconciles with exchange and retries once."""
    # 1st call fails with -2022
    # 2nd call (retry) succeeds with updated amount
    def side_effect_create(*args, **kwargs):
        if mock_executor.gateway.create_exit_order.call_count == 1:
            mock_executor.gateway._last_exit_error = 'binance {"code":-2022,"msg":"ReduceOnly Order is rejected."}'
            return None
        else:
            return {"id": "exit_retry_ok", "amount": 0.2}

    mock_executor.gateway.create_exit_order = AsyncMock(side_effect=side_effect_create)

    # Reconcile updates position from 0.4 to 0.2
    async def mock_reconcile():
        mock_executor.tracker.short_pos.amount = 0.2
        return True

    mock_executor.tracker.reconcile_with_exchange = AsyncMock(side_effect=mock_reconcile)

    resp = await mock_executor._safe_create_exit_order(
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT_MAKER,
        amount=0.4,
        price=129.0,
        client_order_id="tp_short_0",
        purpose=OrderPurpose.TAKE_PROFIT,
    )

    # Assertions
    assert resp is not None
    assert resp["id"] == "exit_retry_ok"
    assert resp["amount"] == 0.2
    assert mock_executor.tracker.reconcile_with_exchange.called
    assert mock_executor.gateway.create_exit_order.call_count == 2
    # Verify retry call passed the updated reconciled amount
    retry_call_kwargs = mock_executor.gateway.create_exit_order.call_args_list[1].kwargs
    assert retry_call_kwargs["amount"] == 0.2


@pytest.mark.asyncio
async def test_exit_order_2022_aborts_retry_when_reconciled_flat(mock_executor):
    """When -2022 occurs and exchange reconcile reveals position is 0 (flat), no retry is placed."""
    mock_executor.gateway._last_exit_error = 'binance {"code":-2022,"msg":"ReduceOnly Order is rejected."}'
    mock_executor.gateway.create_exit_order = AsyncMock(return_value=None)

    async def mock_reconcile_flat():
        mock_executor.tracker.short_pos.amount = 0.0
        return True

    mock_executor.tracker.reconcile_with_exchange = AsyncMock(side_effect=mock_reconcile_flat)

    resp = await mock_executor._safe_create_exit_order(
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT_MAKER,
        amount=0.4,
        price=129.0,
        client_order_id="tp_short_0",
        purpose=OrderPurpose.TAKE_PROFIT,
    )

    assert resp is None
    assert mock_executor.tracker.reconcile_with_exchange.called
    # Only the initial failed call was made; no retry because position is 0
    assert mock_executor.gateway.create_exit_order.call_count == 1
    assert mock_executor.state.active is False


@pytest.mark.asyncio
async def test_non_2022_error_does_not_trigger_reconcile(mock_executor):
    """Standard errors other than -2022 should not trigger position reconciliation."""
    mock_executor.gateway._last_exit_error = 'binance {"code":-1001,"msg":"Internal error"}'
    mock_executor.gateway.create_exit_order = AsyncMock(return_value=None)

    resp = await mock_executor._safe_create_exit_order(
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT_MAKER,
        amount=0.4,
        price=129.0,
        client_order_id="tp_short_0",
        purpose=OrderPurpose.TAKE_PROFIT,
    )

    assert resp is None
    assert not mock_executor.tracker.reconcile_with_exchange.called
    assert mock_executor.gateway.create_exit_order.call_count == 1


@pytest.mark.asyncio
async def test_exit_order_2022_retry_client_order_id_unique_when_original_is_max_length(mock_executor):
    """Ensure retry client_order_id is unique and <= 36 chars when original CID is max length 36."""
    original_cid = "a" * 36

    def side_effect_create(*args, **kwargs):
        if mock_executor.gateway.create_exit_order.call_count == 1:
            mock_executor.gateway._last_exit_error = 'binance {"code":-2022,"msg":"ReduceOnly Order is rejected."}'
            return None
        else:
            return {"id": "exit_retry_ok", "amount": 0.2}

    mock_executor.gateway.create_exit_order = AsyncMock(side_effect=side_effect_create)

    async def mock_reconcile():
        mock_executor.tracker.short_pos.amount = 0.2
        return True

    mock_executor.tracker.reconcile_with_exchange = AsyncMock(side_effect=mock_reconcile)

    resp = await mock_executor._safe_create_exit_order(
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT_MAKER,
        amount=0.4,
        price=129.0,
        client_order_id=original_cid,
        purpose=OrderPurpose.TAKE_PROFIT,
    )

    assert resp is not None
    assert mock_executor.gateway.create_exit_order.call_count == 2

    retry_call_kwargs = mock_executor.gateway.create_exit_order.call_args_list[1].kwargs
    retry_client_id = retry_call_kwargs.get("client_order_id")

    assert retry_client_id is not None
    assert retry_client_id != original_cid
    assert len(retry_client_id) <= 36
    # Verify retry call to gateway used this exact new retry_client_id
    assert retry_call_kwargs["client_order_id"] == retry_client_id

