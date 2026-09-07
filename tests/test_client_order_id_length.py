"""Unit Tests for Client Order ID Length Safety (TASK RO-3)."""
import time
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.core.executor import TripleBarrierExecutor, _safe_client_order_id
from app.models.config import PairConfig
from app.models.state import OrderPurpose, PositionSide


def test_safe_client_order_id_various_lengths():
    """_safe_client_order_id must never exceed max_len (36) regardless of prefix size."""
    prefixes = [
        "short",
        "pe_long",
        "tp_short_0",
        "exit_time_limit_exit_short",
        "extremely_long_prefix_that_is_over_thirty_six_characters_by_itself",
        "a" * 100,
    ]
    for pref in prefixes:
        cid = _safe_client_order_id(pref, max_len=36)
        assert len(cid) <= 36, f"Failed for prefix '{pref}': len={len(cid)} > 36 ({cid})"
        # Verify timestamp is at the end
        parts = cid.split("_")
        assert len(parts) >= 2
        ts_part = parts[-1]
        assert ts_part.isdigit()


@pytest.fixture
def mock_executor():
    cfg = PairConfig(symbol="FET/USDT:USDT")
    gw = MagicMock()
    gw.create_exit_order = AsyncMock(return_value={"id": "exit_123"})
    gw.cancel_order = AsyncMock(return_value=True)

    tracker = MagicMock()
    quoter = MagicMock()
    quoter.quantize_amount = MagicMock(return_value=630.0)

    executor = TripleBarrierExecutor(
        config=cfg,
        position_side=PositionSide.SHORT,
        gateway=gw,
        position_tracker=tracker,
        quoter=quoter,
    )
    executor.state.remaining_qty = 630.0
    executor.state.active = True
    return executor


@pytest.mark.parametrize("purpose", list(OrderPurpose))
@pytest.mark.asyncio
async def test_market_exit_client_id_never_exceeds_36_chars(mock_executor, purpose):
    """Calling _execute_market_exit with any OrderPurpose generates cid <= 36 chars."""
    mock_executor.gateway.create_exit_order.reset_mock()
    mock_executor.state.remaining_qty = 630.0

    await mock_executor._execute_market_exit(purpose=purpose)

    assert mock_executor.gateway.create_exit_order.called
    call_kwargs = mock_executor.gateway.create_exit_order.call_args.kwargs
    cid = call_kwargs["client_order_id"]

    assert len(cid) <= 36, f"Client order ID '{cid}' exceeds 36 chars (len={len(cid)}) for purpose {purpose.value}"
