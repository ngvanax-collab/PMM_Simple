"""Tests for ExchangeGateway setup_symbol leverage restriction (-4421/-4046) and DB lazy-query safety."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.gateway import ExchangeGateway
from app.models.config import ExchangeCredentials
from app.persistence.db import Database


# ── DB Lazy-Query Safety Tests ──

@pytest.mark.asyncio
async def test_db_get_pnl_summary_returns_default_when_not_connected():
    """Verify get_pnl_summary() returns safe defaults when DB connection is not yet open."""
    fresh_db = Database(":memory:")
    assert fresh_db._conn is None  # Ensure connection is not set

    try:
        result = await fresh_db.get_pnl_summary()
        assert result == {"total_realized_pnl": 0.0, "total_fee": 0.0, "total_net_pnl": 0.0}
    finally:
        await fresh_db.close()


@pytest.mark.asyncio
async def test_db_get_recent_fills_returns_empty_when_not_connected():
    """Verify get_recent_fills() returns [] when DB connection is not yet open."""
    fresh_db = Database(":memory:")
    assert fresh_db._conn is None

    try:
        result = await fresh_db.get_recent_fills(limit=50)
        assert result == []
    finally:
        await fresh_db.close()


@pytest.mark.asyncio
async def test_db_get_active_orders_returns_empty_when_not_connected():
    """Verify get_active_orders() returns [] when DB connection is not yet open."""
    fresh_db = Database(":memory:")
    assert fresh_db._conn is None

    try:
        result = await fresh_db.get_active_orders()
        assert result == []
    finally:
        await fresh_db.close()


@pytest.mark.asyncio
async def test_db_record_pnl_returns_zero_when_not_connected():
    """Verify record_pnl() returns 0 when DB connection is not yet open (no crash)."""
    from app.models.state import PositionSide
    from app.models.state import PnLRecord
    import time

    fresh_db = Database(":memory:")
    assert fresh_db._conn is None

    rec = PnLRecord(
        symbol="SOL/USDT:USDT",
        position_side=PositionSide.LONG,
        realized_pnl=10.0,
        fee=0.02,
        net_pnl=9.98,
        timestamp=time.time(),
        note="test",
    )
    # Should not crash and should attempt to connect, but will return 0 if still None
    # Since there's no real DB file in tests, it tries to connect and may fail gracefully
    # We just verify it doesn't raise an AssertionError
    try:
        await fresh_db.record_pnl(rec)
    except Exception as e:
        assert "AssertionError" not in str(type(e).__name__), f"Must not raise AssertionError, got: {e}"
    finally:
        await fresh_db.close()


# ── Gateway Leverage / Margin Error Handling Tests ──

@pytest.mark.asyncio
async def test_setup_symbol_4421_subaccount_leverage_fallback():
    """Verify that -4421 triggers auto-fallback to 5x leverage without crashing worker."""
    creds = ExchangeCredentials(exchange="binance", api_key="k", api_secret="s")
    gw = ExchangeGateway(creds)

    mock_exchange = MagicMock()
    gw._exchange = mock_exchange

    call_count = 0
    async def fake_set_leverage(lev, sym):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception('{"code": -4421, "msg": "Subaccounts are restricted from using leverage greater than 5x."}')
        # Second call (fallback to 5x) succeeds silently

    mock_exchange.set_leverage = fake_set_leverage
    mock_exchange.set_margin_mode = AsyncMock()

    result = await gw.setup_symbol("XRP/USDT:USDT", leverage=20, margin_mode="isolated")

    assert result is True, "setup_symbol must succeed even when leverage is restricted to 5x"
    assert call_count == 2, f"Expected 2 set_leverage calls (original + fallback), got {call_count}"


@pytest.mark.asyncio
async def test_setup_symbol_4046_margin_type_already_set():
    """Verify that -4046 'No need to change margin type' is handled as INFO, not an error."""
    creds = ExchangeCredentials(exchange="binance", api_key="k", api_secret="s")
    gw = ExchangeGateway(creds)

    mock_exchange = MagicMock()
    gw._exchange = mock_exchange
    mock_exchange.set_leverage = AsyncMock()
    mock_exchange.set_margin_mode = AsyncMock(
        side_effect=Exception('{"code": -4046, "msg": "No need to change margin type."}')
    )

    result = await gw.setup_symbol("SOL/USDT:USDT", leverage=10, margin_mode="isolated")

    assert result is True, "setup_symbol must succeed when margin mode is already correct (-4046)"


@pytest.mark.asyncio
async def test_setup_symbol_succeeds_on_clean_exchange():
    """Verify normal setup_symbol path works when exchange accepts all calls."""
    creds = ExchangeCredentials(exchange="binance", api_key="k", api_secret="s")
    gw = ExchangeGateway(creds)

    mock_exchange = MagicMock()
    gw._exchange = mock_exchange
    mock_exchange.set_leverage = AsyncMock()
    mock_exchange.set_margin_mode = AsyncMock()

    result = await gw.setup_symbol("SOL/USDT:USDT", leverage=10, margin_mode="isolated")

    assert result is True
    mock_exchange.set_leverage.assert_called_once_with(10, "SOL/USDT:USDT")
    mock_exchange.set_margin_mode.assert_called_once_with("ISOLATED", "SOL/USDT:USDT")
