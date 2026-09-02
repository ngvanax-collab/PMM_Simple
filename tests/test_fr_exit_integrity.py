"""Tests for FR Engine Exit Integrity, Legging Rollback, and Symbol Canonicalization (Phase 3)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.core.fr_execution.engine import FRExecutionEngine
from app.core.fr_execution.gateway import MultiExchangeGateway
from app.core.fr_execution.kill_switch import ThreeTierKillSwitch
from app.core.fr_execution.models import FRAction, FRPolicy, FRRiskConfig
from app.core.fr_execution.position_tracker import FRPositionTracker
from app.models.config import ExchangeCredentials


@pytest.fixture
def mock_gateway():
    gateway = MultiExchangeGateway(
        binance_creds=ExchangeCredentials(exchange="binance", api_key="k1", api_secret="s1"),
        bybit_creds=ExchangeCredentials(exchange="bybit", api_key="k2", api_secret="s2"),
    )
    gateway._connected = {"binance": True, "bybit": True}
    gateway.market_info = {
        "binance": {
            "SOL/USDT:USDT": {
                "precision": {"price": 2, "amount": 2},
                "limits": {"price": {"min": 0.01}, "amount": {"min": 0.01}},
            },
        },
        "bybit": {
            "SOL/USDT:USDT": {
                "precision": {"price": 2, "amount": 2},
                "limits": {"price": {"min": 0.01}, "amount": {"min": 0.01}},
            },
        },
    }
    gateway.exchanges = {
        "binance": AsyncMock(),
        "bybit": AsyncMock(),
    }
    gateway.get_free_margin = AsyncMock(return_value=1000.0)
    gateway.fetch_ticker_price = AsyncMock(return_value=100.0)
    gateway.fetch_positions = AsyncMock(return_value=[])
    return gateway


@pytest.fixture
def position_tracker():
    return FRPositionTracker()


@pytest.fixture
def kill_switch():
    return ThreeTierKillSwitch()


@pytest.fixture
def execution_engine(mock_gateway, position_tracker, kill_switch):
    cfg = FRRiskConfig(
        max_leverage=5,
        allocated_margin_per_pair=200.0,
        min_expected_edge_bps=15.0,
        max_loss_usd=50.0,
        auto_execution_enabled=True,
    )
    return FRExecutionEngine(mock_gateway, position_tracker, kill_switch, cfg)


@pytest.mark.asyncio
async def test_legging_timeout_uses_actual_filled_qty(execution_engine, mock_gateway, position_tracker):
    """
    TASK H-3: When Long leg partially fills (1.8 / 2.0) on Binance, but Short leg hangs/times out on Bybit:
    - Long task completes with partial fill 1.8 (or exchange returns contracts=1.8)
    - Short task is cancelled on timeout (>5s)
    - Rollback emergency_market_close must be called with actual 1.8 (not 2.0).
    """
    policy = FRPolicy(
        policy_id="pol_legging_test",
        symbol="SOL/USDT:USDT",
        exchange_long="binance",
        exchange_short="bybit",
        action=FRAction.OPEN,
        target_notional_usdt=200.0,
    )

    async def mock_create_hedge(exchange, symbol, side, position_side, amount, **kwargs):
        if exchange == "binance":
            return {"id": "ord_bin_1", "status": "closed", "filled": 1.8}
        elif exchange == "bybit":
            # Simulate hanging order that times out
            await asyncio.sleep(10.0)
            return {"id": "ord_byb_1", "status": "closed"}
        return None

    mock_gateway.create_hedge_order = AsyncMock(side_effect=mock_create_hedge)
    mock_gateway.emergency_market_close = AsyncMock(return_value={"id": "rollback_closed"})

    # Mock exchange reporting actual 1.8 on Binance
    async def mock_fetch_pos(exchange, symbols=None):
        if exchange == "binance":
            return [{"symbol": "SOLUSDT", "positionSide": "LONG", "contracts": 1.8}]
        return []

    mock_gateway.fetch_positions = AsyncMock(side_effect=mock_fetch_pos)

    # Set fast timeout for test
    execution_engine.risk_config.order_timeout_sec = 0.05
    res = await execution_engine.execute_policy(policy)

    assert res["status"] == "LEGGING_ROLLBACK"
    assert res["actual_qty"] == 1.8
    mock_gateway.emergency_market_close.assert_called_once_with("binance", "SOL/USDT:USDT", "LONG", 1.8)

    pos = position_tracker.get_position("SOL/USDT:USDT")
    assert pos.status == "FLAT"


@pytest.mark.asyncio
async def test_exit_partial_failure_keeps_leg_state(execution_engine, mock_gateway, position_tracker):
    """
    TASK H-4: When closing a dual-leg position:
    - Long leg close succeeds
    - Short leg close raises Exception (e.g. Bybit network error)
    - Engine must NOT mark position FLAT or zero short leg size
    - Status becomes EXIT_PARTIAL and short_leg.size remains 5.0.
    """
    sym = "SOL/USDT:USDT"
    position_tracker.update_leg(sym, "binance", "LONG", size=5.0, entry_price=100.0, mark_price=100.0, upnl=0.0)
    position_tracker.update_leg(sym, "bybit", "SHORT", size=5.0, entry_price=100.0, mark_price=100.0, upnl=0.0)

    async def mock_emergency_close(exchange, symbol, position_side, amount):
        if exchange == "binance":
            return {"id": "bin_exit_ok", "status": "closed"}
        elif exchange == "bybit":
            raise RuntimeError("Bybit 500 Internal Error during exit")
        return None

    mock_gateway.emergency_market_close = AsyncMock(side_effect=mock_emergency_close)

    policy = FRPolicy(
        policy_id="pol_exit_partial",
        symbol=sym,
        exchange_long="binance",
        exchange_short="bybit",
        action=FRAction.EXIT,
        target_notional_usdt=0.0,
    )

    res = await execution_engine.execute_policy(policy)

    assert res["status"] == "EXIT_PARTIAL"
    assert res["long_closed"] is True
    assert res["short_closed"] is False

    pos = position_tracker.get_position(sym)
    assert pos.long_leg.size == 0.0
    assert pos.short_leg.size == 5.0
    assert pos.status == "CLOSING"


@pytest.mark.asyncio
async def test_reduce_partial_failure_keeps_leg_size(execution_engine, mock_gateway, position_tracker):
    """
    TASK H-4: When reducing a dual-leg position:
    - Long leg reduce succeeds (-2.0)
    - Short leg reduce raises Exception
    - Engine only reduces Long leg to 3.0; Short leg stays 5.0.
    """
    sym = "SOL/USDT:USDT"
    position_tracker.update_leg(sym, "binance", "LONG", size=5.0, entry_price=100.0, mark_price=100.0, upnl=0.0)
    position_tracker.update_leg(sym, "bybit", "SHORT", size=5.0, entry_price=100.0, mark_price=100.0, upnl=0.0)

    async def mock_create_hedge(exchange, symbol, side, position_side, amount, **kwargs):
        if exchange == "binance":
            return {"id": "bin_reduce_ok", "status": "closed"}
        elif exchange == "bybit":
            raise RuntimeError("Bybit reduce error")
        return None

    mock_gateway.create_hedge_order = AsyncMock(side_effect=mock_create_hedge)

    policy = FRPolicy(
        policy_id="pol_reduce_partial",
        symbol=sym,
        exchange_long="binance",
        exchange_short="bybit",
        action=FRAction.REDUCE,
        target_notional_usdt=500.0,
        reduce_to_notional_usdt=300.0,  # target size 3.0 -> reduce 2.0
    )

    res = await execution_engine.execute_policy(policy)

    assert res["status"] == "REDUCE_PARTIAL"
    assert res["long_reduced"] is True
    assert res["short_reduced"] is False

    pos = position_tracker.get_position(sym)
    assert pos.long_leg.size == pytest.approx(3.0, abs=1e-4)
    assert pos.short_leg.size == pytest.approx(5.0, abs=1e-4)


def test_reconcile_updates_policy_created_position(position_tracker):
    """
    TASK M-2: Symbol canonicalization in FR Position Tracker:
    - Policy creates position with unified key 'SOL/USDT:USDT'
    - REST reconcile returns raw Binance symbol 'SOLUSDT' and raw Bybit symbol 'SOLUSDT'
    - Reconcile must update the SAME position object without creating duplicate keys.
    """
    sym_canonical = "SOL/USDT:USDT"
    # Seed position created by policy
    dual_pos = position_tracker.get_or_create_position(sym_canonical, ex_long="binance", ex_short="bybit")
    assert len(position_tracker.positions) == 1

    # Simulate raw REST snapshots from exchanges
    binance_raw = [
        {
            "symbol": "SOLUSDT",
            "positionSide": "LONG",
            "contracts": "2.5",
            "entryPrice": "100.5",
            "markPrice": "102.0",
            "unrealizedPnl": "3.75",
            "leverage": "5",
        }
    ]
    bybit_raw = [
        {
            "symbol": "SOLUSDT",
            "positionIdx": 2,
            "size": "2.5",
            "avgPrice": "100.8",
            "markPrice": "102.0",
            "unrealisedPnl": "-3.00",
            "leverage": "5",
        }
    ]

    position_tracker.reconcile_with_exchange_positions(binance_raw, bybit_raw)

    # Must STILL have only 1 position in tracker
    assert len(position_tracker.positions) == 1
    updated_pos = position_tracker.get_position(sym_canonical)
    assert updated_pos is not None
    assert updated_pos.long_leg.size == 2.5
    assert updated_pos.long_leg.entry_price == 100.5
    assert updated_pos.short_leg.size == 2.5
    assert updated_pos.short_leg.entry_price == 100.8
    assert updated_pos.status == "OPEN"


@pytest.mark.asyncio
async def test_legging_measurement_failure_trips_kill_switch(execution_engine, mock_gateway, position_tracker, kill_switch):
    """
    TASK F-2: Fail-closed leg measurement:
    When Long leg completes, Short leg hangs/times out, but REST fetch_positions raises Exception:
    - Engine must NOT blindly rollback
    - Must NOT mark position FLAT
    - Must trip exchange kill switch
    - Returns status MEASUREMENT_FAILED
    """
    policy = FRPolicy(
        policy_id="pol_legging_fail_test",
        symbol="SOL/USDT:USDT",
        exchange_long="binance",
        exchange_short="bybit",
        action=FRAction.OPEN,
        target_notional_usdt=200.0,
    )

    async def mock_create_hedge(exchange, symbol, side, position_side, amount, **kwargs):
        if exchange == "binance":
            return {"id": "ord_bin_1", "status": "closed", "filled": 1.8}
        elif exchange == "bybit":
            await asyncio.sleep(10.0)
            return {"id": "ord_byb_1", "status": "closed"}
        return None

    mock_gateway.create_hedge_order = AsyncMock(side_effect=mock_create_hedge)
    mock_gateway.emergency_market_close = AsyncMock()

    # Simulate fetch_positions raising an exception (e.g. REST failure / network partition)
    mock_gateway.fetch_positions = AsyncMock(side_effect=RuntimeError("Binance REST 502 Bad Gateway"))

    execution_engine.risk_config.order_timeout_sec = 0.05
    res = await execution_engine.execute_policy(policy)

    assert res["status"] == "MEASUREMENT_FAILED"
    assert kill_switch.is_exchange_tripped("binance") is True
    pos = position_tracker.get_position("SOL/USDT:USDT")
    assert pos.status != "FLAT"
    mock_gateway.emergency_market_close.assert_not_called()


@pytest.mark.asyncio
async def test_funding_polling_records_deduped_payments(mock_gateway, position_tracker, kill_switch):
    """
    TASK F-3: Funding accrual polling in FRManager._reconciliation_loop:
    - Periodically polls fetch_funding_history for active pairs
    - Accrues real funding payments into tracker.realized_funding_pnl
    - Deduplicates identical funding transaction IDs across polling intervals.
    """
    from app.core.fr_execution.manager import FRManager

    manager = FRManager()
    manager.gateway = mock_gateway
    manager.tracker = position_tracker
    manager.kill_switch = kill_switch
    manager._is_running = True

    # Seed an active position in tracker
    sym = "SOL/USDT:USDT"
    position_tracker.update_leg(sym, "binance", "LONG", size=2.0, entry_price=100.0, mark_price=100.0, upnl=0.0)
    position_tracker.update_leg(sym, "bybit", "SHORT", size=2.0, entry_price=100.0, mark_price=100.0, upnl=0.0)
    assert len(position_tracker.get_active_positions()) == 1

    # Mock gateway funding history
    funding_entry = {
        "id": "tx_1",
        "symbol": "SOL/USDT:USDT",
        "timestamp": 1700000000000,
        "info": {"tranId": "tx_1", "income": "2.5", "positionSide": "LONG"},
        "amount": 2.5,
    }
    mock_gateway.fetch_funding_history = AsyncMock(return_value=[funding_entry])

    # Run 1 reconciliation beat (stopping after 1 loop)
    async def stop_after_one_beat(*args, **kwargs):
        manager._is_running = False

    with patch("asyncio.sleep", side_effect=stop_after_one_beat):
        await manager._reconciliation_loop()

    assert pytest.approx(position_tracker.realized_funding_pnl, abs=1e-5) == 2.5

    # Run second beat with the same funding history -> should be deduplicated (still 2.5)
    manager._is_running = True
    with patch("asyncio.sleep", side_effect=stop_after_one_beat):
        await manager._reconciliation_loop()

    assert pytest.approx(position_tracker.realized_funding_pnl, abs=1e-5) == 2.5
