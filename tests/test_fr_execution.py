"""Comprehensive Test Suite for Funding Rate Arbitrage Execution Engine & Hedge Mode Invariants."""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.fr_execution.engine import FRExecutionEngine
from app.core.fr_execution.gateway import MultiExchangeGateway
from app.core.fr_execution.kill_switch import ThreeTierKillSwitch
from app.core.fr_execution.models import FRAction, FRPolicy, FRRiskConfig
from app.core.fr_execution.position_tracker import FRPositionTracker
from app.models.config import ExchangeCredentials


@pytest.fixture
def mock_gateway():
    """Create a mock MultiExchangeGateway with simulated market info and balances."""
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
            "BTC/USDT:USDT": {
                "precision": {"price": 1, "amount": 3},
                "limits": {"price": {"min": 0.1}, "amount": {"min": 0.001}},
            },
        },
        "bybit": {
            "SOL/USDT:USDT": {
                "precision": {"price": 2, "amount": 2},
                "limits": {"price": {"min": 0.01}, "amount": {"min": 0.01}},
            },
            "BTC/USDT:USDT": {
                "precision": {"price": 1, "amount": 3},
                "limits": {"price": {"min": 0.1}, "amount": {"min": 0.001}},
            },
        },
    }

    # Mock exchange instances
    mock_binance = AsyncMock()
    mock_bybit = AsyncMock()
    gateway.exchanges = {
        "binance": mock_binance,
        "bybit": mock_bybit,
    }

    # Default balances
    gateway.get_free_margin = AsyncMock(return_value=1000.0)
    gateway.fetch_ticker_price = AsyncMock(return_value=100.0)
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


# ── Test 1: Dual-Leg Order Generation & Precision ──

@pytest.mark.asyncio
async def test_dual_leg_order_generation(mock_gateway):
    """
    Test dual-leg order generation:
    - Long Leg on Binance: side=BUY, positionSide=LONG.
    - Short Leg on Bybit: side=SELL, positionSide=SHORT, positionIdx=2.
    - Quantization and precision format correctly.
    """
    # Simulate Binance and Bybit order creation
    mock_gateway.exchanges["binance"].create_order = AsyncMock(return_value={"id": "bin_123", "status": "closed"})
    mock_gateway.exchanges["bybit"].create_order = AsyncMock(return_value={"id": "byb_456", "status": "closed"})

    # 1. Test Binance Long Leg
    res_bin = await mock_gateway.create_hedge_order(
        exchange="binance",
        symbol="SOL/USDT:USDT",
        side="buy",
        position_side="LONG",
        amount=2.3456,
        order_type="market",
    )
    assert res_bin is not None
    mock_gateway.exchanges["binance"].create_order.assert_called_once()
    bin_call_kwargs = mock_gateway.exchanges["binance"].create_order.call_args.kwargs
    assert bin_call_kwargs["side"] == "buy"
    assert bin_call_kwargs["amount"] == 2.34  # Floored by precision 2
    assert bin_call_kwargs["params"]["positionSide"] == "LONG"
    assert "reduceOnly" not in bin_call_kwargs["params"]

    # 2. Test Bybit Short Leg
    res_byb = await mock_gateway.create_hedge_order(
        exchange="bybit",
        symbol="SOL/USDT:USDT",
        side="sell",
        position_side="SHORT",
        amount=2.3456,
        order_type="market",
    )
    assert res_byb is not None
    mock_gateway.exchanges["bybit"].create_order.assert_called_once()
    byb_call_kwargs = mock_gateway.exchanges["bybit"].create_order.call_args.kwargs
    assert byb_call_kwargs["side"] == "sell"
    assert byb_call_kwargs["amount"] == 2.34
    assert byb_call_kwargs["params"]["positionSide"] == "SHORT"
    assert byb_call_kwargs["params"]["positionIdx"] == 2
    assert "reduceOnly" not in byb_call_kwargs["params"]


# ── Test 2: Legging Risk Rollback ──

@pytest.mark.asyncio
async def test_legging_risk_rollback(execution_engine, mock_gateway, position_tracker):
    """
    Test legging risk protection:
    If Long leg succeeds on Binance but Short leg fails on Bybit (error/timeout),
    the system must immediately execute a reverse MARKET order to close the Long leg,
    bringing the account back to 100% Flat.
    """
    policy = FRPolicy(
        policy_id="test_pol_open_01",
        symbol="SOL/USDT:USDT",
        exchange_long="binance",
        exchange_short="bybit",
        action=FRAction.OPEN,
        target_notional_usdt=200.0,
        expected_net_edge_bps=25.0,
    )

    # Long leg on Binance succeeds, Short leg on Bybit raises Exception
    mock_gateway.create_hedge_order = AsyncMock()
    
    async def mock_hedge_order(exchange, symbol, side, position_side, amount, **kwargs):
        if exchange == "binance" and side == "buy":
            return {"id": "bin_fill", "status": "closed"}
        elif exchange == "bybit":
            raise RuntimeError("Bybit WebSocket timeout / Gateway connection error")
        elif exchange == "binance" and side == "sell":
            # Rollback market order
            return {"id": "bin_rollback_fill", "status": "closed"}
        return None

    mock_gateway.create_hedge_order.side_effect = mock_hedge_order
    mock_gateway.emergency_market_close = AsyncMock(return_value={"id": "bin_rollback_closed"})

    res = await execution_engine.execute_policy(policy)

    assert res["status"] == "LEGGING_ROLLBACK"
    assert res["rollback"] is True
    # Verify emergency_market_close was called for Binance LONG
    mock_gateway.emergency_market_close.assert_called_once_with("binance", "SOL/USDT:USDT", "LONG", 2.0)

    # Verify position is FLAT in tracker
    pos = position_tracker.get_position("SOL/USDT:USDT")
    assert pos is not None
    assert pos.status == "FLAT"


# ── Test 3: Exit without reduceOnly ──

@pytest.mark.asyncio
async def test_exit_without_reduce_only(execution_engine, mock_gateway, position_tracker):
    """
    Ensure all exit orders:
    - Never use reduceOnly=True or closePosition=True
    - Close Long: side=SELL, positionSide=LONG
    - Close Short: side=BUY, positionSide=SHORT
    """
    sym = "SOL/USDT:USDT"
    # Seed an open position
    position_tracker.update_leg(sym, "binance", "LONG", size=5.0, entry_price=100.0, mark_price=102.0, upnl=10.0)
    position_tracker.update_leg(sym, "bybit", "SHORT", size=5.0, entry_price=100.0, mark_price=101.0, upnl=-5.0)

    mock_gateway.emergency_market_close = AsyncMock(return_value={"status": "closed"})

    policy = FRPolicy(
        policy_id="test_pol_exit_01",
        symbol=sym,
        exchange_long="binance",
        exchange_short="bybit",
        action=FRAction.EXIT,
        target_notional_usdt=0.0,
    )

    res = await execution_engine.execute_policy(policy)

    assert res["status"] == "EXIT_COMPLETED"
    # Check calls to emergency_market_close
    calls = mock_gateway.emergency_market_close.call_args_list
    assert len(calls) == 2

    # Long leg exit: sell LONG
    bin_call = calls[0].args
    assert bin_call[0] == "binance"
    assert bin_call[1] == sym
    assert bin_call[2] == "LONG"
    assert bin_call[3] == 5.0

    # Short leg exit: buy SHORT
    byb_call = calls[1].args
    assert byb_call[0] == "bybit"
    assert byb_call[1] == sym
    assert byb_call[2] == "SHORT"
    assert byb_call[3] == 5.0

    pos = position_tracker.get_position(sym)
    assert pos.status == "FLAT"
    assert pos.long_leg.size == 0.0
    assert pos.short_leg.size == 0.0


# ── Test 4: Decision Policy Consumption & Action Dispatching ──

@pytest.mark.asyncio
async def test_decision_policy_consumption(execution_engine, mock_gateway, position_tracker):
    """
    Test parsing response from Decision Layer (GET /api/v1/fr/policies) and handling
    OPEN, HOLD, REDUCE, EXIT, and PAUSE actions.
    """
    mock_payload = {
        "generated_at": "2026-08-23T05:00:00Z",
        "items": [
            {
                "policy_id": "pol_open_btc",
                "strategy_version": "v3",
                "risk_policy_version": "v1",
                "symbol": "BTC/USDT:USDT",
                "exchange_long": "binance",
                "exchange_short": "bybit",
                "action": "OPEN",
                "confidence": 0.95,
                "target_notional_usdt": 500.0,
                "max_leverage": 5,
                "expected_net_edge_bps": 20.5,
                "reason_codes": ["STRONG_POSITIVE_SPREAD"],
            },
            {
                "policy_id": "pol_pause_sol",
                "strategy_version": "v3",
                "risk_policy_version": "v1",
                "symbol": "SOL/USDT:USDT",
                "exchange_long": "binance",
                "exchange_short": "bybit",
                "action": "PAUSE",
                "confidence": 1.0,
                "target_notional_usdt": 0.0,
                "reason_codes": ["VOLATILITY_SPIKE"],
            }
        ]
    }

    mock_gateway.create_hedge_order = AsyncMock(return_value={"id": "order_ok"})
    mock_gateway.cancel_all_orders = AsyncMock(return_value=True)

    # Mock HTTP client response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json = MagicMock(return_value=mock_payload)

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)

    results = await execution_engine.fetch_and_execute_policies(client=mock_client)

    assert len(results) == 2
    assert results[0]["status"] == "SUCCESS"
    assert results[0]["action"] == "OPEN"
    assert results[0]["symbol"] == "BTC/USDT:USDT"

    assert results[1]["status"] == "PAUSED"
    assert results[1]["symbol"] == "SOL/USDT:USDT"


# ── Test 5: Three-Tier Kill Switch Hierarchy ──

@pytest.mark.asyncio
async def test_three_tier_kill_switch_hierarchy(kill_switch, execution_engine):
    """
    Test 3-Tier Kill Switch:
    - Tier 1: Symbol Pause blocks only that symbol.
    - Tier 2: Exchange Trip blocks orders involving that exchange.
    - Tier 3: Global Trip blocks all executions.
    """
    pol_btc = FRPolicy(
        policy_id="p1", symbol="BTC/USDT:USDT", exchange_long="binance", exchange_short="bybit",
        action=FRAction.OPEN, target_notional_usdt=100.0
    )
    pol_sol = FRPolicy(
        policy_id="p2", symbol="SOL/USDT:USDT", exchange_long="binance", exchange_short="bybit",
        action=FRAction.OPEN, target_notional_usdt=100.0
    )

    # 1. Tier 1: Pause SOL
    kill_switch.pause_symbol("SOL/USDT:USDT")
    assert kill_switch.is_symbol_paused("SOL/USDT:USDT") is True
    assert kill_switch.is_symbol_paused("BTC/USDT:USDT") is False

    res_sol = await execution_engine.execute_policy(pol_sol)
    assert res_sol["status"] == "BLOCKED"

    # 2. Tier 2: Trip Binance
    kill_switch.trip_exchange("binance", "API connection dropped")
    allowed, reason = kill_switch.is_execution_allowed(exchange="binance")
    assert allowed is False
    assert "Binance" in reason or "BINANCE" in reason

    # 3. Tier 3: Trip Global
    kill_switch.trip_global("Emergency System Halt")
    allowed_global, reason_global = kill_switch.is_execution_allowed()
    assert allowed_global is False
    assert "Global Kill Switch active" in reason_global

    # Reset Global
    kill_switch.reset_global()
    assert kill_switch.is_global_tripped is False


# ── Test 6: Emergency Kill-All 6-Phase Execution ──

@pytest.mark.asyncio
async def test_emergency_kill_all_six_phases(kill_switch, mock_gateway, position_tracker):
    """
    Test 6-Phase Emergency Kill-All:
    1. Lock & Trip Global.
    2. Cancel Binance orders.
    3. Cancel Bybit orders.
    4. Close Binance open positions.
    5. Close Bybit open positions.
    6. Reconcile & confirm 0 remaining positions.
    """
    mock_gateway.cancel_all_orders = AsyncMock(return_value=True)
    mock_gateway.emergency_market_close = AsyncMock(return_value={"id": "closed"})

    # Set initial open positions on both exchanges
    mock_gateway.fetch_positions = AsyncMock(side_effect=[
        # Phase 4 (Binance):
        [{"symbol": "SOL/USDT:USDT", "positionAmt": 5.0, "side": "LONG"}],
        # Phase 5 (Bybit):
        [{"symbol": "SOL/USDT:USDT", "size": 5.0, "positionIdx": 2}],
        # Phase 6 (Binance verify):
        [],
        # Phase 6 (Bybit verify):
        [],
    ])

    report = await kill_switch.emergency_kill_all(mock_gateway, position_tracker)

    assert report["status"] == "COMPLETED"
    assert report["phases"]["phase_1_global_lock"] == "COMPLETED"
    assert report["phases"]["phase_2_cancel_binance_orders"] == "SUCCESS"
    assert report["phases"]["phase_3_cancel_bybit_orders"] == "SUCCESS"
    assert report["phases"]["phase_6_reconcile"]["is_flat"] is True


# ── Test 7: Position Tracker PnL & Funding Metrics ──

def test_fr_position_tracker_metrics(position_tracker):
    """
    Test Position Tracker recalculations for uPnL, accrued funding, holding duration, and APR summary.
    """
    sym = "SOL/USDT:USDT"
    position_tracker.update_leg(sym, "binance", "LONG", size=10.0, entry_price=100.0, mark_price=105.0, upnl=50.0)
    position_tracker.update_leg(sym, "bybit", "SHORT", size=10.0, entry_price=100.0, mark_price=104.0, upnl=-40.0)
    position_tracker.record_funding_payment(sym, "binance", "LONG", payment_amount=-2.5)
    position_tracker.record_funding_payment(sym, "bybit", "SHORT", payment_amount=15.0)

    pos = position_tracker.get_position(sym)
    assert pos is not None
    assert pos.status == "OPEN"
    assert pos.net_upnl == 10.0  # +50 - 40
    assert pos.total_funding_accrued == 12.5  # -2.5 + 15.0
    assert pos.net_pnl == 22.5  # +10 + 12.5

    # Summary metrics
    metrics = position_tracker.get_summary_metrics(binance_free_margin=500.0, bybit_free_margin=600.0)
    assert metrics.total_realized_funding_pnl == 12.5
    assert metrics.active_arb_pairs == 1
    assert metrics.binance_free_margin == 500.0
    assert metrics.bybit_free_margin == 600.0
    assert metrics.total_equity_usdt == 1100.0
