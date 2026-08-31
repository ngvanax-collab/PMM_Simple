"""Models and schemas for Funding Rate Arbitrage Execution Engine (SIGNAL_CONTRACT_V2 compliant)."""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FRAction(str, Enum):
    """Permitted execution actions from Decision Layer."""
    OPEN = "OPEN"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    PAUSE = "PAUSE"


class FRPolicy(BaseModel):
    """Deterministic trading policy contract from Decision Layer (SIGNAL_CONTRACT_V2)."""
    policy_id: str
    strategy_version: str = "v3"
    risk_policy_version: str = "v1"
    symbol: str
    exchange_long: str
    exchange_short: str
    action: FRAction
    confidence: float = 1.0
    target_notional_usdt: float = 0.0
    reduce_to_notional_usdt: Optional[float] = None
    max_leverage: int = 5
    max_holding_hours: int = 72
    expected_net_funding_bps: float = 0.0
    expected_fee_bps: float = 0.0
    expected_slippage_bps: float = 0.0
    expected_net_edge_bps: float = 0.0
    rotation_priority: int = 1
    risk_exit_allowed: bool = True
    score_1d: Optional[float] = None
    score_7d: Optional[float] = None
    score_30d: Optional[float] = None
    reason_codes: List[str] = Field(default_factory=list)
    generated_at: Optional[datetime] = None
    expiry_at: Optional[datetime] = None


class LegPositionState(BaseModel):
    """Position state for one leg of the arbitrage pair."""
    exchange: str  # "binance" or "bybit"
    symbol: str
    position_side: str  # "LONG" or "SHORT"
    entry_price: float = 0.0
    mark_price: float = 0.0
    size: float = 0.0
    notional: float = 0.0
    unrealized_pnl: float = 0.0
    funding_accrued: float = 0.0
    liquidation_price: Optional[float] = 0.0
    leverage: int = 1
    entry_time: float = 0.0
    last_updated: float = 0.0


class DualLegPosition(BaseModel):
    """Dual-leg position tracking for an arbitrage pair."""
    symbol: str
    long_leg: LegPositionState
    short_leg: LegPositionState
    net_upnl: float = 0.0
    total_funding_accrued: float = 0.0
    net_pnl: float = 0.0
    holding_duration_hours: float = 0.0
    is_paused: bool = False
    status: str = "FLAT"  # FLAT, OPENING, OPEN, CLOSING, CLOSED
    last_policy_action: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def recalculate(self) -> None:
        """Compute aggregated metrics across both legs."""
        self.net_upnl = round(self.long_leg.unrealized_pnl + self.short_leg.unrealized_pnl, 4)
        self.total_funding_accrued = round(self.long_leg.funding_accrued + self.short_leg.funding_accrued, 4)
        self.net_pnl = round(self.net_upnl + self.total_funding_accrued, 4)
        
        if self.created_at > 0 and self.status in ("OPEN", "OPENING"):
            import time
            self.holding_duration_hours = round((time.time() - self.created_at) / 3600.0, 2)
        else:
            self.holding_duration_hours = 0.0


class FRRiskConfig(BaseModel):
    """Risk and runtime parameters for Funding Rate Arbitrage."""
    max_leverage: int = Field(default=5, ge=1, le=10, description="Max leverage cap (<=5x recommended)")
    allocated_margin_per_pair: float = Field(default=200.0, ge=10.0, description="Margin USDT allocated per pair")
    min_expected_edge_bps: float = Field(default=15.0, ge=0.0, description="Min expected net spread in basis points")
    max_loss_usd: float = Field(default=50.0, ge=1.0, description="Stop-loss net loss threshold per pair in USDT")
    decision_layer_url: str = Field(default="http://localhost:8102", description="Decision Layer FastAPI URL")
    poll_interval_sec: float = Field(default=10.0, ge=2.0, le=300.0, description="Policy polling interval in seconds")
    auto_execution_enabled: bool = Field(default=True, description="Enable automatic policy execution")


class FRSummaryMetrics(BaseModel):
    """Aggregated portfolio summary metrics for NiceGUI banner."""
    total_realized_funding_pnl: float = 0.0
    net_arbitrage_apr: float = 0.0
    active_arb_pairs: int = 0
    binance_free_margin: float = 0.0
    bybit_free_margin: float = 0.0
    total_equity_usdt: float = 0.0
    last_reconciled_at: float = 0.0
