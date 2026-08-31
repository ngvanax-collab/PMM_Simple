"""Funding Rate Arbitrage Execution Layer Package (Dual-Exchange Hedge Mode)."""
from .models import DualLegPosition, FRAction, FRPolicy, FRRiskConfig, FRSummaryMetrics, LegPositionState
from .gateway import MultiExchangeGateway
from .position_tracker import FRPositionTracker
from .kill_switch import ThreeTierKillSwitch
from .engine import FRExecutionEngine
from .manager import FRManager, fr_manager

__all__ = [
    "DualLegPosition",
    "FRAction",
    "FRPolicy",
    "FRRiskConfig",
    "FRSummaryMetrics",
    "LegPositionState",
    "MultiExchangeGateway",
    "FRPositionTracker",
    "ThreeTierKillSwitch",
    "FRExecutionEngine",
    "FRManager",
    "fr_manager",
]
