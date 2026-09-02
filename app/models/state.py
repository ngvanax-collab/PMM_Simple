"""State and Record Models using Pydantic v2."""
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PositionSide(str, Enum):
    """Position side for Hedge Mode."""
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"  # Note: One-way only, rejected in hedge operations


class OrderSide(str, Enum):
    """Order trade side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""
    LIMIT = "LIMIT"
    LIMIT_MAKER = "LIMIT_MAKER"
    MARKET = "MARKET"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


class OrderStatus(str, Enum):
    """Order lifecycle status."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderPurpose(str, Enum):
    """Business purpose of the order."""
    ENTRY_QUOTE = "ENTRY_QUOTE"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_TAKE_PROFIT = "TRAILING_TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    TIME_LIMIT_EXIT = "TIME_LIMIT_EXIT"
    PASSIVE_TIME_LIMIT_EXIT = "PASSIVE_TIME_LIMIT_EXIT"
    KILL_ALL_EXIT = "KILL_ALL_EXIT"
    CIRCUIT_BREAKER_EXIT = "CIRCUIT_BREAKER_EXIT"
    MANUAL_EXIT = "MANUAL_EXIT"


class OrderRecord(BaseModel):
    """Record of an active or historical order."""
    id: str = Field(..., description="Unique internal order ID or client_order_id")
    client_order_id: str = Field(..., description="Client order ID sent to exchange")
    exchange_order_id: Optional[str] = Field(default=None, description="Exchange order ID if acknowledged")
    symbol: str = Field(..., description="Unified symbol, e.g. SOL/USDT:USDT")
    side: OrderSide = Field(..., description="BUY or SELL")
    position_side: PositionSide = Field(..., description="HEDGE position side: LONG or SHORT")
    order_type: OrderType = Field(..., description="LIMIT, LIMIT_MAKER, MARKET, STOP_MARKET")
    price: float = Field(..., description="Order price (or trigger price for STOP_MARKET)")
    stop_price: Optional[float] = Field(default=None, description="Stop trigger price for conditional orders")
    amount: float = Field(..., description="Order quantity in base asset")
    filled_amount: float = Field(default=0.0, description="Cumulative executed quantity")
    remaining_amount: float = Field(..., description="Remaining quantity to execute")
    status: OrderStatus = Field(default=OrderStatus.NEW, description="Order status")
    purpose: OrderPurpose = Field(default=OrderPurpose.ENTRY_QUOTE, description="Order business purpose")
    level: int = Field(default=0, description="Quote level (0 = base, 1 = layer 1, etc.)")
    created_at: float = Field(..., description="Unix timestamp of order creation")
    updated_at: float = Field(..., description="Unix timestamp of last status update")
    raw_response: Optional[Dict[str, Any]] = Field(default=None, description="Raw exchange payload")


class FillRecord(BaseModel):
    """Record of a trade fill event."""
    id: str = Field(..., description="Unique fill ID (trade id)")
    order_id: str = Field(..., description="Associated exchange order ID")
    client_order_id: Optional[str] = Field(default=None, description="Client order ID")
    symbol: str = Field(..., description="Unified symbol")
    side: OrderSide = Field(..., description="BUY or SELL")
    position_side: PositionSide = Field(..., description="LONG or SHORT")
    price: float = Field(..., description="Execution price")
    amount: float = Field(..., description="Executed quantity")
    quote_amount: float = Field(..., description="Price * amount in USDT")
    fee: float = Field(default=0.0, description="Trade fee paid")
    fee_currency: str = Field(default="USDT", description="Fee currency")
    is_maker: bool = Field(default=True, description="True if maker execution")
    timestamp: float = Field(..., description="Unix timestamp of fill")
    realized_pnl: float = Field(default=0.0, description="Realized PnL reported by exchange")


class SidePositionState(BaseModel):
    """State for one slot of a pair (LONG or SHORT)."""
    symbol: str = Field(..., description="Symbol")
    position_side: PositionSide = Field(..., description="LONG or SHORT")
    amount: float = Field(default=0.0, ge=0.0, description="Gross position size in base currency (always >= 0)")
    entry_price: float = Field(default=0.0, ge=0.0, description="Weighted average entry price")
    current_price: float = Field(default=0.0, ge=0.0, description="Current mark / mid price")
    notional: float = Field(default=0.0, ge=0.0, description="Position notional value in USDT (amount * price)")
    unrealized_pnl: float = Field(default=0.0, description="Unrealized PnL in USDT")
    leverage: int = Field(default=5, description="Position leverage")
    initial_margin: float = Field(default=0.0, description="Required initial margin")
    last_fill_time: float = Field(default=0.0, description="Timestamp of last fill")
    last_sl_time: float = Field(default=0.0, description="Timestamp of last stop-loss execution")
    in_cooldown: bool = Field(default=False, description="True if currently in post-SL cooldown")
    consecutive_sl_count: int = Field(default=0, description="Number of consecutive stop-loss hits")
    cooldown_until: float = Field(default=0.0, description="Timestamp when current cooldown expires")
    filled_levels_count: int = Field(default=0, ge=0, description="Number of order levels currently filled (0, 1, 2, 3...)")
    last_level_fill_time: float = Field(default=0.0, description="Timestamp of most recent level fill")
    next_allowed_level_time: float = Field(default=0.0, description="Timestamp when next level quoting is permitted")
    is_trend_blocked: bool = Field(default=False, description="Flag indicating entry quote is blocked by trend bias")
    pyramid_filled_count: int = Field(default=0, ge=0, description="Number of favorable pyramid fills in current position cycle (max 1)")
    is_guaranteed_sl_locked: bool = Field(default=False, description="Flag confirming Stop Loss is locked to guaranteed breakeven/profit")
    trend_bias_regime: str = Field(default="NEUTRAL", description="Trend regime: BULLISH, BEARISH, NEUTRAL")


class ExecutorBarrierState(BaseModel):
    """Runtime state of the Triple-Barrier Executor for a specific positionSide."""
    symbol: str = Field(..., description="Symbol")
    position_side: PositionSide = Field(..., description="LONG or SHORT")
    active: bool = Field(default=False, description="Whether position is open and barrier is actively managing")
    entry_price: float = Field(default=0.0, description="Average entry price")
    initial_entry_price: float = Field(default=0.0, description="Initial entry price before pyramiding")
    total_qty: float = Field(default=0.0, description="Total initial quantity of position cycle")
    initial_qty: float = Field(default=0.0, description="Initial quantity before pyramiding")
    remaining_qty: float = Field(default=0.0, description="Remaining unclosed quantity")
    tp_orders: List[Dict[str, Any]] = Field(default_factory=list, description="Active TP orders: [{order_id, price, qty, status, level}]")
    sl_order_id: Optional[str] = Field(default=None, description="Active server-side STOP_MARKET order ID if any")
    sl_price: float = Field(default=0.0, description="Trigger price for Virtual Stop Loss")
    sl_qty: float = Field(default=0.0, description="Quantity for stop loss")
    trailing_active: bool = Field(default=False, description="Whether trailing stop activation price was crossed")
    trailing_high_watermark: float = Field(default=0.0, description="Extreme price achieved after activation")
    trailing_tp_active: bool = Field(default=False, description="Whether Dynamic Trailing Take Profit is active")
    peak_price: float = Field(default=0.0, description="Peak price tracked for LONG Trailing Take Profit")
    trough_price: float = Field(default=float('inf'), description="Trough price tracked for SHORT Trailing Take Profit")
    passive_exit_active: bool = Field(default=False, description="Whether passive maker time-limit exit is currently open")
    passive_exit_order_id: Optional[str] = Field(default=None, description="Order ID of active passive maker exit order")
    passive_exit_start_time: float = Field(default=0.0, description="When passive maker exit was initiated")
    entry_timestamp: float = Field(default=0.0, description="When position was opened")
    last_update_time: float = Field(default=0.0, description="Last update timestamp")
    filled_levels_count: int = Field(default=0, ge=0, description="Number of levels filled in current position cycle")
    is_trend_blocked: bool = Field(default=False, description="Flag indicating entry quote is blocked by trend bias")
    pyramid_filled_count: int = Field(default=0, ge=0, description="Number of favorable pyramid fills in current position cycle (max 1)")
    pending_pyramid_client_id: Optional[str] = Field(default=None, description="Client order ID of pending pyramid market entry")
    pending_pyramid_started_at: float = Field(default=0.0, description="Timestamp when pyramid market order was dispatched")
    is_guaranteed_sl_locked: bool = Field(default=False, description="Flag confirming Stop Loss is locked to guaranteed breakeven/profit")
    trend_bias_regime: str = Field(default="NEUTRAL", description="Trend regime: BULLISH, BEARISH, NEUTRAL")


class PnLRecord(BaseModel):
    """Historical PnL record for reporting."""
    id: Optional[int] = Field(default=None, description="Database auto-increment ID")
    symbol: str = Field(..., description="Symbol")
    position_side: PositionSide = Field(..., description="LONG or SHORT")
    realized_pnl: float = Field(..., description="Realized PnL in USDT")
    fee: float = Field(default=0.0, description="Total fees in USDT")
    net_pnl: float = Field(..., description="Net PnL = realized_pnl - fee")
    timestamp: float = Field(..., description="Timestamp of close / calculation")
    note: str = Field(default="", description="Description / exit reason (TP, SL, Trailing, Kill-All)")
