"""Configuration Models using Pydantic v2."""
from typing import List, Optional, Tuple
from pydantic import AliasChoices, BaseModel, Field, field_validator


class TrailingStopConfig(BaseModel):
    """Trailing stop configuration."""
    activation_price: float = Field(..., description="Distance in % (e.g. 0.012 = 1.2%) from entry to activate trailing")
    trailing_delta: float = Field(..., description="Trailing distance in % (e.g. 0.004 = 0.4%) to trigger market exit")


class PairConfig(BaseModel):
    """Configuration for a single trading pair (Futures Hedge Mode)."""
    symbol: str = Field(..., description="Unified futures symbol, e.g. SOL/USDT:USDT")
    exchange: str = Field(default="binance", description="Target exchange: binance or bybit")
    enabled: bool = Field(default=True, description="Enable quoting and execution")
    leverage: int = Field(default=5, ge=1, le=125, description="Leverage multiplier")
    margin_mode: str = Field(default="isolated", description="Margin mode: isolated or cross")

    # ── Quoting ──
    order_amount_usdt: float = Field(default=30.0, gt=0, description="Base order amount in USDT per side")
    bid_spread: float = Field(default=0.0035, gt=0, description="Half spread for bids (0.0035 = 0.35%)")
    ask_spread: float = Field(default=0.0035, gt=0, description="Half spread for asks (0.0035 = 0.35%)")
    minimum_spread: float = Field(default=0.0025, ge=0, description="Minimum spread floor buffer (0.0025 = 0.25%)")
    order_levels: int = Field(default=3, ge=1, le=10, description="Number of order levels per side")
    order_level_spread: float = Field(default=0.0025, ge=0, description="Spread increment per level (0.0025 = 0.25%)")
    order_level_amount: float = Field(default=0.0, ge=0, description="Amount increment per level in USDT")
    level_cooldown_sec: int = Field(default=1800, ge=0, description="Cooldown in seconds after each level fill before next level can be placed (default 1800s = 30m)")
    order_refresh_time: int = Field(default=45, ge=1, description="Periodic requote interval in seconds")
    requote_threshold_pct: float = Field(default=0.001, gt=0, description="Requote if mid price moves > % (0.001 = 0.1%)")
    min_quote_lifespan_sec: float = Field(default=5.0, ge=0.0, le=60.0, description="Minimum lifespan in seconds before an order can be cancelled for requoting")
    ping_pong_enabled: bool = Field(default=False, description="If true, pause new entries on side with open position")
    hanging_orders_enabled: bool = Field(default=True, description="Keep unfilled orders until threshold exceeded")
    hanging_orders_cancel_pct: float = Field(default=0.02, ge=0.001, description="Cancel hanging order if distance exceeds this %")
    price_ceiling: float = Field(default=-1.0, description="Max allowed ask price (-1 disables)")
    price_floor: float = Field(default=-1.0, description="Min allowed bid price (-1 disables)")

    # ── Inventory & Skew (Hedge: 2 independent sides) ──
    inventory_skew_enabled: bool = Field(default=True, description="Enable independent per-side skew")
    max_long_usdt: float = Field(default=96.0, gt=0, description="Max gross position size for LONG in USDT")
    max_short_usdt: float = Field(default=96.0, gt=0, description="Max gross position size for SHORT in USDT")
    gross_exposure_cap_usdt: float = Field(default=105.0, gt=0, description="Total gross cap (LONG + SHORT) in USDT")
    allocated_margin_usdt: Optional[float] = Field(default=21.0, description="Explicit margin quota cap in USDT allocated to this pair")

    skew_kappa: float = Field(default=1.0, ge=0.0, description="Spread widening coefficient for inventory skew")
    skew_gamma_net: float = Field(default=0.001, ge=0.0, description="Center price shift factor according to net inventory")
    tp_skew_boost: float = Field(default=0.5, ge=0.0, description="Tighten TP as inventory ratio increases")

    @property
    def effective_margin_cap(self) -> float:
        """
        Effective margin cap in USDT allocated to this pair.
        - If allocated_margin_usdt is set (> 0), uses allocated_margin_usdt.
        - Otherwise, derived from gross_exposure_cap_usdt / leverage.
        """
        if self.allocated_margin_usdt is not None and self.allocated_margin_usdt > 0:
            return float(self.allocated_margin_usdt)
        lev = max(1, self.leverage)
        return float(self.gross_exposure_cap_usdt / lev)

    # ── Triple Barrier & Trailing Take Profit (Applied per positionSide) ──
    take_profit: float = Field(default=0.008, gt=0, description="Default single take profit % (e.g. 0.008 = 0.8%)")
    take_profit_order_type: str = Field(default="LIMIT_MAKER", description="LIMIT_MAKER, LIMIT, or MARKET")
    tp_levels: List[List[float]] = Field(
        default_factory=lambda: [[0.008, 0.6], [0.015, 0.4]],
        description="Multi-tier TP: [[tp_pct_1, fraction_1], [tp_pct_2, fraction_2]] where fractions sum to 1.0"
    )
    trailing_tp_enabled: bool = Field(default=True, description="Enable Dynamic Trailing Take Profit")
    trailing_tp_activation_pct: float = Field(default=0.008, gt=0, description="Minimum profit % from entry to activate trailing TP (e.g. 0.008 = 0.8%)")
    trailing_tp_callback_pct: float = Field(default=0.003, gt=0, description="Pullback % from peak/trough to trigger take profit (e.g. 0.003 = 0.3%)")
    stop_loss: float = Field(default=0.018, gt=0, description="Virtual Stop Loss % from entry (e.g. 0.018 = 1.8%)")
    stop_loss_order_type: str = Field(default="MARKET", description="Emergency exit order type on Virtual SL trigger")
    trailing_stop: Optional[TrailingStopConfig] = Field(
        default_factory=lambda: TrailingStopConfig(activation_price=0.012, trailing_delta=0.004)
    )
    time_limit: int = Field(
        default=14400, ge=60,
        validation_alias=AliasChoices("time_limit", "time_limit_sec"),
        description="Max hold time in seconds before passive maker exit"
    )
    min_holding_sec: float = Field(default=10.0, ge=0.0, description="Minimum holding lock in seconds before non-SL exits")
    passive_exit_timeout_sec: float = Field(default=120.0, ge=10.0, description="Timeout in seconds for passive maker time-limit exit order")
    passive_exit_spread_pct: float = Field(default=0.0006, ge=0.0, description="Price offset % for passive maker exit (0.0006 = 0.06% around breakeven)")
    cooldown_time: int = Field(default=600, ge=0, description="Legacy cooldown in seconds after SL")

    # ── Progressive Cooldown & Isolated Worker Risk Limits ──
    base_cooldown_sec: int = Field(default=600, ge=0, description="Base cooldown in seconds on 1st SL (default 600s = 10m)")
    cooldown_multiplier: float = Field(default=2.0, ge=1.0, le=10.0, description="Progressive cooldown multiplier")
    max_cooldown_sec: int = Field(
        default=43200, ge=0,
        validation_alias=AliasChoices("max_cooldown_sec", "max_cooldown_cap_sec"),
        description="Max cooldown cap in seconds (default 43200s = 12h)"
    )
    worker_max_loss_usdt: float = Field(
        default=20.0, ge=1.0,
        validation_alias=AliasChoices("worker_max_loss_usdt", "worker_max_loss_limit_usdt"),
        description="Worker max cumulative loss limit in USDT"
    )
    worker_max_drawdown_usdt: float = Field(
        default=15.0, ge=1.0,
        validation_alias=AliasChoices("worker_max_drawdown_usdt", "worker_max_drawdown_limit_usdt"),
        description="Worker max drawdown limit from peak PnL in USDT"
    )
    is_locked: bool = Field(default=False, description="Flag indicating worker is locked due to Max Loss/Drawdown breach")


    # ── Dynamic Volatility Circuit Breaker ──
    circuit_breaker_enabled: bool = Field(default=True, description="Bật cầu dao ngắt 60s")
    circuit_breaker_lookback_sec: int = Field(default=60, ge=10, description="Cửa sổ trượt đo giật giá (60s)")
    circuit_breaker_natr_multiplier: float = Field(default=1.0, ge=0.1, description="Hệ số nhân NATR 15m để kích hoạt ngắt")
    circuit_breaker_min_threshold_pct: float = Field(default=0.008, ge=0.001, description="Ngưỡng sàn tối thiểu chống kích hoạt giả (0.80%)")
    circuit_breaker_pause_sec: int = Field(default=60, ge=5, description="Thời gian tạm dừng đặt lệnh entry (60s)")

    # ── Risk & Volatility Filters ──
    max_loss_usdt: float = Field(default=15.0, gt=0, description="Max single position loss threshold")
    daily_loss_limit_usdt: float = Field(
        default=30.0, gt=0,
        validation_alias=AliasChoices("daily_loss_limit_usdt", "max_daily_loss_usdt"),
        description="Max cumulative daily loss for this pair"
    )
    reconcile_interval_sec: int = Field(default=60, ge=10, description="Periodic REST reconcile interval in seconds")
    vol_pause_pct: float = Field(default=0.025, gt=0, description="Pause quoting if price moves > % in lookback window")
    vol_lookback_sec: int = Field(default=60, ge=10, description="Lookback window for volatility calculation")
    max_book_spread_pct: float = Field(default=0.005, gt=0, description="Pause quoting if orderbook spread exceeds %")
    min_depth_usdt: float = Field(default=50000.0, ge=0, description="Min depth in USDT within 1% of mid")
    max_basis_pct: float = Field(default=0.004, gt=0, description="Max allowed basis |mid - mark| / mark")
    funding_avoid_window_sec: int = Field(default=300, ge=0, description="Seconds around funding time to reduce inventory or pause")

    @property
    def max_daily_loss_usdt(self) -> float:
        """Alias for daily_loss_limit_usdt."""
        return self.daily_loss_limit_usdt

    @field_validator("margin_mode")
    @classmethod
    def validate_margin_mode(cls, v: str) -> str:
        v_clean = v.lower().strip()
        if v_clean not in {"isolated", "cross"}:
            raise ValueError("margin_mode must be 'isolated' or 'cross'")
        return v_clean

    @field_validator("take_profit_order_type")
    @classmethod
    def validate_tp_order_type(cls, v: str) -> str:
        v_clean = v.upper().strip()
        if v_clean not in {"LIMIT_MAKER", "LIMIT", "MARKET"}:
            raise ValueError("take_profit_order_type must be LIMIT_MAKER, LIMIT, or MARKET")
        return v_clean


class GlobalConfig(BaseModel):
    """Global system configuration."""
    account_daily_loss_limit_usdt: float = Field(default=100.0, gt=0, description="Max total account daily loss before Kill-All")
    min_margin_ratio: float = Field(default=1.5, gt=1.0, description="Margin ratio warning threshold (1.5 = 150%)")
    kill_margin_ratio: float = Field(default=1.2, gt=1.0, description="Margin ratio emergency kill threshold (1.2 = 120%)")
    order_budget_per_min: int = Field(default=900, ge=100, description="Global rate limit orders/min")
    weight_budget_per_min: int = Field(default=1800, ge=100, description="Global rate limit request weight/min")
    position_mode: str = Field(default="hedge", description="Contract position mode: hedge (mandatory)")
    testnet: bool = Field(default=False, description="Use testnet environment")
    reconcile_interval_sec: int = Field(default=60, ge=10, description="Periodic REST reconcile interval in seconds")


class ExchangeCredentials(BaseModel):
    """Exchange API credentials."""
    exchange: str = Field(default="binance", description="Exchange name (binance or bybit)")
    api_key: str = Field(default="", description="Exchange API Key")
    api_secret: str = Field(default="", description="Exchange API Secret")
    passphrase: Optional[str] = Field(default=None, description="Optional API Passphrase (for Bybit V5 if needed)")
    testnet: bool = Field(default=False, description="Use exchange testnet")


class ScreenerConfig(BaseModel):
    """Configuration for Quantitative Screener Engine with Volatility & Pump-Dump Filters."""
    min_volume_24h_usdt: float = Field(default=20_000_000.0, description="Min 24h volume in USDT ($20M - $200M)")
    min_spread_bps: float = Field(default=2.0, description="Min natural top-of-book spread in basis points (2 bps = 0.02%)")
    max_top_of_book_spread_bps: float = Field(default=8.0, description="Max natural top-of-book spread in basis points (8 bps = 0.08%)")
    min_depth_1pct_usdt: float = Field(default=50_000.0, description="Min depth +-1% in USDT")
    max_funding_rate_abs: float = Field(default=0.0001, description="Ideal max absolute funding rate per 8h (0.01% = 0.0001)")
    max_natr_pct: float = Field(default=1.8, description="Max NATR 14 1h ceiling (reject if NATR > 1.8%)")
    min_natr_pct: float = Field(default=0.8, description="Min NATR 14 1h floor (reject if NATR < 0.8%)")
    max_24h_price_change_pct: float = Field(default=6.0, description="Max 24h price change % (reject if |ΔP_24h| > 6.0%)")
    max_volatility_ratio: float = Field(default=1.25, description="Max short/long term volatility ratio (reject if σ_7d / σ_30d > 1.25)")
    max_hurst_exponent: float = Field(default=0.46, description="Max Hurst exponent (reject if H >= 0.46, i.e. trending)")
    ohlcv_timeframe: str = Field(default="1h", description="Candle timeframe for statistical analysis")
    ohlcv_limit: int = Field(default=100, description="Number of historical candles to analyze")
    max_concurrency: int = Field(default=10, description="Max concurrent requests via Semaphore")
    blacklist: List[str] = Field(
        default_factory=lambda: [
            "USDC/USDT:USDT", "FDUSD/USDT:USDT", "BUSD/USDT:USDT",
            "TUSD/USDT:USDT", "EUR/USDT:USDT", "USDP/USDT:USDT",
            "DAI/USDT:USDT", "USTC/USDT:USDT", "AEUR/USDT:USDT"
        ],
        description="Blacklisted pairs (e.g. stablecoins, delisting candidates)"
    )
