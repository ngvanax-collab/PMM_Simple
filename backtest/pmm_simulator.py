"""High-Performance Historical PMM Backtest Engine (Futures Hedge Mode Native)."""
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from loguru import logger
import numpy as np
import pandas as pd

from backtest.data_loader import precompute_market_indicators
from backtest.metrics import calculate_metrics


@dataclass
class PositionSlot:
    """Represents an open futures position slot in Hedge Mode."""
    side: str  # "LONG" or "SHORT"
    entry_price: float
    qty: float
    opened_at: Any
    entry_fee: float
    peak_price: float = 0.0  # For LONG trailing tracking
    trough_price: float = float("inf")  # For SHORT trailing tracking
    trailing_active: bool = False
    initial_sl_pct: float = 0.012
    initial_tp_pct: float = 0.006
    opened_dt: Optional[pd.Timestamp] = None
    opened_idx: int = 0


class PMMBacktestEngine:
    """
    Pure Market Making Historical Backtest Engine for Binance Futures Hedge Mode.

    Digital Twin Invariants (OpenPMM-Engine v3 Replication):
    1. Zero Look-Ahead Bias & In-Memory Parquet Processing.
    2. Dynamic VAMM Spread Scaling:
       - s_base = max(0.0025, k_spread * natr_15m)
       - TP_base = max(0.0050, k_tp * natr_15m)
       - SL_base = s_base + (k_sl_mult * natr_15m)
    3. Exit Priority Preservation:
       - Circuit Breakers and Trend Bias ONLY block new entry quotes.
       - Open positions strictly retain all protective exit barriers (SL, TP, Trailing TP, Toxic, Decay).
    4. Realistic Fees & Slippage Accounting:
       - Maker fee: 0.02% (Entry & Limit TP)
       - Taker fee: 0.05% + Slippage: 0.05% (Stop Loss & Fast Toxic Exits)
    5. Conservative Maker Fill Model:
       - Entry Bid fills only if Low <= Bid - 0.5 * TickSize
       - Entry Ask fills only if High >= Ask + 0.5 * TickSize
    """

    def __init__(
        self,
        symbol: str = "SOLUSDT",
        initial_balance: float = 1000.0,
        order_amount_usdt: float = 30.0,
        base_spread_pct: float = 0.0035,
        take_profit_pct: float = 0.006,
        stop_loss_pct: float = 0.012,
        trailing_callback_pct: float = 0.003,
        trailing_activation_pct: Optional[float] = None,
        maker_fee: float = 0.0002,      # 0.02%
        taker_fee: float = 0.0005,      # 0.05%
        slippage_pct: float = 0.0005,   # 0.05%
        tick_size: float = 0.01,
        qty_step: float = 0.001,
        entry_ref: str = "open",        # "open", "prev_close", or "close"
        close_positions_at_end: bool = True,
        # Dynamic Fidelity Parameters
        k_spread: float = 0.35,
        k_tp: float = 0.65,
        k_sl_mult: float = 1.8,
        cb_threshold_pct: float = 0.010,
        vol_spike_threshold: float = 3.0,
        time_decay_mins: int = 30,
        use_vamm: bool = True,
        use_circuit_breaker: bool = True,
        use_trend_bias: bool = True,
        use_toxic_exit: bool = True,
        use_time_decay_sl: bool = True,
        max_holding_mins: Optional[int] = 244,
    ):
        self.symbol = symbol
        self.initial_balance = float(initial_balance)
        self.order_amount_usdt = float(order_amount_usdt)
        self.base_spread_pct = float(base_spread_pct)
        self.take_profit_pct = float(take_profit_pct)
        self.stop_loss_pct = float(stop_loss_pct)
        self.trailing_callback_pct = float(trailing_callback_pct)
        self.trailing_activation_pct = (
            float(trailing_activation_pct) if trailing_activation_pct is not None else None
        )
        self.maker_fee = float(maker_fee)
        self.taker_fee = float(taker_fee)
        self.slippage_pct = float(slippage_pct)
        self.tick_size = float(tick_size)
        self.qty_step = float(qty_step)
        self.entry_ref = entry_ref.lower()
        self.close_positions_at_end = close_positions_at_end

        # Dynamic Engine Configuration
        self.k_spread = float(k_spread)
        self.k_tp = float(k_tp)
        self.k_sl_mult = float(k_sl_mult)
        self.cb_threshold_pct = float(cb_threshold_pct)
        self.vol_spike_threshold = float(vol_spike_threshold)
        self.time_decay_mins = int(time_decay_mins)
        self.use_vamm = bool(use_vamm)
        self.use_circuit_breaker = bool(use_circuit_breaker)
        self.use_trend_bias = bool(use_trend_bias)
        self.use_toxic_exit = bool(use_toxic_exit)
        self.use_time_decay_sl = bool(use_time_decay_sl)
        self.max_holding_mins = int(max_holding_mins) if max_holding_mins is not None else None

        # Simulation State
        self.balance: float = self.initial_balance
        self.long_pos: Optional[PositionSlot] = None
        self.short_pos: Optional[PositionSlot] = None
        self.trades: List[Dict[str, Any]] = []
        self.equity_records: List[Dict[str, Any]] = []
        self.equity_curve: pd.DataFrame = pd.DataFrame()

        # Quantitative Diagnostics State
        self.cb_trigger_count: int = 0
        self.blocked_trend_trades: int = 0
        self.fast_exit_count: int = 0
        self.fast_exit_saved_usdt: float = 0.0
        self.cb_pause_until: Optional[pd.Timestamp] = None

    def reset(self) -> None:
        """Reset simulation state and diagnostic counters."""
        self.balance = self.initial_balance
        self.long_pos = None
        self.short_pos = None
        self.trades = []
        self.equity_records = []
        self.equity_curve = pd.DataFrame()

        self.cb_trigger_count = 0
        self.blocked_trend_trades = 0
        self.fast_exit_count = 0
        self.fast_exit_saved_usdt = 0.0
        self.cb_pause_until = None

    def quantize_price(self, price: float, is_bid: Optional[bool] = None) -> float:
        """
        Quantize price according to tick size.
        Bid: floor to nearest tick. Ask: ceil to nearest tick.
        """
        if self.tick_size <= 0:
            return price
        if is_bid is True:
            ticks = math.floor(price / self.tick_size + 1e-9)
        elif is_bid is False:
            ticks = math.ceil(price / self.tick_size - 1e-9)
        else:
            ticks = round(price / self.tick_size)
        return round(ticks * self.tick_size, 8)

    def quantize_qty(self, qty: float) -> float:
        """Quantize quantity to step size."""
        if self.qty_step <= 0:
            return qty
        steps = math.floor(qty / self.qty_step + 1e-9)
        return round(steps * self.qty_step, 8)

    def _close_position(
        self,
        slot: PositionSlot,
        exit_price: float,
        exit_time: Any,
        exit_reason: str,
        is_taker: bool = False,
        slippage_cost: float = 0.0,
    ) -> None:
        """Close an open position slot, compute fees and PnL, and record trade."""
        exit_notional = exit_price * slot.qty
        fee_rate = self.taker_fee if is_taker else self.maker_fee
        exit_fee = exit_notional * fee_rate

        if slot.side == "LONG":
            gross_pnl = (exit_price - slot.entry_price) * slot.qty
        else:  # SHORT
            gross_pnl = (slot.entry_price - exit_price) * slot.qty

        total_fee = slot.entry_fee + exit_fee
        net_pnl = gross_pnl - total_fee - slippage_cost

        # Update balance: add gross PnL and subtract exit fee and slippage
        # (entry fee was already subtracted upon entry fill)
        self.balance += gross_pnl - exit_fee - slippage_cost

        trade_record = {
            "symbol": self.symbol,
            "side": slot.side,
            "entry_time": slot.opened_at,
            "exit_time": exit_time,
            "entry_price": slot.entry_price,
            "exit_price": exit_price,
            "qty": slot.qty,
            "notional_entry": slot.entry_price * slot.qty,
            "notional_exit": exit_notional,
            "exit_reason": exit_reason,
            "entry_fee": slot.entry_fee,
            "exit_fee": exit_fee,
            "slippage_cost": slippage_cost,
            "total_fee": total_fee,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "is_taker": is_taker,
        }
        self.trades.append(trade_record)

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest over historical candles DataFrame.

        Expected DataFrame columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
        If dynamic indicator columns ('natr_15m', 'trend_bias_1h', 'delta_p_60s', 'vol_ratio_1m')
        are not present, they will be precomputed automatically without look-ahead bias.

        Returns dictionary of quantitative performance metrics.
        """
        self.reset()
        if df.empty:
            return calculate_metrics(
                trades=[],
                equity_curve=pd.DataFrame(),
                initial_balance=self.initial_balance,
                cb_trigger_count=0,
                blocked_trend_trades=0,
                fast_exit_count=0,
                fast_exit_saved_usdt=0.0,
            )

        # Ensure indicators are precomputed
        req_cols = {"natr_15m", "trend_bias_1h", "delta_p_60s", "vol_ratio_1m"}
        if not req_cols.issubset(df.columns):
            df = precompute_market_indicators(df)

        # Pre-extract numpy arrays for maximum iteration performance
        timestamps = df["timestamp"].values
        timestamps_dt = [pd.Timestamp(ts) for ts in timestamps]
        opens = df["open"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)
        closes = df["close"].values.astype(np.float64)
        volumes = df["volume"].values.astype(np.float64)

        natr_15m_arr = df["natr_15m"].values.astype(np.float64) if "natr_15m" in df.columns else np.full(len(df), np.nan)
        trend_bias_arr = df["trend_bias_1h"].values if "trend_bias_1h" in df.columns else np.full(len(df), "NEUTRAL")
        delta_p_60s_arr = df["delta_p_60s"].values.astype(np.float64) if "delta_p_60s" in df.columns else (highs - lows) / np.where(lows > 0, lows, 1.0)
        vol_ratio_arr = df["vol_ratio_1m"].values.astype(np.float64) if "vol_ratio_1m" in df.columns else np.ones(len(df))

        n_candles = len(df)
        half_tick = 0.5 * self.tick_size
        prev_close: Optional[float] = None

        for i in range(n_candles):
            ts = timestamps[i]
            ts_dt = timestamps_dt[i]
            o = opens[i]
            h = highs[i]
            l = lows[i]
            c = closes[i]
            v = volumes[i]

            natr = natr_15m_arr[i]
            trend_bias = trend_bias_arr[i]
            delta_60s = delta_p_60s_arr[i]
            vol_ratio = vol_ratio_arr[i]

            # ─────────────────────────────────────────────────────────────
            # 1. CO GIÃN DẢI LƯỚI VAMM DỰA TRÊN NATR 15M ĐÃ ĐÓNG
            # ─────────────────────────────────────────────────────────────
            if self.use_vamm and not np.isnan(natr) and natr > 0:
                s_base = max(0.0025, self.k_spread * natr)
                tp_base = max(0.0050, self.k_tp * natr)
                sl_base = s_base + (self.k_sl_mult * natr)
            else:
                s_base = self.base_spread_pct
                tp_base = self.take_profit_pct
                sl_base = self.stop_loss_pct

            # ─────────────────────────────────────────────────────────────
            # 2. BẢO TỒN LỐI THOÁT (EXIT PRIORITY FIRST)
            # ─────────────────────────────────────────────────────────────

            # ── 2.1 Quản trị thoát vị thế LONG ──
            if self.long_pos is not None:
                pos = self.long_pos
                pos.peak_price = max(pos.peak_price, h)

                # Duration for Time-Decay SL
                duration_mins = (ts_dt - pd.Timestamp(pos.opened_at)).total_seconds() / 60.0
                is_time_decay = False
                eff_sl_pct = pos.initial_sl_pct
                if self.use_time_decay_sl and duration_mins > self.time_decay_mins:
                    eff_sl_pct = 0.5 * pos.initial_sl_pct
                    is_time_decay = True

                sl_price = pos.entry_price * (1.0 - eff_sl_pct)
                hard_sl_price = pos.entry_price * (1.0 - pos.initial_sl_pct)
                tp_price = pos.entry_price * (1.0 + pos.initial_tp_pct)

                # Trailing Take Profit check
                act_pct = self.trailing_activation_pct if self.trailing_activation_pct is not None else pos.initial_tp_pct
                act_price = pos.entry_price * (1.0 + act_pct)
                if not pos.trailing_active and pos.peak_price >= act_price:
                    pos.trailing_active = True

                trailing_trigger = (
                    pos.peak_price * (1.0 - self.trailing_callback_pct)
                    if pos.trailing_active
                    else 0.0
                )

                # Current candle uPnL for Toxic Volume Exit
                upnl_pct = (c - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0.0

                # A. FAST TOXIC EXIT: uPnL <= -0.8% and vol_ratio >= threshold
                if self.use_toxic_exit and upnl_pct <= -0.008 and vol_ratio >= self.vol_spike_threshold:
                    exec_price = c * (1.0 - self.slippage_pct)
                    slippage_cost = (c - exec_price) * pos.qty
                    hard_sl_exec = hard_sl_price * (1.0 - self.slippage_pct)
                    saved_usdt = max(0.0, (exec_price - hard_sl_exec) * pos.qty)
                    self.fast_exit_count += 1
                    self.fast_exit_saved_usdt += saved_usdt
                    self._close_position(
                        slot=pos,
                        exit_price=exec_price,
                        exit_time=ts,
                        exit_reason="TOXIC_VOLUME_EXIT",
                        is_taker=True,
                        slippage_cost=slippage_cost,
                    )
                    self.long_pos = None

                # B. TIME LIMIT EXIT (Max holding minutes reached)
                elif self.max_holding_mins is not None and duration_mins >= self.max_holding_mins:
                    exec_price = self.quantize_price(c, is_bid=False)
                    self._close_position(
                        slot=pos,
                        exit_price=exec_price,
                        exit_time=ts,
                        exit_reason="TIME_LIMIT_EXIT",
                        is_taker=False,
                    )
                    self.long_pos = None

                # C. STOP LOSS (Time-Decay tightened SL or Hard SL)
                elif l <= sl_price:
                    exec_price = sl_price * (1.0 - self.slippage_pct)
                    slippage_cost = (sl_price - exec_price) * pos.qty
                    if is_time_decay and sl_price > hard_sl_price:
                        hard_sl_exec = hard_sl_price * (1.0 - self.slippage_pct)
                        saved_usdt = max(0.0, (exec_price - hard_sl_exec) * pos.qty)
                        self.fast_exit_count += 1
                        self.fast_exit_saved_usdt += saved_usdt
                        exit_reason = "TIME_DECAY_SL"
                    else:
                        exit_reason = "STOP_LOSS"

                    self._close_position(
                        slot=pos,
                        exit_price=exec_price,
                        exit_time=ts,
                        exit_reason=exit_reason,
                        is_taker=True,
                        slippage_cost=slippage_cost,
                    )
                    self.long_pos = None

                # C. FIXED TAKE PROFIT (Maker limit order)
                elif h >= tp_price:
                    exec_price = self.quantize_price(tp_price, is_bid=False)
                    self._close_position(
                        slot=pos,
                        exit_price=exec_price,
                        exit_time=ts,
                        exit_reason="TAKE_PROFIT",
                        is_taker=False,
                    )
                    self.long_pos = None

                # D. TRAILING TAKE PROFIT (Maker callback)
                elif pos.trailing_active and l <= trailing_trigger and trailing_trigger > pos.entry_price:
                    exec_price = self.quantize_price(trailing_trigger, is_bid=False)
                    self._close_position(
                        slot=pos,
                        exit_price=exec_price,
                        exit_time=ts,
                        exit_reason="TRAILING_TP",
                        is_taker=False,
                    )
                    self.long_pos = None

            # ── 2.2 Quản trị thoát vị thế SHORT ──
            if self.short_pos is not None:
                pos = self.short_pos
                pos.trough_price = min(pos.trough_price, l)

                duration_mins = (ts_dt - pd.Timestamp(pos.opened_at)).total_seconds() / 60.0
                is_time_decay = False
                eff_sl_pct = pos.initial_sl_pct
                if self.use_time_decay_sl and duration_mins > self.time_decay_mins:
                    eff_sl_pct = 0.5 * pos.initial_sl_pct
                    is_time_decay = True

                sl_price = pos.entry_price * (1.0 + eff_sl_pct)
                hard_sl_price = pos.entry_price * (1.0 + pos.initial_sl_pct)
                tp_price = pos.entry_price * (1.0 - pos.initial_tp_pct)

                act_pct = self.trailing_activation_pct if self.trailing_activation_pct is not None else pos.initial_tp_pct
                act_price = pos.entry_price * (1.0 - act_pct)
                if not pos.trailing_active and pos.trough_price <= act_price:
                    pos.trailing_active = True

                trailing_trigger = (
                    pos.trough_price * (1.0 + self.trailing_callback_pct)
                    if pos.trailing_active
                    else float("inf")
                )

                upnl_pct = (pos.entry_price - c) / pos.entry_price if pos.entry_price > 0 else 0.0

                # A. FAST TOXIC EXIT
                if self.use_toxic_exit and upnl_pct <= -0.008 and vol_ratio >= self.vol_spike_threshold:
                    exec_price = c * (1.0 + self.slippage_pct)
                    slippage_cost = (exec_price - c) * pos.qty
                    hard_sl_exec = hard_sl_price * (1.0 + self.slippage_pct)
                    saved_usdt = max(0.0, (hard_sl_exec - exec_price) * pos.qty)
                    self.fast_exit_count += 1
                    self.fast_exit_saved_usdt += saved_usdt
                    self._close_position(
                        slot=pos,
                        exit_price=exec_price,
                        exit_time=ts,
                        exit_reason="TOXIC_VOLUME_EXIT",
                        is_taker=True,
                        slippage_cost=slippage_cost,
                    )
                    self.short_pos = None

                # B. TIME LIMIT EXIT (Max holding minutes reached)
                elif self.max_holding_mins is not None and duration_mins >= self.max_holding_mins:
                    exec_price = self.quantize_price(c, is_bid=True)
                    self._close_position(
                        slot=pos,
                        exit_price=exec_price,
                        exit_time=ts,
                        exit_reason="TIME_LIMIT_EXIT",
                        is_taker=False,
                    )
                    self.short_pos = None

                # C. STOP LOSS
                elif h >= sl_price:
                    exec_price = sl_price * (1.0 + self.slippage_pct)
                    slippage_cost = (exec_price - sl_price) * pos.qty
                    if is_time_decay and sl_price < hard_sl_price:
                        hard_sl_exec = hard_sl_price * (1.0 + self.slippage_pct)
                        saved_usdt = max(0.0, (hard_sl_exec - exec_price) * pos.qty)
                        self.fast_exit_count += 1
                        self.fast_exit_saved_usdt += saved_usdt
                        exit_reason = "TIME_DECAY_SL"
                    else:
                        exit_reason = "STOP_LOSS"

                    self._close_position(
                        slot=pos,
                        exit_price=exec_price,
                        exit_time=ts,
                        exit_reason=exit_reason,
                        is_taker=True,
                        slippage_cost=slippage_cost,
                    )
                    self.short_pos = None

                # C. FIXED TAKE PROFIT
                elif l <= tp_price:
                    exec_price = self.quantize_price(tp_price, is_bid=True)
                    self._close_position(
                        slot=pos,
                        exit_price=exec_price,
                        exit_time=ts,
                        exit_reason="TAKE_PROFIT",
                        is_taker=False,
                    )
                    self.short_pos = None

                # D. TRAILING TAKE PROFIT
                elif pos.trailing_active and h >= trailing_trigger and trailing_trigger < pos.entry_price:
                    exec_price = self.quantize_price(trailing_trigger, is_bid=True)
                    self._close_position(
                        slot=pos,
                        exit_price=exec_price,
                        exit_time=ts,
                        exit_reason="TRAILING_TP",
                        is_taker=False,
                    )
                    self.short_pos = None

            # ─────────────────────────────────────────────────────────────
            # 3. MỞ VỊ THẾ MAKER ENTRY (NẾU KHÔNG BỊ KHÓA BỞI CB / TREND)
            # ─────────────────────────────────────────────────────────────
            # Is entry currently frozen by a circuit breaker triggered by a prior surge?
            is_cb_frozen = (
                self.use_circuit_breaker
                and self.cb_pause_until is not None
                and ts_dt <= self.cb_pause_until
            )

            if not is_cb_frozen:
                if self.entry_ref == "prev_close" and prev_close is not None:
                    ref_price = prev_close
                elif self.entry_ref == "close":
                    ref_price = c
                else:
                    ref_price = o  # Default: open of current candle

                # Trend Bias Filtering
                allow_long = True
                allow_short = True

                if self.use_trend_bias:
                    if trend_bias == "BULLISH":
                        allow_short = False
                        if self.short_pos is None:
                            self.blocked_trend_trades += 1
                    elif trend_bias == "BEARISH":
                        allow_long = False
                        if self.long_pos is None:
                            self.blocked_trend_trades += 1

                # ── 3.1 LONG Entry Check ──
                if allow_long and self.long_pos is None and ref_price > 0:
                    raw_bid = ref_price * (1.0 - s_base)
                    bid_price = self.quantize_price(raw_bid, is_bid=True)

                    # Conservative Maker Fill: Low <= Bid - 0.5 * TickSize
                    if l <= (bid_price - half_tick):
                        qty = self.quantize_qty(self.order_amount_usdt / bid_price)
                        if qty > 0:
                            entry_fee = (bid_price * qty) * self.maker_fee
                            self.balance -= entry_fee
                            self.long_pos = PositionSlot(
                                side="LONG",
                                entry_price=bid_price,
                                qty=qty,
                                opened_at=ts,
                                opened_dt=ts_dt,
                                opened_idx=i,
                                entry_fee=entry_fee,
                                peak_price=bid_price,
                                initial_sl_pct=sl_base,
                                initial_tp_pct=tp_base,
                            )

                # ── 3.2 SHORT Entry Check ──
                if allow_short and self.short_pos is None and ref_price > 0:
                    raw_ask = ref_price * (1.0 + s_base)
                    ask_price = self.quantize_price(raw_ask, is_bid=False)

                    # Conservative Maker Fill: High >= Ask + 0.5 * TickSize
                    if h >= (ask_price + half_tick):
                        qty = self.quantize_qty(self.order_amount_usdt / ask_price)
                        if qty > 0:
                            entry_fee = (ask_price * qty) * self.maker_fee
                            self.balance -= entry_fee
                            self.short_pos = PositionSlot(
                                side="SHORT",
                                entry_price=ask_price,
                                qty=qty,
                                opened_at=ts,
                                opened_dt=ts_dt,
                                opened_idx=i,
                                entry_fee=entry_fee,
                                trough_price=ask_price,
                                initial_sl_pct=sl_base,
                                initial_tp_pct=tp_base,
                            )

            # ─────────────────────────────────────────────────────────────
            # 4. CẦU DAO BIẾN ĐỘNG 60S (CIRCUIT BREAKER KÍCH HOẠT CHO NẾN SAU)
            # ─────────────────────────────────────────────────────────────
            eff_cb_threshold = (
                max(self.cb_threshold_pct, float(natr))
                if (not np.isnan(natr) and natr > 0)
                else self.cb_threshold_pct
            )

            # Check if this 60s bar experienced a surge exceeding threshold
            if self.use_circuit_breaker and delta_60s >= eff_cb_threshold:
                self.cb_pause_until = ts_dt + pd.Timedelta(seconds=60)
                self.cb_trigger_count += 1

            # ─────────────────────────────────────────────────────────────
            # 5. GHI NHẬN EQUITY CURVE TẠI MỖI BƯỚC NẾN
            # ─────────────────────────────────────────────────────────────
            long_upnl = (
                (c - self.long_pos.entry_price) * self.long_pos.qty
                if self.long_pos is not None
                else 0.0
            )
            short_upnl = (
                (self.short_pos.entry_price - c) * self.short_pos.qty
                if self.short_pos is not None
                else 0.0
            )
            total_upnl = long_upnl + short_upnl
            current_equity = self.balance + total_upnl

            self.equity_records.append({
                "timestamp": ts,
                "equity": current_equity,
                "balance": self.balance,
                "upnl": total_upnl,
                "long_open": self.long_pos is not None,
                "short_open": self.short_pos is not None,
            })

            prev_close = c

        # Close any lingering positions at end of data if requested
        if self.close_positions_at_end and n_candles > 0:
            final_ts = timestamps[-1]
            final_c = closes[-1]
            if self.long_pos is not None:
                self._close_position(
                    slot=self.long_pos,
                    exit_price=final_c,
                    exit_time=final_ts,
                    exit_reason="END_OF_DATA",
                    is_taker=True,
                )
                self.long_pos = None
            if self.short_pos is not None:
                self._close_position(
                    slot=self.short_pos,
                    exit_price=final_c,
                    exit_time=final_ts,
                    exit_reason="END_OF_DATA",
                    is_taker=True,
                )
                self.short_pos = None

        self.equity_curve = pd.DataFrame(self.equity_records)
        metrics = calculate_metrics(
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_balance=self.initial_balance,
            cb_trigger_count=self.cb_trigger_count,
            blocked_trend_trades=self.blocked_trend_trades,
            fast_exit_count=self.fast_exit_count,
            fast_exit_saved_usdt=self.fast_exit_saved_usdt,
        )
        return metrics
