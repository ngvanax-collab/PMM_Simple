"""Historical Data Loader for Binance Futures with Apache Parquet Caching & Vectorized Indicators."""
import os
import re
import time
from pathlib import Path
from typing import Optional, Union
from loguru import logger
import numpy as np
import pandas as pd
import requests

DEFAULT_CACHE_DIR = Path(__file__).parent / "data_historical"


def normalize_symbol(symbol: str) -> str:
    """Normalize exchange symbols like SOL/USDT:USDT or sol-usdt to SOLUSDT."""
    cleaned = re.sub(r"[/:\-_]", "", symbol).upper()
    return cleaned


def precompute_market_indicators(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Precompute vectorized multi-timeframe market indicators from 1m candles.

    Strict Invariants:
    1. Zero Look-Ahead Bias:
       - Resampled 15m and 1h indicators are shifted by 1 candle before mapping back
         to 1m candles, ensuring no data from currently-forming candles leaks into 1m bars.
       - No backward fill (bfill) across time horizons to avoid future data contamination.
    2. High-Performance Vectorization using Pandas and NumPy.

    Generated Indicators:
    - delta_p_60s: (High - Low) / Low on the 1m candle (60s price swing).
    - vol_ma10: Rolling 10-period SMA of 1m volume.
    - vol_ratio_1m: Volume / vol_ma10.
    - natr_15m: Normalized Average True Range (14 periods) on 15m candles, shifted 1 candle.
    - ema50_1h: Exponential Moving Average (50 periods) on 1h candles, shifted 1 candle.
    - atr14_1h: Average True Range (14 periods) on 1h candles, shifted 1 candle.
    - trend_bias_1h: 'BULLISH', 'BEARISH', or 'NEUTRAL' based on Close_1m vs EMA50_1h +- 0.5 * ATR14_1h.
    """
    if df_1m.empty:
        return df_1m.copy()

    df = df_1m.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────
    # 1. KHUNG 1M: MICRO-VOLATILITY & VOLUME SPIKES
    # ─────────────────────────────────────────────────────────────
    low_safe = np.where(df["low"] > 0, df["low"], 1e-9)
    df["delta_p_60s"] = (df["high"] - df["low"]) / low_safe

    df["vol_ma10"] = df["volume"].rolling(window=10, min_periods=1).mean()
    ma_safe = np.where(df["vol_ma10"] > 0, df["vol_ma10"], 1.0)
    df["vol_ratio_1m"] = df["volume"] / ma_safe

    df_indexed = df.set_index("timestamp")

    # ─────────────────────────────────────────────────────────────
    # 2. KHUNG 15M (RESAMPLE TỪ 1M): NATR 15M CO GIÃN VAMM
    # ─────────────────────────────────────────────────────────────
    res_15m = df_indexed.resample("15min", closed="left", label="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    if len(res_15m) > 1:
        high_15m = res_15m["high"]
        low_15m = res_15m["low"]
        prev_close_15m = res_15m["close"].shift(1).fillna(res_15m["open"])

        tr0 = high_15m - low_15m
        tr1 = (high_15m - prev_close_15m).abs()
        tr2 = (low_15m - prev_close_15m).abs()
        tr_15m = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)

        # RMA/EMA 14 periods for ATR
        atr14_15m = tr_15m.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=1).mean()
        c_15m_safe = np.where(res_15m["close"] > 0, res_15m["close"], 1.0)
        natr_15m_raw = atr14_15m / c_15m_safe

        # STRICT LOOK-AHEAD ELIMINATION: Shift 1 candle forward
        natr_15m_shifted = natr_15m_raw.shift(1)

        map_15m = pd.DataFrame({"natr_15m": natr_15m_shifted}, index=res_15m.index)
        df["natr_15m"] = map_15m.reindex(df_indexed.index, method="ffill")["natr_15m"].values
    else:
        df["natr_15m"] = np.nan

    # ─────────────────────────────────────────────────────────────
    # 3. KHUNG 1H (RESAMPLE TỪ 1M): TREND BIAS REGIME DETECTION
    # ─────────────────────────────────────────────────────────────
    res_1h = df_indexed.resample("1h", closed="left", label="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    if len(res_1h) > 1:
        high_1h = res_1h["high"]
        low_1h = res_1h["low"]
        prev_close_1h = res_1h["close"].shift(1).fillna(res_1h["open"])

        tr0_1h = high_1h - low_1h
        tr1_1h = (high_1h - prev_close_1h).abs()
        tr2_1h = (low_1h - prev_close_1h).abs()
        tr_1h = pd.concat([tr0_1h, tr1_1h, tr2_1h], axis=1).max(axis=1)

        atr14_1h = tr_1h.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=1).mean()
        ema50_1h = res_1h["close"].ewm(span=50, adjust=False, min_periods=1).mean()

        # STRICT LOOK-AHEAD ELIMINATION: Shift 1 candle forward
        atr14_1h_shifted = atr14_1h.shift(1)
        ema50_1h_shifted = ema50_1h.shift(1)

        map_1h = pd.DataFrame({
            "ema50_1h": ema50_1h_shifted,
            "atr14_1h": atr14_1h_shifted,
        }, index=res_1h.index)

        mapped_1h_df = map_1h.reindex(df_indexed.index, method="ffill")
        df["ema50_1h"] = mapped_1h_df["ema50_1h"].values
        df["atr14_1h"] = mapped_1h_df["atr14_1h"].values
    else:
        df["ema50_1h"] = np.nan
        df["atr14_1h"] = np.nan

    # ─────────────────────────────────────────────────────────────
    # 4. XÁC ĐỊNH TREND BIAS 1H
    # ─────────────────────────────────────────────────────────────
    upper_band = df["ema50_1h"] + (0.5 * df["atr14_1h"])
    lower_band = df["ema50_1h"] - (0.5 * df["atr14_1h"])

    conditions = [
        df["close"] > upper_band,
        df["close"] < lower_band,
    ]
    choices = ["BULLISH", "BEARISH"]
    df["trend_bias_1h"] = np.select(conditions, choices, default="NEUTRAL")

    return df


def fetch_binance_futures_klines(
    symbol: str,
    interval: str = "1m",
    days: int = 14,
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: Optional[bool] = None,
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch historical klines from Binance USDⓈ-M Futures API with in-memory Parquet caching.

    Freshness Invariant:
    If local Parquet cache exists and its newest candle is > 24 hours older than now,
    the loader will automatically attempt to refresh the data from Binance Futures,
    unless `force_download=False` is hardcoded to use the cache as-is.
    If a specific historical date range (start_time_ms and end_time_ms) is requested,
    the data is cached by date range and considered permanently valid once downloaded.

    Parameters:
    -----------
    symbol: Unified or standard symbol (e.g. 'SOLUSDT', 'SOL/USDT:USDT')
    interval: Kline interval ('1m', '3m', '5m', '15m', '1h', etc.)
    days: Historical lookback period in days (used when start_time_ms is None)
    cache_dir: Custom directory to store Parquet cache files (defaults to backtest/data_historical)
    force_download: If True, bypass cache and re-download. If False, strictly use cache.
                    If None (default), auto-refresh if cache is > 24h old.
    start_time_ms: Optional start timestamp in milliseconds
    end_time_ms: Optional end timestamp in milliseconds (defaults to current time if start_time_ms is None)

    Returns:
    --------
    pd.DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    """
    clean_symbol = normalize_symbol(symbol)
    target_cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    target_cache_dir.mkdir(parents=True, exist_ok=True)

    if start_time_ms is not None and end_time_ms is not None:
        start_dt = pd.Timestamp(start_time_ms, unit="ms", tz="UTC")
        end_dt = pd.Timestamp(end_time_ms, unit="ms", tz="UTC")
        cache_file = target_cache_dir / f"{clean_symbol}_{interval}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
    else:
        cache_file = target_cache_dir / f"{clean_symbol}_{interval}_{days}d.parquet"

    # 1. Check local Parquet cache & freshness
    cached_df: Optional[pd.DataFrame] = None
    if force_download is not True and cache_file.exists():
        try:
            df = pd.read_parquet(cache_file, engine="pyarrow")
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype("float64")
            df = df.sort_values("timestamp").reset_index(drop=True)

            if not df.empty:
                # If specific historical window in the past, cache is permanent
                if start_time_ms is not None and end_time_ms is not None:
                    logger.info(
                        f"Loaded historical date-range data from cache {cache_file.name} "
                        f"({len(df):,} candles, {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]})."
                    )
                    return df

                last_candle_time = df["timestamp"].iloc[-1]
                now_utc = pd.Timestamp.now(tz="UTC")
                age_hours = (now_utc - last_candle_time).total_seconds() / 3600.0

                if force_download is False or age_hours <= 24.0:
                    logger.info(
                        f"Loaded historical data from cache {cache_file} "
                        f"({len(df):,} candles, last: {last_candle_time}, age: {age_hours:.1f}h)."
                    )
                    return df
                else:
                    logger.info(
                        f"Cache {cache_file.name} is older than 24h ({age_hours:.1f}h old, last: {last_candle_time}). "
                        f"Auto-refreshing from Binance Futures..."
                    )
                    cached_df = df
        except Exception as e:
            logger.warning(f"Failed to read cache {cache_file} ({e}), re-fetching from API.")

    # 2. Paginate fetch from Binance Futures REST API
    logger.info(f"Fetching historical {interval} klines for {clean_symbol} from Binance Futures...")
    endpoint = "https://fapi.binance.com/fapi/v1/klines"

    if end_time_ms is None:
        end_time_ms = int(time.time() * 1000)
    if start_time_ms is None:
        start_time_ms = end_time_ms - (days * 24 * 60 * 60 * 1000)

    current_start = start_time_ms
    limit = 1500
    all_raw_klines = []

    session = requests.Session()
    session.headers.update({"User-Agent": "OpenPMM-Backtest/1.0"})

    fetch_success = False
    try:
        while current_start < end_time_ms:
            params = {
                "symbol": clean_symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_time_ms,
                "limit": limit,
            }

            try:
                resp = session.get(endpoint, params=params, timeout=15)
                if resp.status_code == 429:
                    logger.warning("Binance rate limit hit (429), backing off for 5 seconds...")
                    time.sleep(5)
                    continue
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"Error fetching klines batch at start_time={current_start}: {e}")
                time.sleep(2)
                try:
                    resp = session.get(endpoint, params=params, timeout=20)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as retry_err:
                    logger.critical(f"Fatal error on retry: {retry_err}")
                    break

            if not data:
                break

            all_raw_klines.extend(data)
            last_kline_open = int(data[-1][0])
            if last_kline_open <= current_start:
                break

            current_start = last_kline_open + 1
            time.sleep(0.05)

        if all_raw_klines:
            fetch_success = True
    except Exception as exc:
        logger.error(f"Failed to download klines from Binance: {exc}")

    # Fallback to cached data if download failed but cache was present
    if not fetch_success or not all_raw_klines:
        if cached_df is not None and not cached_df.empty:
            logger.warning(
                f"Download from Binance failed/empty, falling back to existing cached data ({len(cached_df):,} candles)."
            )
            return cached_df
        raise ValueError(f"No kline data retrieved for {clean_symbol} ({interval}, {days} days).")

    # 3. Format DataFrame
    df_raw = pd.DataFrame(
        all_raw_klines,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore"
        ]
    )

    df = pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df_raw["open_time"].astype("int64"), unit="ms", utc=True)
    df["open"] = df_raw["open"].astype("float64")
    df["high"] = df_raw["high"].astype("float64")
    df["low"] = df_raw["low"].astype("float64")
    df["close"] = df_raw["close"].astype("float64")
    df["volume"] = df_raw["volume"].astype("float64")

    # Deduplicate and sort
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # 4. Save to Parquet with snappy compression
    try:
        df.to_parquet(cache_file, engine="pyarrow", compression="snappy", index=False)
        logger.info(f"Saved {len(df):,} candles to Parquet cache: {cache_file}")
    except Exception as e:
        logger.warning(f"Could not save Parquet cache to {cache_file}: {e}")

    return df
