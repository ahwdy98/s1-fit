from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(frame: pd.DataFrame, include_future: bool = True) -> pd.DataFrame:
    out = frame.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"].replace(0, np.nan)

    out["ret_1"] = close.pct_change()
    out["ret_2"] = close.pct_change(2)
    out["ret_3"] = close.pct_change(3)
    out["ret_4"] = close.pct_change(4)
    out["ret_5"] = close.pct_change(5)
    out["ret_6"] = close.pct_change(6)
    out["ret_8"] = close.pct_change(8)
    out["ret_10"] = close.pct_change(10)
    out["ret_13"] = close.pct_change(13)
    out["ret_20"] = close.pct_change(20)
    out["ret_21"] = close.pct_change(21)
    out["ret_34"] = close.pct_change(34)
    prev_close = close.shift(1)
    out["gap_pct"] = out["open"] / prev_close.replace(0, np.nan) - 1
    out["high_from_prev_close_pct"] = high / prev_close.replace(0, np.nan) - 1
    out["low_from_prev_close_pct"] = low / prev_close.replace(0, np.nan) - 1
    out["open_to_high_pct"] = high / out["open"].replace(0, np.nan) - 1
    out["open_to_low_pct"] = low / out["open"].replace(0, np.nan) - 1
    out["close_to_high_pct"] = close / high.replace(0, np.nan) - 1
    out["close_to_low_pct"] = close / low.replace(0, np.nan) - 1
    out["ema_5"] = ema(close, 5)
    out["ema_8"] = ema(close, 8)
    out["ema_10"] = ema(close, 10)
    out["ema_13"] = ema(close, 13)
    out["ema_20"] = ema(close, 20)
    out["ema_21"] = ema(close, 21)
    out["ema_34"] = ema(close, 34)
    out["ema_50"] = ema(close, 50)
    out["ema_55"] = ema(close, 55)
    out["ema_89"] = ema(close, 89)
    out["ema_144"] = ema(close, 144)
    out["ema_169"] = ema(close, 169)
    out["ema_200"] = ema(close, 200)
    out["ema_5_slope"] = out["ema_5"].pct_change(3)
    out["ema_8_slope"] = out["ema_8"].pct_change(3)
    out["ema_10_slope"] = out["ema_10"].pct_change(5)
    out["ema_13_slope"] = out["ema_13"].pct_change(5)
    out["ema_20_slope"] = out["ema_20"].pct_change(10)
    out["ema_21_slope"] = out["ema_21"].pct_change(10)
    out["ema_34_slope"] = out["ema_34"].pct_change(13)
    out["ema_55_slope"] = out["ema_55"].pct_change(21)
    out["ema_5_20_spread"] = close / out["ema_20"].replace(0, np.nan) - 1
    out["ema_8_21_spread"] = out["ema_8"] / out["ema_21"].replace(0, np.nan) - 1
    out["ema_13_34_spread"] = out["ema_13"] / out["ema_34"].replace(0, np.nan) - 1
    out["ema_10_50_spread"] = out["ema_10"] / out["ema_50"].replace(0, np.nan) - 1
    out["ema_20_50_spread"] = out["ema_20"] / out["ema_50"].replace(0, np.nan) - 1
    out["ema_34_89_spread"] = out["ema_34"] / out["ema_89"].replace(0, np.nan) - 1
    out["ema_55_144_spread"] = out["ema_55"] / out["ema_144"].replace(0, np.nan) - 1
    out["close_ema_8_spread"] = close / out["ema_8"].replace(0, np.nan) - 1
    out["close_ema_21_spread"] = close / out["ema_21"].replace(0, np.nan) - 1
    out["close_ema_55_spread"] = close / out["ema_55"].replace(0, np.nan) - 1
    out["close_ema_200_spread"] = close / out["ema_200"].replace(0, np.nan) - 1
    out["vegas_mid"] = (out["ema_144"] + out["ema_169"]) / 2
    out["vegas_pos"] = close / out["vegas_mid"].replace(0, np.nan) - 1
    out["trend_stack"] = (
        (out["ema_5"] > out["ema_10"]).astype(int)
        + (out["ema_10"] > out["ema_20"]).astype(int)
        + (out["ema_20"] > out["ema_50"]).astype(int)
    )
    out["trend_stack_fast"] = (
        (out["ema_8"] > out["ema_13"]).astype(int)
        + (out["ema_13"] > out["ema_21"]).astype(int)
        + (out["ema_21"] > out["ema_34"]).astype(int)
        + (out["ema_34"] > out["ema_55"]).astype(int)
    )
    out["trend_stack_slow"] = (
        (out["ema_34"] > out["ema_55"]).astype(int)
        + (out["ema_55"] > out["ema_89"]).astype(int)
        + (out["ema_89"] > out["ema_144"]).astype(int)
        + (out["ema_144"] > out["ema_169"]).astype(int)
    )
    out["rsi_14"] = rsi(close, 14)
    out["rsi_14_delta"] = out["rsi_14"].diff(3)

    macd_fast = ema(close, 12)
    macd_slow = ema(close, 26)
    out["macd"] = macd_fast - macd_slow
    out["macd_signal"] = ema(out["macd"], 9)
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    out["macd_hist_delta"] = out["macd_hist"].diff()

    middle = close.rolling(20, min_periods=10).mean()
    std = close.rolling(20, min_periods=10).std()
    out["bb_mid"] = middle
    out["bb_upper"] = middle + 2 * std
    out["bb_lower"] = middle - 2 * std
    out["bb_pos"] = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / middle.replace(0, np.nan)
    out["bb_width_z"] = _rolling_zscore(out["bb_width"], 60, 20)

    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    out["atr_pct"] = out["atr_14"] / close.replace(0, np.nan)
    out["atr_pct_z"] = _rolling_zscore(out["atr_pct"], 60, 20)
    out["atr_pct_rank_60"] = _rolling_percentile_rank(out["atr_pct"], 60, 20)
    out["body"] = (close - out["open"]).abs()
    out["body_pct"] = (close - out["open"]) / out["open"].replace(0, np.nan)
    out["body_abs_pct"] = out["body"] / out["open"].replace(0, np.nan)
    out["upper_wick"] = high - pd.concat([out["open"], close], axis=1).max(axis=1)
    out["lower_wick"] = pd.concat([out["open"], close], axis=1).min(axis=1) - low
    out["range"] = (high - low).replace(0, np.nan)
    out["close_pos"] = (close - low) / out["range"]
    out["range_pct"] = out["range"] / close.replace(0, np.nan)
    out["range_pct_rank_60"] = _rolling_percentile_rank(out["range_pct"], 60, 20)
    out["range_ratio_20"] = out["range"] / out["range"].rolling(20, min_periods=10).mean().replace(0, np.nan)
    out["range_ratio_5_20"] = out["range"].rolling(5, min_periods=3).mean() / out["range"].rolling(20, min_periods=10).mean().replace(0, np.nan)
    out["upper_wick_pct"] = out["upper_wick"] / out["range"]
    out["lower_wick_pct"] = out["lower_wick"] / out["range"]
    out["body_to_range"] = out["body"] / out["range"]
    out["upper_lower_wick_ratio"] = out["upper_wick"] / out["lower_wick"].replace(0, np.nan)
    out["long_upper_wick"] = (out["upper_wick_pct"] >= 0.45).astype(int)
    out["long_lower_wick"] = (out["lower_wick_pct"] >= 0.45).astype(int)
    out["close_near_high"] = (out["close_pos"] >= 0.8).astype(int)
    out["close_near_low"] = (out["close_pos"] <= 0.2).astype(int)
    out["bullish_body"] = ((out["body_pct"] > 0) & (out["close_pos"] >= 0.55)).astype(int)
    out["bearish_body"] = ((out["body_pct"] < 0) & (out["close_pos"] <= 0.45)).astype(int)
    out["gap_up_close_green"] = ((out["gap_pct"] > 0) & (out["body_pct"] > 0)).astype(int)
    out["gap_down_close_red"] = ((out["gap_pct"] < 0) & (out["body_pct"] < 0)).astype(int)
    out = out.copy()

    vol_mean = volume.rolling(20, min_periods=10).mean()
    vol_std = volume.rolling(20, min_periods=10).std()
    out["volume_ma20"] = vol_mean
    out["volume_z"] = (volume - vol_mean) / vol_std.replace(0, np.nan)
    out["volume_z_50"] = _rolling_zscore(volume, 50, 20)
    out["volume_rank_60"] = _rolling_percentile_rank(volume, 60, 20)
    out["volume_ratio_10"] = volume / volume.rolling(10, min_periods=5).mean().replace(0, np.nan)
    out["volume_ratio_20"] = volume / vol_mean.replace(0, np.nan)
    out["volume_ratio_50"] = volume / volume.rolling(50, min_periods=20).mean().replace(0, np.nan)
    out["volume_ratio_5_20"] = volume.rolling(5, min_periods=3).mean() / vol_mean.replace(0, np.nan)
    out["volume_ratio_3_10"] = volume.rolling(3, min_periods=2).mean() / volume.rolling(10, min_periods=5).mean().replace(0, np.nan)
    out["volume_dryup_20"] = volume / volume.rolling(20, min_periods=10).max().replace(0, np.nan)
    dollar_volume_proxy = close * out["volume"].fillna(0)
    if "amount" in out:
        amount = pd.to_numeric(out["amount"], errors="coerce")
        out["dollar_volume"] = amount.where(amount > 0, np.nan).fillna(dollar_volume_proxy)
    else:
        out["dollar_volume"] = dollar_volume_proxy
    dollar_volume = out["dollar_volume"].replace(0, np.nan)
    dollar_mean = dollar_volume.rolling(20, min_periods=10).mean()
    dollar_std = dollar_volume.rolling(20, min_periods=10).std()
    out["dollar_volume_ratio_10"] = dollar_volume / dollar_volume.rolling(10, min_periods=5).mean().replace(0, np.nan)
    out["dollar_volume_ratio_20"] = dollar_volume / dollar_mean.replace(0, np.nan)
    out["dollar_volume_ratio_50"] = dollar_volume / dollar_volume.rolling(50, min_periods=20).mean().replace(0, np.nan)
    out["dollar_volume_z"] = (dollar_volume - dollar_mean) / dollar_std.replace(0, np.nan)
    out["dollar_volume_rank_60"] = _rolling_percentile_rank(dollar_volume, 60, 20)
    if "turnover_rate" in out:
        turnover_rate = pd.to_numeric(out["turnover_rate"], errors="coerce")
    else:
        turnover_rate = pd.Series(np.nan, index=out.index)
    out["turnover_rate"] = turnover_rate
    out["turnover_rate_delta"] = turnover_rate.diff()
    out["turnover_rate_ratio_20"] = turnover_rate / turnover_rate.rolling(20, min_periods=10).mean().replace(0, np.nan)
    out["turnover_rate_z"] = _rolling_zscore(turnover_rate, 50, 20)
    out["turnover_rate_rank_60"] = _rolling_percentile_rank(turnover_rate, 60, 20)

    out["high_10"] = high.rolling(10, min_periods=5).max()
    out["low_10"] = low.rolling(10, min_periods=5).min()
    out["high_20"] = high.rolling(20, min_periods=5).max()
    out["low_20"] = low.rolling(20, min_periods=5).min()
    out["high_50"] = high.rolling(50, min_periods=10).max()
    out["low_50"] = low.rolling(50, min_periods=10).min()
    out["high_100"] = high.rolling(100, min_periods=20).max()
    out["low_100"] = low.rolling(100, min_periods=20).min()
    out["dist_10_low"] = close / out["low_10"] - 1
    out["dist_10_high"] = close / out["high_10"] - 1
    out["dist_20_low"] = close / out["low_20"] - 1
    out["dist_20_high"] = close / out["high_20"] - 1
    out["dist_50_low"] = close / out["low_50"] - 1
    out["dist_50_high"] = close / out["high_50"] - 1
    out["dist_100_low"] = close / out["low_100"] - 1
    out["dist_100_high"] = close / out["high_100"] - 1
    out["breakout_10"] = close / out["high_10"].shift(1).replace(0, np.nan) - 1
    out["breakout_20"] = close / out["high_20"].shift(1).replace(0, np.nan) - 1
    out["breakout_50"] = close / out["high_50"].shift(1).replace(0, np.nan) - 1
    out["breakdown_10"] = close / out["low_10"].shift(1).replace(0, np.nan) - 1
    out["breakdown_20"] = close / out["low_20"].shift(1).replace(0, np.nan) - 1
    out["breakdown_50"] = close / out["low_50"].shift(1).replace(0, np.nan) - 1
    out["high_breakout_10"] = high / out["high_10"].shift(1).replace(0, np.nan) - 1
    out["high_breakout_20"] = high / out["high_20"].shift(1).replace(0, np.nan) - 1
    out["high_breakout_50"] = high / out["high_50"].shift(1).replace(0, np.nan) - 1
    out["low_breakdown_10"] = low / out["low_10"].shift(1).replace(0, np.nan) - 1
    out["low_breakdown_20"] = low / out["low_20"].shift(1).replace(0, np.nan) - 1
    out["low_breakdown_50"] = low / out["low_50"].shift(1).replace(0, np.nan) - 1
    out["drawdown_10"] = close / out["high_10"].replace(0, np.nan) - 1
    out["drawdown_20"] = close / out["high_20"].replace(0, np.nan) - 1
    out["drawdown_50"] = close / out["high_50"].replace(0, np.nan) - 1
    out["drawdown_100"] = close / out["high_100"].replace(0, np.nan) - 1
    out["rebound_10"] = close / out["low_10"].replace(0, np.nan) - 1
    out["rebound_20"] = close / out["low_20"].replace(0, np.nan) - 1
    out["rebound_50"] = close / out["low_50"].replace(0, np.nan) - 1
    out["rebound_100"] = close / out["low_100"].replace(0, np.nan) - 1
    out["failed_high_breakout_20"] = ((out["high_breakout_20"] > 0) & (out["breakout_20"] < 0)).astype(int)
    out["failed_low_breakdown_20"] = ((out["low_breakdown_20"] < 0) & (out["breakdown_20"] > 0)).astype(int)
    out["up_days_5"] = (close.diff() > 0).rolling(5, min_periods=1).sum()
    out["down_days_5"] = (close.diff() < 0).rolling(5, min_periods=1).sum()
    out["up_days_10"] = (close.diff() > 0).rolling(10, min_periods=1).sum()
    out["down_days_10"] = (close.diff() < 0).rolling(10, min_periods=1).sum()
    out["up_streak"] = _streak(close.diff() > 0)
    out["down_streak"] = _streak(close.diff() < 0)
    out["inside_day"] = ((high <= high.shift(1)) & (low >= low.shift(1))).astype(int)
    out["outside_day"] = ((high >= high.shift(1)) & (low <= low.shift(1))).astype(int)
    out["td_up_count"] = _sequential_count(close > close.shift(4), 13)
    out["td_down_count"] = _sequential_count(close < close.shift(4), 13)
    out["td_up_count_delta"] = out["td_up_count"].diff()
    out["td_down_count_delta"] = out["td_down_count"].diff()
    out["td_up_8_13"] = out["td_up_count"].between(8, 13).astype(int)
    out["td_down_8_13"] = out["td_down_count"].between(8, 13).astype(int)
    out["td_up_13"] = (out["td_up_count"] == 13).astype(int)
    out["td_down_13"] = (out["td_down_count"] == 13).astype(int)

    if include_future:
        future_features = {}
        for window in [1, 2, 3, 5, 8, 10, 13, 20, 34]:
            future_close = close.shift(-window)
            future_max_close = _future_rolling(close, window, "max")
            future_min_close = _future_rolling(close, window, "min")
            future_max_high = _future_rolling(high, window, "max")
            future_min_low = _future_rolling(low, window, "min")
            future_available = _future_rolling(
                pd.Series(1.0, index=out.index), window, "sum"
            ).fillna(0)

            future_features[f"future_available_{window}"] = future_available
            future_features[f"future_complete_{window}"] = (
                future_available >= window
            ).astype(int)
            future_features[f"future_ret_{window}"] = (
                future_close / close.replace(0, np.nan) - 1
            )
            future_features[f"future_max_close_ret_{window}"] = (
                future_max_close / close.replace(0, np.nan) - 1
            )
            future_features[f"future_min_close_ret_{window}"] = (
                future_min_close / close.replace(0, np.nan) - 1
            )
            future_features[f"future_max_high_ret_{window}"] = (
                future_max_high / close.replace(0, np.nan) - 1
            )
            future_features[f"future_min_low_ret_{window}"] = (
                future_min_low / close.replace(0, np.nan) - 1
            )
            future_features[f"future_drawdown_{window}"] = (
                future_min_low / close.replace(0, np.nan) - 1
            )
            future_features[f"future_rebound_{window}"] = (
                future_max_high / close.replace(0, np.nan) - 1
            )
            future_features[f"future_up_days_{window}"] = _future_rolling(
                close.diff() > 0, window, "sum"
            )
            future_features[f"future_down_days_{window}"] = _future_rolling(
                close.diff() < 0, window, "sum"
            )
            future_features[f"future_breaks_high_{window}"] = (
                future_max_high > high
            ).astype(int)
            future_features[f"future_breaks_low_{window}"] = (
                future_min_low < low
            ).astype(int)
            future_features[f"future_close_above_{window}"] = (
                future_close > close
            ).astype(int)
            future_features[f"future_close_below_{window}"] = (
                future_close < close
            ).astype(int)
            future_features[f"future_confirmed_local_low_{window}"] = (
                low <= future_min_low
            ).astype(int)
            future_features[f"future_confirmed_local_high_{window}"] = (
                high >= future_max_high
            ).astype(int)
        out = pd.concat(
            [out, pd.DataFrame(future_features, index=out.index)], axis=1
        )
    return out.copy()


def _rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


def _rolling_percentile_rank(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    return series.rolling(window, min_periods=min_periods).apply(_last_percentile_rank, raw=True)


def _future_rolling(series: pd.Series, window: int, method: str) -> pd.Series:
    future = series.shift(-1).iloc[::-1]
    rolling = future.rolling(window, min_periods=1)
    if method == "max":
        value = rolling.max()
    elif method == "min":
        value = rolling.min()
    elif method == "sum":
        value = rolling.sum()
    else:
        raise ValueError(f"Unsupported future rolling method: {method}")
    return value.iloc[::-1].reindex(series.index)


def _last_percentile_rank(values: np.ndarray) -> float:
    current = values[-1]
    if not np.isfinite(current):
        return np.nan
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.mean(values <= current))


def _streak(condition: pd.Series) -> pd.Series:
    counts = []
    count = 0
    for active in condition.fillna(False).astype(bool):
        count = count + 1 if active else 0
        counts.append(count)
    return pd.Series(counts, index=condition.index, dtype="int64")


def _sequential_count(condition: pd.Series, max_count: int) -> pd.Series:
    counts = []
    count = 0
    exhausted = False
    for active in condition.fillna(False).astype(bool):
        if not active:
            count = 0
            exhausted = False
        elif exhausted:
            count = 0
        else:
            count = min(count + 1, max_count)
            if count >= max_count:
                exhausted = True
        counts.append(count)
    return pd.Series(counts, index=condition.index, dtype="int64")
