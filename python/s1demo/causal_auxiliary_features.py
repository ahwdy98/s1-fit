from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import add_indicators


LAG_STEPS = (1, 2, 3, 5, 8, 13)
LEVEL_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "ema_5",
    "ema_8",
    "ema_10",
    "ema_13",
    "ema_20",
    "ema_21",
    "ema_34",
    "ema_50",
    "ema_55",
    "ema_89",
    "ema_144",
    "ema_169",
    "ema_200",
    "vegas_mid",
    "macd",
    "macd_signal",
    "macd_hist",
    "macd_hist_delta",
    "bb_mid",
    "bb_upper",
    "bb_lower",
    "atr_14",
    "body",
    "upper_wick",
    "lower_wick",
    "range",
    "volume_ma20",
    "dollar_volume",
    "high_10",
    "low_10",
    "high_20",
    "low_20",
    "high_50",
    "low_50",
    "high_100",
    "low_100",
}


def build_causal_auxiliary_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build scale-free auxiliary features from the current and earlier bars."""
    indicators = add_indicators(prices, include_future=False)
    close = indicators["close"].replace(0, np.nan)
    volume = indicators["volume"].replace(0, np.nan)
    amount = indicators.get("amount", indicators["close"] * indicators["volume"])
    amount = pd.to_numeric(amount, errors="coerce").replace(0, np.nan)

    indicators["macd_pct"] = indicators["macd"] / close
    indicators["macd_signal_pct"] = indicators["macd_signal"] / close
    indicators["macd_hist_pct"] = indicators["macd_hist"] / close
    indicators["macd_hist_delta_pct"] = indicators["macd_hist_delta"] / close
    for period in (1, 2, 3, 5):
        indicators[f"amount_change_{period}"] = amount.pct_change(
            period, fill_method=None
        )
    for period in (1, 2, 3, 5):
        indicators[f"volume_change_{period}"] = volume.pct_change(
            period, fill_method=None
        )
    indicators["average_trade_price_gap"] = amount / volume / close - 1

    base_columns = [
        column
        for column in indicators.columns
        if column != "date"
        and column not in LEVEL_COLUMNS
        and not column.startswith("future_")
        and pd.api.types.is_numeric_dtype(indicators[column])
    ]
    causal = indicators[base_columns].copy()
    lagged = [causal]
    for lag in LAG_STEPS:
        lagged.append(causal.shift(lag).add_suffix(f"_lag_{lag}"))
    result = pd.concat(lagged, axis=1).astype(float)
    return result.mask(np.isinf(result))
