from __future__ import annotations

import numpy as np
import pandas as pd


def futu_volume_ratio(frame: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Return Futu volumeRatio or its daily-close historical proxy.

    Futu exposes volumeRatio on quote snapshots but not historical K-line rows.
    At the daily close, current volume divided by the mean volume of the five
    preceding complete sessions is the corresponding historical proxy.
    """
    if lookback < 1:
        raise ValueError("volumeRatio lookback must be positive")
    volume = pd.to_numeric(frame["volume"], errors="coerce").replace(0, np.nan)
    previous_mean = volume.shift(1).rolling(
        lookback, min_periods=lookback
    ).mean()
    proxy = (volume / previous_mean.replace(0, np.nan)).rename("volume_ratio")
    direct_column = next(
        (name for name in ("volume_ratio", "volumeRatio") if name in frame),
        None,
    )
    if direct_column is None:
        return proxy
    direct = pd.to_numeric(frame[direct_column], errors="coerce")
    return direct.combine_first(proxy).rename("volume_ratio")


def futu_turnover_rate(frame: pd.DataFrame) -> pd.Series:
    """Return normalized Futu turnoverRate: 1.0 means one percent.

    Snapshot ``turnoverRate`` is already percentage-form. Callers ingesting
    decimal-form K-line values must normalize them to percentage-form first.
    """
    column = next(
        (name for name in ("turnover_rate", "turnoverRate") if name in frame),
        None,
    )
    if column is None:
        raise ValueError("Futu turnoverRate field is required")
    return pd.to_numeric(frame[column], errors="coerce").rename(
        "turnover_rate"
    )
