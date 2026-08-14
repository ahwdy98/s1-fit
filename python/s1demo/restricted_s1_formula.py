from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .auxiliary_signals import (
    _td_runs,
    bounded_continuity_counts,
    td_sequential_direction,
)
from .futu_metrics import futu_turnover_rate, futu_volume_ratio


@dataclass(frozen=True)
class RestrictedFormulaConfig:
    """Thresholds for the compact, hand-written comparison profile."""

    volume_ratio_lookback: int = 5
    event_volume_ratio_min: float = 2.60
    event_turnover_rate_min: float = 2.75
    event_s4_min: float = 4.00
    ib_volume_ratio_min: float = 1.20
    ib_turnover_rate_min: float = 1.00
    ib_s4_min: float = 2.00
    ib_prior_volume_growth_min: float = 0.10
    ib_prior_day_volume_change_min: float = -0.05
    ib_recovery_close_position_min: float = 0.85
    continuity_confirmation: int = 13
    continuity_terminal_minimum: int = 5
    continuity_terminal_maximum: int = 8
    continuity_terminal_volume_ratio_min: float = 1.00
    continuity_terminal_s4_median_max: float = 9.00


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def four_mean_oscillation(frame: pd.DataFrame) -> pd.Series:
    """Mean of the daily high-low percentage over the latest four bars."""
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    daily_oscillation = _safe_ratio(high - low, close) * 100.0
    return daily_oscillation.rolling(4, min_periods=4).mean().rename("s4")


def _optional_mask(
    values: np.ndarray | pd.Series | None,
    length: int,
    name: str,
) -> np.ndarray:
    if values is None:
        return np.zeros(length, dtype=bool)
    result = np.asarray(values, dtype=bool)
    if len(result) != length:
        raise ValueError(f"{name} length must match the price frame")
    return result


def restricted_ib_signal(
    frame: pd.DataFrame,
    volume_ratio: pd.Series,
    turnover_rate: pd.Series,
    s4: pd.Series,
    config: RestrictedFormulaConfig,
) -> pd.Series:
    """Identify constructive institutional buying without future bars."""
    open_values = pd.to_numeric(frame["open"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce").replace(0, np.nan)

    prior_volume_growth = _safe_ratio(volume.shift(1), volume.shift(3)) - 1.0
    prior_day_volume_change = _safe_ratio(volume.shift(1), volume.shift(2)) - 1.0
    constructive_sequence = (
        volume.gt(volume.shift(1))
        & prior_volume_growth.gt(config.ib_prior_volume_growth_min)
        & prior_day_volume_change.ge(config.ib_prior_day_volume_change_min)
        & close.shift(1).ge(open_values.shift(1))
        & close.shift(2).gt(open_values.shift(2))
        & turnover_rate.diff().shift(2).fillna(0.0).ge(0.0)
    )
    close_position = _safe_ratio(close - low, high - low)
    constructive_price = close.ge(open_values) & (
        close.ge(close.shift(1))
        | close_position.ge(config.ib_recovery_close_position_min)
    )
    active_market = (
        volume_ratio.ge(config.ib_volume_ratio_min)
        | turnover_rate.ge(config.ib_turnover_rate_min)
        | s4.ge(config.ib_s4_min)
    )
    return (constructive_sequence & constructive_price & active_market).fillna(False)


def _td_setup_count(condition: pd.Series, confirmation: int) -> np.ndarray:
    counts = np.zeros(len(condition), dtype=int)
    count = 0
    exhausted = False
    for index, active in enumerate(condition.fillna(False).to_numpy(bool)):
        if not active:
            count = 0
            exhausted = False
        elif exhausted:
            count = 0
        else:
            count = min(count + 1, confirmation)
            exhausted = count >= confirmation
        counts[index] = count
    return counts


def restricted_td_continuity(
    frame: pd.DataFrame,
    volume_ratio: pd.Series,
    s4: pd.Series,
    reset_mask: np.ndarray,
    cap: int,
    config: RestrictedFormulaConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct continuity from confirmed or currently active TD runs."""
    close = pd.to_numeric(frame["close"], errors="coerce")
    up = _td_setup_count(
        close.gt(close.shift(4)),
        config.continuity_confirmation,
    )
    down = _td_setup_count(
        close.lt(close.shift(4)),
        config.continuity_confirmation,
    )
    active = np.zeros(len(frame), dtype=bool)
    runs = [(*run, 1) for run in _td_runs(up)] + [(*run, -1) for run in _td_runs(down)]

    for start, end, maximum, _side in runs:
        if maximum >= config.continuity_confirmation:
            stop = end + 1
            while (
                stop < len(frame)
                and stop - start < cap
                and up[stop] == 0
                and down[stop] == 0
            ):
                stop += 1
            active[start:stop] = True
            continue

        terminal = (
            end == len(frame) - 1
            and config.continuity_terminal_minimum
            <= maximum
            <= config.continuity_terminal_maximum
            and (
                volume_ratio.iloc[end]
                >= config.continuity_terminal_volume_ratio_min
                or s4.iloc[start : end + 1].median()
                < config.continuity_terminal_s4_median_max
            )
        )
        if terminal:
            active[start : end + 1] = True

    td_features = pd.DataFrame(
        {"td_up_count": up, "td_down_count": down},
        index=frame.index,
    )
    direction = np.where(active, td_sequential_direction(td_features), 0)
    setup_restart = np.zeros(len(frame), dtype=bool)
    setup_restart[1:] = (
        (up[:-1] == 0) & (down[:-1] == 0) & ((up[1:] == 1) | (down[1:] == 1))
    )
    counts = bounded_continuity_counts(
        direction,
        reset_mask | setup_restart,
        cap,
    )
    return np.where(counts > 0, direction, 0), counts


def generate_restricted_formula_auxiliary_signals(
    frame: pd.DataFrame,
    reset_mask: np.ndarray | pd.Series | None = None,
    buy_trigger_mask: np.ndarray | pd.Series | None = None,
    sell_trigger_mask: np.ndarray | pd.Series | None = None,
    continuity_cap: int = 24,
    config: RestrictedFormulaConfig | None = None,
) -> pd.DataFrame:
    """Generate compact IB/E signals and TD-based continuity counts."""
    config = config or RestrictedFormulaConfig()
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if "turnover_rate" not in frame and "turnoverRate" not in frame:
        missing.add("turnoverRate")
    if missing:
        raise ValueError(f"Restricted formula requires columns: {sorted(missing)}")

    volume_ratio = futu_volume_ratio(frame, config.volume_ratio_lookback)
    turnover_rate = futu_turnover_rate(frame)
    s4 = four_mean_oscillation(frame)

    _optional_mask(buy_trigger_mask, len(frame), "buy_trigger_mask")
    _optional_mask(sell_trigger_mask, len(frame), "sell_trigger_mask")

    is_event = volume_ratio.ge(config.event_volume_ratio_min) & (
        turnover_rate.ge(config.event_turnover_rate_min) | s4.ge(config.event_s4_min)
    )
    is_ib = restricted_ib_signal(
        frame,
        volume_ratio,
        turnover_rate,
        s4,
        config,
    )
    reset = _optional_mask(reset_mask, len(frame), "reset_mask")
    direction, continuity = restricted_td_continuity(
        frame,
        volume_ratio,
        s4,
        reset,
        continuity_cap,
        config,
    )

    out = pd.DataFrame(index=frame.index)
    out["s4"] = s4
    out["volume_ratio"] = volume_ratio
    out["turnover_rate"] = turnover_rate
    out["is_ib"] = is_ib.fillna(False).to_numpy(bool)
    out["is_e"] = is_event.fillna(False).to_numpy(bool)
    out["continuity_dir"] = direction
    out["continuity"] = continuity
    out["ib_formula_reason"] = np.where(
        out["is_ib"], "constructive_volume_sequence+green_close+activity", ""
    )
    out["e_formula_reason"] = np.where(
        out["is_e"], "volumeRatio+(turnoverRate_or_s4)", ""
    )
    return out
