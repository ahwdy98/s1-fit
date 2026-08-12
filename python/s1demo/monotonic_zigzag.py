from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import add_indicators


@dataclass
class MonotonicCandidate:
    marker: int
    extreme: int
    side: str
    confirmed: bool = False
    kind: str = "standard"
    confirmation: int | None = None


def generate_monotonic_candidates(
    frame: pd.DataFrame,
    buy_threshold: float = 0.08,
    sell_threshold: float = 0.10,
    minimum_history: int = 3,
) -> list[MonotonicCandidate]:
    """Return B/S candidates that can disappear but never backfill or return."""
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(float)
    buy_items, _ = _track_side(close, buy_threshold, "B")
    sell_items, _ = _track_side(close, sell_threshold, "S")
    buys = [item for item in buy_items if item.marker >= minimum_history]
    sells = [item for item in sell_items if item.marker >= minimum_history]
    features = add_indicators(frame)

    buys = [item for item in buys if not _reject_buy(features, item.marker)]
    sells = [item for item in sells if not _reject_sell(features, item.marker)]

    filtered_sells: list[MonotonicCandidate] = []
    buy_markers = {item.marker for item in buys}
    sell_by_marker = {item.marker: item for item in sells}
    locked = False
    for index in range(len(close)):
        if index in buy_markers:
            locked = False
        if index in sell_by_marker and not locked:
            filtered_sells.append(sell_by_marker[index])
            locked = True

    open_values = pd.to_numeric(frame["open"], errors="coerce").to_numpy(float)
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(float)
    volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0).to_numpy(float)

    for index in range(1, len(close)):
        age = len(close) - 1 - index
        if age > 6:
            continue
        lookback_start = max(0, index - 8)
        previous_is_low = (
            close[index - 1] <= np.min(close[lookback_start:index]) * 1.000001
        )
        return_1 = close[index] / close[index - 1] - 1.0
        worst_after_return = np.min(close[index:]) / close[index] - 1.0
        if (
            previous_is_low
            and return_1 >= 0.05
            and close[index] > open_values[index]
            and return_1 <= 0.152901
            and worst_after_return > -0.173247
            and not _reject_buy(features, index)
        ):
            buys.append(
                MonotonicCandidate(index, index - 1, "B", True, "fallback", index)
            )

    for index in range(13, len(close)):
        age = len(close) - 1 - index
        if age > 10:
            continue
        previous_close = close[index - 1]
        previous_is_high = previous_close >= np.max(close[index - 13 : index]) * 0.999
        prior_run = previous_close / close[index - 9] - 1.0
        return_1 = close[index] / previous_close - 1.0
        candle_range = high[index] - low[index]
        close_position = (
            (close[index] - low[index]) / candle_range if candle_range > 0 else 1.0
        )
        history_volume = volume[max(0, index - 20) : index]
        average_volume = float(np.mean(history_volume)) if len(history_volume) else 0.0
        volume_ratio = volume[index] / average_volume if average_volume > 0 else 0.0
        red = close[index] < open_values[index]
        strong = (
            -0.05 <= return_1 <= -0.01 and close_position <= 0.2 and volume_ratio <= 1.2
        )
        weak = (
            -0.01 <= return_1 <= -0.001
            and 0.3 <= close_position <= 0.5
            and 1.0 <= volume_ratio <= 1.2
        )
        prior_low = np.min(close[max(0, index - 8) : index])
        terminal = (
            age == 0
            and previous_close / prior_low - 1.0 >= 0.10
            and -0.02 <= return_1 < 0
            and (close_position <= 0.3 or red)
        )
        regular = previous_is_high and prior_run >= 0.01 and red and (strong or weak)
        if regular or terminal:
            kind = "turning" if regular else "terminal_turning"
            filtered_sells.append(
                MonotonicCandidate(index, index - 1, "S", True, kind, index)
            )

    return buys + filtered_sells


def _track_side(close: np.ndarray, threshold: float, expected_side: str):
    if not len(close):
        return [], set()
    completed: list[MonotonicCandidate] = []
    born: set[tuple[int, str]] = set()
    direction = 0
    low_index = high_index = 0
    candidate_b = candidate_s = None
    for index in range(1, len(close)):
        if close[index] < close[low_index]:
            low_index = index
            candidate_b = None
        if close[index] > close[high_index]:
            high_index = index
            candidate_s = None
        if direction == 0:
            if index == low_index + 1 and candidate_b is None:
                candidate_b = MonotonicCandidate(index, low_index, "B")
                born.add((index, "B"))
            if index == high_index + 1 and candidate_s is None:
                candidate_s = MonotonicCandidate(index, high_index, "S")
                born.add((index, "S"))
            if close[index] >= close[low_index] * (1 + threshold):
                candidate_b = candidate_b or MonotonicCandidate(
                    low_index + 1, low_index, "B"
                )
                candidate_b.confirmed = True
                candidate_b.confirmation = index
                completed.append(candidate_b)
                candidate_s = None
                direction = 1
                high_index = index
            elif close[index] <= close[high_index] * (1 - threshold):
                candidate_s = candidate_s or MonotonicCandidate(
                    high_index + 1, high_index, "S"
                )
                candidate_s.confirmed = True
                candidate_s.confirmation = index
                completed.append(candidate_s)
                candidate_b = None
                direction = -1
                low_index = index
        elif direction == 1:
            if index == high_index + 1 and candidate_s is None:
                candidate_s = MonotonicCandidate(index, high_index, "S")
                born.add((index, "S"))
            if close[index] <= close[high_index] * (1 - threshold):
                candidate_s = candidate_s or MonotonicCandidate(
                    high_index + 1, high_index, "S"
                )
                candidate_s.confirmed = True
                candidate_s.confirmation = index
                completed.append(candidate_s)
                direction = -1
                low_index = index
                candidate_b = None
        else:
            if index == low_index + 1 and candidate_b is None:
                candidate_b = MonotonicCandidate(index, low_index, "B")
                born.add((index, "B"))
            if close[index] >= close[low_index] * (1 + threshold):
                candidate_b = candidate_b or MonotonicCandidate(
                    low_index + 1, low_index, "B"
                )
                candidate_b.confirmed = True
                candidate_b.confirmation = index
                completed.append(candidate_b)
                direction = 1
                high_index = index
                candidate_s = None
    active = [item for item in completed if item.side == expected_side]
    pending = candidate_b if expected_side == "B" else candidate_s
    if pending is not None:
        active.append(pending)
    return active, born


def _reject_buy(features: pd.DataFrame, index: int) -> bool:
    row = features.iloc[index]
    return bool(
        (row["ret_1"] > 0.15 and row["body_pct"] < 0)
        or (
            row["atr_pct"] <= 0.04
            and row["ret_20"] > -0.02
            and row["volume_ratio_50"] <= 2.59
        )
        or (2.59 < row["volume_ratio_50"] < 3.0 and row["ret_5"] <= -0.13)
        or (
            row["dollar_volume_ratio_20"] > 3.25
            and row["close_pos"] > 0.72
            and row["breakout_20"] < -0.15
        )
    )


def _reject_sell(features: pd.DataFrame, index: int) -> bool:
    row = features.iloc[index]
    return bool(
        (
            -0.02 < row["ret_1"] < 0
            and row["range_pct"] <= 0.03
            and row["upper_wick_pct"] > 0.41
        )
        or (
            row["ret_1"] > -0.005
            and row["body_pct"] > 0.015
            and row["close_pos"] > 0.60
        )
        or (row["ret_1"] > -0.02 and row["body_pct"] > 0.04)
    )
