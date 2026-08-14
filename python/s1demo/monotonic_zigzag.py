from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .futu_metrics import futu_turnover_rate, futu_volume_ratio
from .indicators import add_indicators


@dataclass
class MonotonicCandidate:
    marker: int
    extreme: int
    side: str
    confirmed: bool = False
    kind: str = "standard"
    confirmation: int | None = None


@dataclass(frozen=True)
class RestrictedCandidateConfig:
    """Small set of ZigZag confirmation rules using activity and S4."""

    volume_ratio_lookback: int = 5
    bearish_spike_return_min: float = 0.15
    pending_volume_ratio_max: float = 2.50
    weak_pending_max_age: int = 3
    weak_pending_return_max: float = 0.03
    weak_pending_volume_ratio_max: float = 1.10
    weak_pending_turnover_rate_max: float = 1.50
    weak_pending_s4_max: float = 6.50
    pending_sell_reversal_min: float = 0.02
    pending_sell_low_s4_max: float = 2.00


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
            sells.append(
                MonotonicCandidate(index, index - 1, "S", True, kind, index)
            )

    return _lock_repeated_sells(buys, sells)


def generate_restricted_monotonic_candidates(
    frame: pd.DataFrame,
    buy_threshold: float = 0.08,
    sell_threshold: float = 0.10,
    minimum_history: int = 3,
    config: RestrictedCandidateConfig | None = None,
) -> list[MonotonicCandidate]:
    """Generate B/S from ZigZag plus volume, turnover and S4 confirmation."""
    config = config or RestrictedCandidateConfig()
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if "turnover_rate" not in frame and "turnoverRate" not in frame:
        missing.add("turnoverRate")
    if missing:
        raise ValueError(f"Restricted ZigZag requires columns: {sorted(missing)}")

    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(float)
    open_values = pd.to_numeric(frame["open"], errors="coerce").to_numpy(float)
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(float)
    volume_ratio, turnover_rate, s4 = _restricted_futu_inputs(
        frame, config.volume_ratio_lookback
    )
    buy_items, _ = _track_side(close, buy_threshold, "B")
    sell_items, _ = _track_side(close, sell_threshold, "S")
    buys = [
        item
        for item in buy_items
        if item.marker >= minimum_history
        and not _reject_restricted_buy(
            item,
            close,
            open_values,
            volume_ratio,
            turnover_rate,
            s4,
            config,
        )
    ]
    sells = [
        item
        for item in sell_items
        if item.marker >= minimum_history
        and not _reject_restricted_sell(
            item, close, open_values, high, low, s4, config
        )
    ]

    return _lock_repeated_sells(buys, sells)


def generate_removed_restricted_candidates(
    frame: pd.DataFrame,
    buy_threshold: float = 0.08,
    sell_threshold: float = 0.10,
    minimum_history: int = 3,
) -> list[MonotonicCandidate]:
    """Replay historical cutoffs and return B/S markers later withdrawn."""
    seen: dict[tuple[int, str], MonotonicCandidate] = {}
    for stop in range(minimum_history + 1, len(frame) + 1):
        prefix = frame.iloc[:stop]
        for candidate in generate_restricted_monotonic_candidates(
            prefix,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            minimum_history=minimum_history,
        ):
            seen.setdefault((candidate.marker, candidate.side), candidate)

    active = {
        (candidate.marker, candidate.side)
        for candidate in generate_restricted_monotonic_candidates(
            frame,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            minimum_history=minimum_history,
        )
    }
    return [
        candidate
        for key, candidate in sorted(seen.items())
        if key not in active
    ]


def _restricted_futu_inputs(
    frame: pd.DataFrame, lookback: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    volume_ratio = futu_volume_ratio(frame, lookback)
    turnover_rate = futu_turnover_rate(frame)
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce").replace(0, np.nan)
    s4 = (((high - low) / close) * 100.0).rolling(4, min_periods=4).mean()
    return volume_ratio, turnover_rate, s4


def _reject_restricted_buy(
    item: MonotonicCandidate,
    close: np.ndarray,
    open_values: np.ndarray,
    volume_ratio: pd.Series,
    turnover_rate: pd.Series,
    s4: pd.Series,
    config: RestrictedCandidateConfig,
) -> bool:
    index = item.marker
    return_1 = close[index] / close[index - 1] - 1.0
    if (
        return_1 > config.bearish_spike_return_min
        and close[index] < open_values[index]
    ):
        return True
    if item.confirmed:
        return False
    age = len(close) - 1 - index
    if volume_ratio.iloc[index] > config.pending_volume_ratio_max:
        return True
    return bool(
        age <= config.weak_pending_max_age
        and return_1 < config.weak_pending_return_max
        and volume_ratio.iloc[index] < config.weak_pending_volume_ratio_max
        and turnover_rate.iloc[index] < config.weak_pending_turnover_rate_max
        and s4.iloc[index] < config.weak_pending_s4_max
    )


def _reject_restricted_sell(
    item: MonotonicCandidate,
    close: np.ndarray,
    open_values: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    s4: pd.Series,
    config: RestrictedCandidateConfig,
) -> bool:
    if item.confirmed:
        return False
    index = item.marker
    age = len(close) - 1 - index
    partial_reversal = -(close[-1] / close[item.extreme] - 1.0)
    prior_low = np.min(close[max(0, index - 8) : index])
    prior_rebound = close[index - 1] / prior_low - 1.0
    candle_range = high[index] - low[index]
    close_position = (
        (close[index] - low[index]) / candle_range if candle_range > 0 else 1.0
    )
    terminal_turn = (
        age == 0
        and prior_rebound >= 0.10
        and close[index] < close[index - 1]
        and (close_position <= 0.30 or close[index] < open_values[index])
    )
    return bool(
        age == 0
        and partial_reversal < config.pending_sell_reversal_min
        and s4.iloc[index] > config.pending_sell_low_s4_max
        and not terminal_turn
    )


def _lock_repeated_sells(
    buys: list[MonotonicCandidate], sells: list[MonotonicCandidate]
) -> list[MonotonicCandidate]:
    """Allow another S only after an intervening B candidate."""
    candidates_by_marker: dict[int, dict[str, MonotonicCandidate]] = {}
    for candidate in buys + sells:
        slot = candidates_by_marker.setdefault(candidate.marker, {})
        current = slot.get(candidate.side)
        if current is None or (candidate.confirmed and not current.confirmed):
            slot[candidate.side] = candidate

    accepted: list[MonotonicCandidate] = []
    sell_locked = False
    for marker in sorted(candidates_by_marker):
        slot = candidates_by_marker[marker]
        buy = slot.get("B")
        sell = slot.get("S")
        if buy is not None:
            accepted.append(buy)
            sell_locked = False
        if sell is not None and not sell_locked:
            accepted.append(sell)
            sell_locked = True
    return accepted


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
    calm_bullish_rebound = (
        row["atr_pct"] <= 0.025
        and 0.015 <= row["ret_1"] <= 0.03
        and row["body_pct"] > 0
        and row["close_pos"] >= 0.60
    )
    strong_capitulation_rebound = (
        row["ret_1"] >= 0.04
        and row["body_pct"] > 0
        and row["close_pos"] >= 0.40
    )
    return bool(
        (row["ret_1"] > 0.15 and row["body_pct"] < 0)
        or (
            row["atr_pct"] <= 0.04
            and row["ret_20"] > -0.02
            and row["volume_ratio_50"] <= 2.59
            and not calm_bullish_rebound
        )
        or (
            2.59 < row["volume_ratio_50"] < 3.0
            and row["ret_5"] <= -0.13
            and not strong_capitulation_rebound
        )
        or (
            row["dollar_volume_ratio_20"] > 3.25
            and row["close_pos"] > 0.72
            and row["breakout_20"] < -0.15
        )
    )


def _reject_sell(features: pd.DataFrame, index: int) -> bool:
    row = features.iloc[index]
    strong_momentum_peak = row["ret_20"] > 0.50
    return bool(
        (
            -0.02 < row["ret_1"] < 0
            and row["range_pct"] <= 0.03
            and row["upper_wick_pct"] > 0.41
            and row["close_pos"] > 0.20
        )
        or (
            row["ret_1"] > -0.005
            and row["body_pct"] > 0.015
            and row["close_pos"] > 0.60
            and not strong_momentum_peak
        )
        or (
            row["ret_1"] > -0.02
            and row["body_pct"] > 0.04
            and not strong_momentum_peak
        )
    )
