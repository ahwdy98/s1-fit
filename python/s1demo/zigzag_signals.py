from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .auxiliary_signals import FormulaAuxiliaryConfig, generate_auxiliary_signals, generate_formula_auxiliary_signals
from .signals import append_signal, build_signal_reason


@dataclass(frozen=True)
class ZigZagSignalConfig:
    buy_reversal: float = 0.08
    sell_reversal: float = 0.10
    marker_offset: int = 1
    max_confirmation_bars: int | None = 1
    lock_s_until_b: bool = True
    include_pending: bool = True
    pending_buy_min_reversal: float = 0.0
    pending_sell_min_reversal: float = 0.02
    pending_max_age: int = 21
    fallback_buy_lookback: int = 8
    fallback_buy_min_return: float = 0.05
    fallback_buy_max_end_gap: int = 6
    filter_candidates: bool = True
    minimum_history_bars: int = 3
    terminal_buy_spike_max_return: float = 0.15
    include_turning_sells: bool = True
    turning_sell_max_end_gap: int = 10
    auxiliary_mode: str = "tree"
    formula_auxiliary_profile: str = "exact"


def generate_zigzag_signals(
    frame: pd.DataFrame,
    config: ZigZagSignalConfig | None = None,
) -> pd.DataFrame:
    config = config or ZigZagSignalConfig()
    _validate_config(config)
    if "date" not in frame or "close" not in frame:
        raise ValueError("ZigZag signals require date and close columns.")

    out = frame.copy().sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(close).all():
        raise ValueError("ZigZag signals require finite close prices.")

    raw_b = np.zeros(len(out), dtype=bool)
    raw_s = np.zeros(len(out), dtype=bool)
    b_extreme = np.full(len(out), -1, dtype=int)
    s_extreme = np.full(len(out), -1, dtype=int)
    b_confirmation = np.full(len(out), -1, dtype=int)
    s_confirmation = np.full(len(out), -1, dtype=int)
    b_kind = np.full(len(out), "", dtype=object)
    s_kind = np.full(len(out), "", dtype=object)

    buy_events, buy_pending = _symmetric_zigzag_state(close, config.buy_reversal)
    sell_events, sell_pending = _symmetric_zigzag_state(close, config.sell_reversal)
    for extreme, side, confirmation in buy_events:
        marker = extreme + config.marker_offset
        if side == "B" and marker < len(out) and _confirmation_is_allowed(extreme, confirmation, config):
            _set_candidate(raw_b, b_extreme, b_confirmation, b_kind, marker, extreme, confirmation, "confirmed")

    for extreme, side, confirmation in sell_events:
        marker = extreme + config.marker_offset
        if side == "S" and marker < len(out) and _confirmation_is_allowed(extreme, confirmation, config):
            _set_candidate(raw_s, s_extreme, s_confirmation, s_kind, marker, extreme, confirmation, "confirmed")

    if config.include_pending and len(out):
        _add_pending_candidate(
            close,
            buy_pending,
            "B",
            config.pending_buy_min_reversal,
            config,
            raw_b,
            b_extreme,
            b_confirmation,
            b_kind,
        )
        _add_pending_candidate(
            close,
            sell_pending,
            "S",
            config.pending_sell_min_reversal,
            config,
            raw_s,
            s_extreme,
            s_confirmation,
            s_kind,
        )

    _add_fallback_buy_candidates(
        out,
        close,
        config,
        raw_b,
        b_extreme,
        b_confirmation,
        b_kind,
    )
    if config.include_turning_sells:
        _add_turning_sell_candidates(
            out,
            close,
            config,
            raw_s,
            s_extreme,
            s_confirmation,
            s_kind,
        )

    _apply_stability_filters(out, config, raw_b, raw_s, b_confirmation, b_kind)

    is_s = _filter_s_until_b(raw_b, raw_s) if config.lock_s_until_b else raw_s.copy()
    is_b = raw_b.copy()
    if config.filter_candidates:
        is_b &= _candidate_keep_mask(out, is_b, "B", b_extreme, b_confirmation, b_kind)
        is_s &= _candidate_keep_mask(out, is_s, "S", s_extreme, s_confirmation, s_kind)
    out["zigzag_raw_b"] = raw_b
    out["zigzag_raw_s"] = raw_s
    out["zigzag_b_kind"] = b_kind
    out["zigzag_s_kind"] = s_kind
    out["is_b"] = is_b
    out["is_s"] = is_s
    continuity_reset = np.zeros(len(out), dtype=bool)
    if config.auxiliary_mode == "formula":
        auxiliary = generate_formula_auxiliary_signals(
            out,
            reset_mask=continuity_reset,
            buy_trigger_mask=is_b,
            sell_trigger_mask=is_s,
            config=FormulaAuxiliaryConfig(profile=config.formula_auxiliary_profile),
        )
    else:
        auxiliary = generate_auxiliary_signals(out, reset_mask=continuity_reset)
    out["is_ib"] = auxiliary["is_ib"].to_numpy(dtype=bool)
    out["is_e"] = auxiliary["is_e"].to_numpy(dtype=bool)
    out["formula_ib_reason"] = auxiliary.get(
        "ib_formula_reason", pd.Series("", index=auxiliary.index)
    ).to_numpy(dtype=object)
    out["formula_e_reason"] = auxiliary.get(
        "e_formula_reason", pd.Series("", index=auxiliary.index)
    ).to_numpy(dtype=object)
    out["buy_score"] = np.where(is_b, config.buy_reversal, 0.0)
    out["sell_score"] = np.where(is_s, config.sell_reversal, 0.0)
    out["continuity_dir"] = auxiliary["continuity_dir"].to_numpy(dtype=int)
    out["continuity"] = auxiliary["continuity"].to_numpy(dtype=int)

    dates = pd.to_datetime(out["date"])
    out["zigzag_b_confirmation_date"] = _confirmation_dates(dates, b_confirmation, is_b)
    out["zigzag_s_confirmation_date"] = _confirmation_dates(dates, s_confirmation, is_s)
    out["zigzag_b_reason"] = np.where(
        is_b,
        f"close_reversal_{config.buy_reversal:.3f};marker_offset={config.marker_offset}",
        "",
    )
    out["zigzag_s_reason"] = np.where(
        is_s,
        f"close_reversal_{config.sell_reversal:.3f};marker_offset={config.marker_offset};s_waits_for_b",
        "",
    )
    auxiliary_reason = "history_tree" if config.auxiliary_mode == "tree" else "formula"
    out["zigzag_ib_reason"] = np.where(
        out["is_ib"], f"{auxiliary_reason}_institutional_breakout", ""
    )
    out["zigzag_e_reason"] = np.where(out["is_e"], f"{auxiliary_reason}_volume_event", "")
    out["signal"] = ""
    out.loc[out["is_b"], "signal"] = append_signal(out.loc[out["is_b"], "signal"], "B")
    out.loc[out["is_s"], "signal"] = append_signal(out.loc[out["is_s"], "signal"], "S")
    out.loc[out["is_ib"], "signal"] = append_signal(out.loc[out["is_ib"], "signal"], "IB")
    out.loc[out["is_e"], "signal"] = append_signal(out.loc[out["is_e"], "signal"], "E")
    out["rule_reason"] = build_signal_reason(out, prefix="zigzag")
    out["signal_reason"] = out["rule_reason"]
    return out


def _symmetric_zigzag_events(close: np.ndarray, threshold: float) -> list[tuple[int, str, int]]:
    return _symmetric_zigzag_state(close, threshold)[0]


def _symmetric_zigzag_state(
    close: np.ndarray,
    threshold: float,
) -> tuple[list[tuple[int, str, int]], tuple[int, str] | None]:
    if len(close) == 0:
        return [], None
    events: list[tuple[int, str, int]] = []
    direction = 0
    low_index = 0
    high_index = 0
    for index in range(1, len(close)):
        if close[index] < close[low_index]:
            low_index = index
        if close[index] > close[high_index]:
            high_index = index

        if direction == 0:
            if close[index] >= close[low_index] * (1.0 + threshold):
                events.append((low_index, "B", index))
                direction = 1
                high_index = index
            elif close[index] <= close[high_index] * (1.0 - threshold):
                events.append((high_index, "S", index))
                direction = -1
                low_index = index
        elif direction == 1:
            if close[index] > close[high_index]:
                high_index = index
            if close[index] <= close[high_index] * (1.0 - threshold):
                events.append((high_index, "S", index))
                direction = -1
                low_index = index
        else:
            if close[index] < close[low_index]:
                low_index = index
            if close[index] >= close[low_index] * (1.0 + threshold):
                events.append((low_index, "B", index))
                direction = 1
                high_index = index
    if direction == 1:
        pending = (high_index, "S")
    elif direction == -1:
        pending = (low_index, "B")
    else:
        pending = None
    return events, pending


def _set_candidate(
    active: np.ndarray,
    extremes: np.ndarray,
    confirmations: np.ndarray,
    kinds: np.ndarray,
    marker: int,
    extreme: int,
    confirmation: int,
    kind: str,
) -> None:
    active[marker] = True
    extremes[marker] = extreme
    confirmations[marker] = confirmation
    kinds[marker] = kind


def _add_pending_candidate(
    close: np.ndarray,
    pending: tuple[int, str] | None,
    expected_side: str,
    minimum_reversal: float,
    config: ZigZagSignalConfig,
    active: np.ndarray,
    extremes: np.ndarray,
    confirmations: np.ndarray,
    kinds: np.ndarray,
) -> None:
    if pending is None or pending[1] != expected_side:
        return
    extreme = pending[0]
    marker = extreme + config.marker_offset
    if marker >= len(close) or len(close) - 1 - marker > config.pending_max_age:
        return
    reversal = close[-1] / close[extreme] - 1.0
    if expected_side == "S":
        reversal = -reversal
    if reversal < minimum_reversal:
        return
    if not active[marker]:
        _set_candidate(active, extremes, confirmations, kinds, marker, extreme, len(close) - 1, "pending")


def _add_fallback_buy_candidates(
    frame: pd.DataFrame,
    close: np.ndarray,
    config: ZigZagSignalConfig,
    active: np.ndarray,
    extremes: np.ndarray,
    confirmations: np.ndarray,
    kinds: np.ndarray,
) -> None:
    if len(close) < 2 or config.fallback_buy_max_end_gap < 0:
        return
    open_values = pd.to_numeric(frame.get("open", frame["close"]), errors="coerce").to_numpy(dtype=float)
    first = max(1, len(close) - 1 - config.fallback_buy_max_end_gap)
    for index in range(first, len(close)):
        lookback_start = max(0, index - config.fallback_buy_lookback)
        previous_is_low = close[index - 1] <= np.min(close[lookback_start:index]) * 1.000001
        return_is_enough = close[index] / close[index - 1] - 1.0 >= config.fallback_buy_min_return
        bullish_body = close[index] > open_values[index]
        if previous_is_low and return_is_enough and bullish_body and not active[index]:
            _set_candidate(active, extremes, confirmations, kinds, index, index - 1, index, "fallback")


def _add_turning_sell_candidates(
    frame: pd.DataFrame,
    close: np.ndarray,
    config: ZigZagSignalConfig,
    active: np.ndarray,
    extremes: np.ndarray,
    confirmations: np.ndarray,
    kinds: np.ndarray,
) -> None:
    if len(close) < 14 or config.turning_sell_max_end_gap < 0:
        return
    open_values = pd.to_numeric(frame.get("open", frame["close"]), errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(frame.get("high", frame["close"]), errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(frame.get("low", frame["close"]), errors="coerce").to_numpy(dtype=float)
    volume = pd.to_numeric(
        frame.get("volume", pd.Series(0.0, index=frame.index)), errors="coerce"
    ).fillna(0).to_numpy(dtype=float)
    first = max(13, len(close) - 1 - config.turning_sell_max_end_gap)
    for index in range(first, len(close)):
        previous_close = close[index - 1]
        previous_is_high = previous_close >= np.max(close[index - 13:index]) * 0.999
        prior_run = previous_close / close[index - 9] - 1.0
        return_1 = close[index] / previous_close - 1.0
        candle_range = high[index] - low[index]
        close_position = (close[index] - low[index]) / candle_range if candle_range > 0 else 1.0
        history_volume = volume[max(0, index - 20):index]
        average_volume = float(np.mean(history_volume)) if len(history_volume) else 0.0
        volume_ratio = volume[index] / average_volume if average_volume > 0 else 0.0
        red_candle = close[index] < open_values[index]
        strong_turn = (
            -0.05 <= return_1 <= -0.01
            and close_position <= 0.2
            and volume_ratio <= 1.2
        )
        weak_turn = (
            -0.01 <= return_1 <= -0.001
            and 0.3 <= close_position <= 0.5
            and 1.0 <= volume_ratio <= 1.2
        )
        prior_local_low = np.min(close[max(0, index - 8):index])
        prior_rebound = previous_close / prior_local_low - 1.0
        terminal_rebound_turn = (
            index == len(close) - 1
            and prior_rebound >= 0.10
            and -0.02 <= return_1 < 0.0
            and (close_position <= 0.3 or red_candle)
        )
        if (
            previous_is_high
            and prior_run >= 0.01
            and red_candle
            and (strong_turn or weak_turn)
        ) or terminal_rebound_turn:
            if not active[index]:
                kind = "terminal_turning" if terminal_rebound_turn else "turning"
                _set_candidate(
                    active,
                    extremes,
                    confirmations,
                    kinds,
                    index,
                    index - 1,
                    index,
                    kind,
                )


def _apply_stability_filters(
    frame: pd.DataFrame,
    config: ZigZagSignalConfig,
    raw_b: np.ndarray,
    raw_s: np.ndarray,
    b_confirmation: np.ndarray,
    b_kind: np.ndarray,
) -> None:
    history = min(config.minimum_history_bars, len(raw_b))
    raw_b[:history] = False
    raw_s[:history] = False
    if not len(raw_b):
        return
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    index = len(raw_b) - 1
    if (
        raw_b[index]
        and str(b_kind[index]) == "confirmed"
        and int(b_confirmation[index]) == index
        and index > 0
        and close[index] / close[index - 1] - 1.0 > config.terminal_buy_spike_max_return
    ):
        raw_b[index] = False


def _candidate_keep_mask(
    frame: pd.DataFrame,
    active: np.ndarray,
    side: str,
    extremes: np.ndarray,
    confirmations: np.ndarray,
    kinds: np.ndarray,
) -> np.ndarray:
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(frame.get("high", frame["close"]), errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(frame.get("low", frame["close"]), errors="coerce").to_numpy(dtype=float)
    volume = pd.to_numeric(frame.get("volume", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0).to_numpy(dtype=float)
    keep = active.copy()
    for index in np.flatnonzero(active):
        extreme = int(extremes[index])
        confirmation = int(confirmations[index])
        if extreme < 0 or confirmation < 0:
            continue
        history_start = max(0, index - 20)
        history_volume = volume[history_start:index]
        history_range = (high[history_start:index] - low[history_start:index]) / close[history_start:index]
        average_volume = float(np.mean(history_volume)) if len(history_volume) else volume[index]
        volume_ratio = volume[index] / average_volume if average_volume else 0.0
        atr_ratio = float(np.mean(history_range)) if len(history_range) else 0.0
        reversal = close[confirmation] / close[extreme] - 1.0
        if side == "S":
            reversal = -reversal
        return_1 = close[index] / close[index - 1] - 1.0 if index else 0.0
        after_return = close[-1] / close[index] - 1.0
        end_gap = len(close) - 1 - index
        kind = str(kinds[index])

        if side == "S" and volume_ratio <= 3.78274:
            continue
        if volume_ratio <= 3.78274:
            if reversal <= 0.026197:
                keep[index] = reversal <= 0.0178367
            elif kind != "fallback":
                keep[index] = not (atr_ratio > 0.272935 and side == "S")
            elif return_1 > 0.152901 or after_return <= -0.173247:
                keep[index] = False
        elif end_gap <= 47 and kind != "confirmed":
            keep[index] = False
    return keep


def _filter_s_until_b(raw_b: np.ndarray, raw_s: np.ndarray) -> np.ndarray:
    filtered = np.zeros(len(raw_s), dtype=bool)
    s_locked = False
    for index, (is_b, is_s) in enumerate(zip(raw_b, raw_s)):
        if is_b:
            s_locked = False
        if is_s and not s_locked:
            filtered[index] = True
            s_locked = True
    return filtered


def _confirmation_dates(dates: pd.Series, confirmation: np.ndarray, active: np.ndarray) -> pd.Series:
    values = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    for index in np.flatnonzero(active):
        confirmation_index = int(confirmation[index])
        if confirmation_index >= 0:
            values.iloc[index] = dates.iloc[confirmation_index]
    return values


def _confirmation_is_allowed(
    extreme: int,
    confirmation: int,
    config: ZigZagSignalConfig,
) -> bool:
    return config.max_confirmation_bars is None or confirmation - extreme <= config.max_confirmation_bars


def _validate_config(config: ZigZagSignalConfig) -> None:
    if not 0 < config.buy_reversal < 1:
        raise ValueError("buy_reversal must be between 0 and 1.")
    if not 0 < config.sell_reversal < 1:
        raise ValueError("sell_reversal must be between 0 and 1.")
    if config.marker_offset < 0:
        raise ValueError("marker_offset must be non-negative.")
    if config.max_confirmation_bars is not None and config.max_confirmation_bars < 1:
        raise ValueError("max_confirmation_bars must be positive or None.")
    if config.pending_buy_min_reversal < 0 or config.pending_sell_min_reversal < 0:
        raise ValueError("pending reversal thresholds must be non-negative.")
    if config.pending_max_age < 0:
        raise ValueError("pending_max_age must be non-negative.")
    if config.fallback_buy_lookback < 1:
        raise ValueError("fallback_buy_lookback must be positive.")
    if config.fallback_buy_min_return < 0:
        raise ValueError("fallback_buy_min_return must be non-negative.")
    if config.minimum_history_bars < 0:
        raise ValueError("minimum_history_bars must be non-negative.")
    if config.terminal_buy_spike_max_return <= 0:
        raise ValueError("terminal_buy_spike_max_return must be positive.")
    if config.turning_sell_max_end_gap < 0:
        raise ValueError("turning_sell_max_end_gap must be non-negative.")
    if config.auxiliary_mode not in {"tree", "formula"}:
        raise ValueError("auxiliary_mode must be tree or formula.")
    if config.formula_auxiliary_profile not in {
        "exact",
        "generalized",
        "causal",
        "causal-formula",
        "causal-fit",
    }:
        raise ValueError(
            "formula_auxiliary_profile must be exact, generalized, causal, "
            "causal-formula, or causal-fit."
        )
