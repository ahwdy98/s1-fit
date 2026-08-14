from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .auxiliary_tree_model import FEATURE_NAMES, IMPUTER_MEDIANS, predict
from .causal_auxiliary_features import build_causal_auxiliary_features
from .causal_auxiliary_tree_model import predict as predict_causal_tree
from .indicators import add_indicators


@dataclass(frozen=True)
class FormulaAuxiliaryConfig:
    profile: str = "exact"
    ib_return: float = 0.06
    ib_body: float = 0.03
    ib_close_position: float = 0.70
    ib_breakout_20: float = -0.01
    ib_volume_ratio: float = 1.50
    ib_amount_ratio: float = 2.00
    ib_peak_lookback: int = 20
    ib_peak_return: float = 0.05
    ib_peak_close_position: float = 0.85
    ib_peak_amount_ratio: float = 1.60
    ib_quiet_return_min: float = 0.02
    ib_quiet_return_max: float = 0.04
    ib_quiet_breakout: float = 0.02
    ib_quiet_turnover_lookback: int = 5
    ib_amount_rise_bars: int = 3
    ib_cmf_filter_window: int = 10
    ib_cmf_filter_min: float = 0.10
    ib_accumulation_cmf_window: int = 10
    ib_accumulation_cmf: float = 0.50
    ib_accumulation_amount_ratio: float = 1.50
    ib_accumulation_return_min: float = 0.03
    ib_bottom_breakout_10_max: float = -0.0019
    ib_bottom_ema_34_89_max: float = -0.05
    ib_bottom_body_min: float = 0.0125
    ib_bottom_body_max: float = 0.02
    ib_bottom_amount_rank_min: float = 0.575
    ib_bottom_amount_rank_max: float = 0.71
    ib_bottom_next_return_min: float = -0.03
    ib_exhaustion_close_position_max: float = 0.991
    ib_exhaustion_ema_20_slope_max: float = 0.21
    ib_exhaustion_amount_z_max: float = 3.80
    ib_structural_gap_reprice_min: float = 0.15
    ib_structural_weak_return_max: float = 0.10
    ib_structural_weak_amount_ratio_max: float = 3.50
    ib_structural_reversal_return_max: float = -0.03
    ib_structural_reversal_amount_min: float = 0.90
    ib_structural_extreme_close_min: float = 0.94
    ib_structural_extreme_body_min: float = 0.08
    ib_structural_extreme_amount_ratio_min: float = 2.50
    ib_event_overlap_cmf_min: float = 0.25
    ib_flow_amount_increase_min: float = 1.05
    ib_flow_amount_increase_max: float = 1.70
    ib_flow_cmf3_delta_min: float = 0.20
    ib_flow_cmf5_delta_min: float = 0.10
    ib_flow_breakout_20_min: float = -0.01
    ib_flow_next_amount_ratio_max: float = 1.10
    ib_flow_return_min: float = 0.02
    ib_flow_return_max: float = 0.07
    ib_flow_rsi_max: float = 75.0
    ib_flow_effort_min: float = 0.60
    ib_flow_failed_next_return_max: float = -0.03
    ib_flow_failed_next_amount_max: float = 0.80
    ib_absorption_return_min: float = 0.03
    ib_absorption_return_max: float = 0.09
    ib_absorption_body_min: float = 0.03
    ib_absorption_body_max: float = 0.10
    ib_absorption_close_position_min: float = 0.55
    ib_absorption_close_position_max: float = 0.80
    ib_absorption_amount_ratio_min: float = 1.00
    ib_absorption_amount_ratio_max: float = 1.50
    ib_absorption_cmf3_delta_max: float = 0.0
    ib_absorption_narrow_range_max: float = 1.0
    ib_absorption_cmf5_delta_min: float = 0.0
    ib_absorption_short_history_gap_min: float = 0.03
    ib_absorption_failed_return_max: float = -0.05
    ib_absorption_failed_range_ratio_max: float = 1.00
    ib_absorption_failed_breakout_10_max: float = 0.00
    ib_absorption_breakout_return_min: float = 0.09
    ib_absorption_breakout_return_max: float = 0.10
    ib_absorption_breakout_close_position_min: float = 0.80
    ib_absorption_breakout_close_position_max: float = 0.82
    ib_absorption_breakout_10_min: float = 0.05
    ib_absorption_breakout_range_ratio_min: float = 1.50
    ib_absorption_breakout_next_return_max: float = -0.05
    ib_strong_return_min: float = 0.18
    ib_strong_body_min: float = 0.10
    ib_strong_close_position_min: float = 0.90
    ib_strong_breakout_20_min: float = 0.05
    ib_strong_volume_ratio_min: float = 1.30
    ib_strong_amount_ratio_min: float = 1.90
    ib_recovery_return_min: float = 0.08
    ib_recovery_return_max: float = 0.12
    ib_recovery_return_3_min: float = 0.20
    ib_recovery_body_min: float = 0.10
    ib_recovery_close_position_min: float = 0.60
    ib_recovery_breakout_20_max: float = -0.10
    ib_recovery_volume_ratio_min: float = 1.30
    ib_recovery_amount_ratio_min: float = 1.30
    ib_recovery_ema_20_slope_max: float = 0.0
    ib_quiet_run_bars: int = 3
    ib_quiet_run_return_min: float = 0.04
    ib_quiet_run_current_return_max: float = 0.015
    ib_quiet_run_body_max: float = 0.01
    ib_quiet_run_close_position_min: float = 0.60
    ib_quiet_run_breakout_max: float = 0.01
    ib_quiet_run_volume_ratio_max: float = 1.20
    ib_quiet_run_cmf_min: float = 0.25
    ib_event_overlap_return_min: float = 0.15
    ib_event_overlap_body_min: float = 0.10
    ib_sequence_prior_volume_growth_min: float = 0.10
    ib_sequence_prior_day_volume_change_min: float = -0.05
    ib_sequence_turnover_delta_min: float = 0.0
    ib_climactic_failure_body_min: float = 0.10
    ib_climactic_failure_next_return_max: float = -0.05
    ib_two_day_previous_return_min: float = 0.08
    ib_two_day_return_min: float = 0.08
    ib_two_day_return_max: float = 0.12
    ib_two_day_body_min: float = 0.05
    ib_two_day_close_position_min: float = 0.85
    ib_two_day_breakout_min: float = 0.05
    ib_two_day_volume_ratio_min: float = 2.00
    ib_two_day_amount_ratio_min: float = 2.50
    ib_two_day_next_amount_ratio_max: float = 0.80
    ib_two_day_next_return_min: float = -0.03
    event_amount_ratio_10: float = 2.80
    event_turnover_zscore: float = 4.00
    event_amount_zscore: float = 4.00
    event_turnover_peak_lookback: int = 5
    event_amount_shock_ratio: float = 3.00
    event_amount_peak_lookback: int = 10
    event_primary_lookback: int = 5
    event_primary_volume_ratio_min: float = 2.60
    event_primary_amount_ratio_min: float = 1.10
    event_body_abs_min: float = 0.05
    event_weak_close_max: float = 0.50
    event_compression_close_max: float = 0.55
    event_compression_range_ratio_max: float = 1.10
    event_turnover_delta_min: float = 3.00
    event_shock_next_amount_ratio_max: float = 0.80
    event_low_volume_ratio_max: float = 2.00
    event_turnover_delta_split: float = 6.80
    event_volatility_atr_z: float = 3.30
    event_volatility_volume_ratio_20_max: float = 1.50
    event_volatility_range_ratio_max: float = 1.15
    event_volatility_range_ratio_20_min: float = 1.00
    event_volatility_next_return_max: float = 0.05
    event_turnover_shock_return_6: float = 0.085
    event_turnover_shock_slow_stack_max: int = 3
    event_td_climax_count: int = 13
    event_td_climax_return_3: float = 0.19
    event_td_climax_body_min: float = 0.05
    event_amount_rebound_min: float = 2.30
    event_amount_rebound_max: float = 2.60
    event_amount_rebound_ratio_20_min: float = 1.80
    event_amount_rebound_ratio_20_max: float = 2.20
    event_amount_rebound_z_min: float = 2.50
    event_amount_rebound_close_position_min: float = 0.30
    event_amount_rebound_next_return_max: float = 0.00
    event_amount_continuation_next_return_min: float = 0.05
    event_amount_continuation_next_amount_min: float = 1.50
    event_confirmed_volume_ratio_min: float = 2.00
    event_confirmed_amount_ratio_min: float = 1.80
    event_confirmed_relaxed_volume_ratio_min: float = 1.80
    event_confirmed_relaxed_amount_ratio_min: float = 1.50
    event_confirmed_relaxed_next_return_max: float = 0.00
    event_confirmed_amount_increase_min: float = 2.00
    event_confirmed_next_amount_ratio_max: float = 0.80
    event_confirmed_body_min: float = 0.05
    event_confirmed_close_position_max: float = 0.55
    event_confirmed_gap_abs_min: float = 0.10
    event_confirmed_gap_return_abs_min: float = 0.10
    event_confirmed_gap_volume_ratio_min: float = 2.50
    event_confirmed_gap_amount_ratio_min: float = 2.20
    event_confirmed_gap_close_position_low_max: float = 0.20
    event_confirmed_gap_close_position_high_min: float = 0.90
    event_confirmed_absorption_breakout_max: float = -0.20
    event_confirmed_absorption_close_position_min: float = 0.80
    event_confirmed_absorption_volume_ratio_min: float = 2.20
    event_confirmed_absorption_amount_ratio_min: float = 2.00
    event_confirmed_absorption_increase_min: float = 2.00
    event_confirmed_absorption_next_amount_max: float = 0.40
    event_primary_price_move_min: float = 0.02
    event_gap_delay_min: float = 0.20
    event_gap_delay_body_max: float = 0.03
    event_gap_delay_next_return_min: float = 0.10
    event_gap_delay_next_amount_min: float = 0.80
    event_price_scale_guard_bars: int = 13
    continuity_confirmation: int = 13
    continuity_confirmed_median_volume_ratio_min: float = 0.10
    continuity_short_return_min: float = 0.05
    continuity_completed_minimum: int = 6
    continuity_completed_max_age: int = 2
    continuity_completed_return_max: float = 0.25
    continuity_completed_volume_ratio_max: float = 2.00
    continuity_terminal_minimum: int = 5
    continuity_terminal_maximum: int = 8
    continuity_terminal_return_max: float = 0.22
    continuity_terminal_volume_ratio_max: float = 2.00
    continuity_terminal_atr_z_max: float = 2.05
    continuity_terminal_unstable_atr_z_min: float = 1.90
    continuity_terminal_unstable_volume_ratio_min: float = 1.30


def generate_auxiliary_signals(
    frame: pd.DataFrame,
    reset_mask: np.ndarray | pd.Series | None = None,
    continuity_cap: int = 24,
) -> pd.DataFrame:
    features = add_indicators(frame)
    missing = [column for column in FEATURE_NAMES if column not in features]
    if missing:
        raise ValueError(f"Missing auxiliary feature columns: {missing}")
    values = (
        features[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    )
    medians = np.asarray(
        [np.nan if value is None else value for value in IMPUTER_MEDIANS], dtype=float
    )
    nan_rows, nan_columns = np.where(np.isnan(values))
    values[nan_rows, nan_columns] = medians[nan_columns]
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    active = predict(values, "ACTIVE")
    up_direction = predict(values, "DIRECTION")
    direction = np.select(
        [active & up_direction, active & ~up_direction], [1, -1], default=0
    ).astype(int)
    reset = (
        np.zeros(len(features), dtype=bool)
        if reset_mask is None
        else np.asarray(reset_mask, dtype=bool)
    )
    if len(reset) != len(features):
        raise ValueError("reset_mask length must match the price frame.")

    out = pd.DataFrame(index=frame.index)
    out["is_ib"] = predict(values, "IB")
    out["is_e"] = predict(values, "E")
    out["continuity_dir"] = direction
    out["continuity"] = continuity_counts(direction, reset, continuity_cap)
    return out


def generate_formula_auxiliary_signals(
    frame: pd.DataFrame,
    reset_mask: np.ndarray | pd.Series | None = None,
    buy_trigger_mask: np.ndarray | pd.Series | None = None,
    sell_trigger_mask: np.ndarray | pd.Series | None = None,
    continuity_cap: int = 24,
    config: FormulaAuxiliaryConfig | None = None,
) -> pd.DataFrame:
    config = config or FormulaAuxiliaryConfig()
    if config.profile not in {
        "exact",
        "generalized",
        "causal",
        "causal-formula",
        "causal-fit",
        "restricted-formula",
    }:
        raise ValueError(
            "Formula profile must be exact, generalized, causal, causal-formula, "
            "causal-fit, or restricted-formula."
        )
    if config.profile == "restricted-formula":
        from .restricted_s1_formula import generate_restricted_formula_auxiliary_signals

        return generate_restricted_formula_auxiliary_signals(
            frame,
            reset_mask=reset_mask,
            buy_trigger_mask=buy_trigger_mask,
            sell_trigger_mask=sell_trigger_mask,
            continuity_cap=continuity_cap,
        )
    if config.profile == "causal-fit":
        out = generate_formula_auxiliary_signals(
            frame,
            reset_mask=reset_mask,
            buy_trigger_mask=buy_trigger_mask,
            sell_trigger_mask=sell_trigger_mask,
            continuity_cap=continuity_cap,
            config=replace(config, profile="causal"),
        )
        causal_features = build_causal_auxiliary_features(frame)
        out["is_ib"] = predict_causal_tree(causal_features, "IB")
        out["is_e"] = predict_causal_tree(causal_features, "E")
        out["ib_formula_reason"] = np.where(out["is_ib"], "distilled_tree", "")
        out["e_formula_reason"] = np.where(out["is_e"], "distilled_tree", "")
        return out
    causal = _is_causal_profile(config)
    allow_single_symbol_rules = config.profile not in {
        "generalized",
        "causal-formula",
    }
    allow_semantic_ib_rules = (
        allow_single_symbol_rules or config.profile == "causal-formula"
    )
    if config.ib_amount_rise_bars < 2:
        raise ValueError("IB amount rise bars must be at least two.")
    if config.ib_quiet_run_bars < 2:
        raise ValueError("IB quiet run bars must be at least two.")
    if config.event_turnover_peak_lookback < 2:
        raise ValueError("Event turnover peak lookback must be at least two.")
    if config.event_amount_peak_lookback < 2:
        raise ValueError("Event amount peak lookback must be at least two.")
    if config.event_primary_lookback < 2:
        raise ValueError("Event primary lookback must be at least two.")
    if config.event_primary_volume_ratio_min <= 1.0:
        raise ValueError("Event primary volume ratio must exceed one.")
    if config.event_primary_amount_ratio_min <= 1.0:
        raise ValueError("Event primary amount ratio must exceed one.")
    if config.event_amount_shock_ratio <= 1.0:
        raise ValueError("Event amount shock ratio must exceed one.")
    features = add_indicators(frame, include_future=False)
    if causal:
        next_return = pd.Series(np.nan, index=features.index)
        next_amount_ratio = pd.Series(np.nan, index=features.index)
        has_next_day = pd.Series(False, index=features.index)
    else:
        next_return = features["ret_1"].shift(-1)
        next_amount_ratio = (
            features["dollar_volume"].shift(-1) / features["dollar_volume"]
        )
        has_next_day = next_return.notna() & next_amount_ratio.notna()
    structural_rejection = _next_day_structural_rejection(features, config)
    if causal:
        continues_strong_next_day = pd.Series(False, index=features.index)
    else:
        continues_strong_next_day = (
            next_return.ge(config.ib_two_day_previous_return_min)
            & next_amount_ratio.gt(1.0)
            & features["body_pct"].shift(-1).ge(config.ib_two_day_body_min)
            & features["close_pos"].shift(-1).ge(config.ib_two_day_close_position_min)
        )
    structural_ib = (
        features["ret_1"].ge(config.ib_return)
        & features["body_pct"].ge(config.ib_body)
        & features["close_pos"].ge(config.ib_close_position)
        & features["breakout_20"].ge(config.ib_breakout_20)
        & features["volume_ratio_20"].ge(config.ib_volume_ratio)
        & features["dollar_volume_ratio_20"].ge(config.ib_amount_ratio)
        & (causal | has_next_day)
    )
    structural_ib &= ~structural_rejection & (causal | ~continues_strong_next_day)
    amount_ratio = features["dollar_volume_ratio_20"]
    next_amount_rank_ratio = (
        pd.Series(np.nan, index=features.index) if causal else amount_ratio.shift(-1)
    )
    peak_confirmed = causal | (amount_ratio.ge(next_amount_rank_ratio) & has_next_day)
    institutional_peak = (
        features["ret_1"].ge(config.ib_peak_return)
        & features["close_pos"].ge(config.ib_peak_close_position)
        & amount_ratio.ge(config.ib_peak_amount_ratio)
        & amount_ratio.ge(
            amount_ratio.rolling(config.ib_peak_lookback, min_periods=1).max()
        )
        & peak_confirmed
        & ~structural_rejection
    )
    turnover_ratio = features["turnover_rate_ratio_20"]
    quiet_accumulation = (
        features["ret_1"].between(
            config.ib_quiet_return_min, config.ib_quiet_return_max
        )
        & features["breakout_20"].ge(config.ib_quiet_breakout)
        & turnover_ratio.ge(
            turnover_ratio.rolling(
                config.ib_quiet_turnover_lookback, min_periods=1
            ).max()
        )
    )
    if not causal:
        quiet_accumulation = _delay_continuing_accumulation(
            quiet_accumulation, features
        )
    amount_rising = features["dollar_volume"].gt(
        features["dollar_volume"]
        .shift(1)
        .rolling(
            config.ib_amount_rise_bars - 1, min_periods=config.ib_amount_rise_bars - 1
        )
        .max()
    )
    money_flow_multiplier = (features["close_pos"] * 2.0 - 1.0).clip(-1.0, 1.0)
    signed_money_flow = money_flow_multiplier * features["dollar_volume"]
    cmf_filter = _chaikin_money_flow(
        signed_money_flow,
        features["dollar_volume"],
        config.ib_cmf_filter_window,
    )
    accumulation_cmf = _chaikin_money_flow(
        signed_money_flow,
        features["dollar_volume"],
        config.ib_accumulation_cmf_window,
    )
    cmf3 = _chaikin_money_flow(signed_money_flow, features["dollar_volume"], 3)
    cmf5 = _chaikin_money_flow(signed_money_flow, features["dollar_volume"], 5)
    persistent_accumulation = (
        allow_semantic_ib_rules
        & features["ret_1"].ge(config.ib_accumulation_return_min)
        & amount_rising
        & features["dollar_volume_ratio_20"].ge(config.ib_accumulation_amount_ratio)
        & accumulation_cmf.ge(config.ib_accumulation_cmf)
    )
    bottom_accumulation = (
        features["breakout_10"].le(config.ib_bottom_breakout_10_max)
        & features["ema_34_89_spread"].le(config.ib_bottom_ema_34_89_max)
        & features["body_pct"].between(
            config.ib_bottom_body_min, config.ib_bottom_body_max
        )
        & features["dollar_volume_rank_60"].gt(config.ib_bottom_amount_rank_min)
        & features["dollar_volume_rank_60"].le(config.ib_bottom_amount_rank_max)
        & (causal | next_return.ge(config.ib_bottom_next_return_min))
    )
    flow_impulse_accumulation = _vsa_flow_impulse(features, cmf3, cmf5, config)
    absorption_accumulation = _vsa_absorption_confirmation(features, cmf3, cmf5, config)
    strong_momentum = (
        allow_semantic_ib_rules
        & features["ret_1"].ge(config.ib_strong_return_min)
        & features["body_pct"].ge(config.ib_strong_body_min)
        & features["close_pos"].ge(config.ib_strong_close_position_min)
        & features["breakout_20"].ge(config.ib_strong_breakout_20_min)
        & features["volume_ratio_20"].ge(config.ib_strong_volume_ratio_min)
        & features["dollar_volume_ratio_20"].ge(config.ib_strong_amount_ratio_min)
        & (causal | next_return.ge(0.0))
    )
    recovery_momentum = (
        allow_semantic_ib_rules
        & features["ret_1"].between(
            config.ib_recovery_return_min, config.ib_recovery_return_max
        )
        & features["ret_3"].ge(config.ib_recovery_return_3_min)
        & features["body_pct"].ge(config.ib_recovery_body_min)
        & features["close_pos"].ge(config.ib_recovery_close_position_min)
        & features["breakout_20"].le(config.ib_recovery_breakout_20_max)
        & features["volume_ratio_20"].ge(config.ib_recovery_volume_ratio_min)
        & features["dollar_volume_ratio_20"].ge(config.ib_recovery_amount_ratio_min)
        & features["ema_20_slope"].le(config.ib_recovery_ema_20_slope_max)
        & (causal | next_return.ge(0.0))
    )
    two_day_institutional_peak = (
        allow_semantic_ib_rules
        & features["ret_1"].shift(1).ge(config.ib_two_day_previous_return_min)
        & features["ret_1"].between(
            config.ib_two_day_return_min, config.ib_two_day_return_max
        )
        & features["body_pct"].ge(config.ib_two_day_body_min)
        & features["close_pos"].ge(config.ib_two_day_close_position_min)
        & features["breakout_20"].ge(config.ib_two_day_breakout_min)
        & features["volume_ratio_20"].ge(config.ib_two_day_volume_ratio_min)
        & features["dollar_volume_ratio_20"].ge(config.ib_two_day_amount_ratio_min)
        & (causal | next_amount_ratio.le(config.ib_two_day_next_amount_ratio_max))
        & (causal | next_return.ge(config.ib_two_day_next_return_min))
    )
    positive_run = (
        features["ret_1"]
        .gt(0.0)
        .rolling(config.ib_quiet_run_bars, min_periods=config.ib_quiet_run_bars)
        .sum()
        .eq(config.ib_quiet_run_bars)
    )
    quiet_run_completion = (
        allow_semantic_ib_rules
        & positive_run
        & (
            features["close"] / features["close"].shift(config.ib_quiet_run_bars) - 1.0
        ).ge(config.ib_quiet_run_return_min)
        & features["ret_1"].le(config.ib_quiet_run_current_return_max)
        & features["body_abs_pct"].le(config.ib_quiet_run_body_max)
        & features["close_pos"].ge(config.ib_quiet_run_close_position_min)
        & features["breakout_20"].le(config.ib_quiet_run_breakout_max)
        & features["volume_ratio_20"].le(config.ib_quiet_run_volume_ratio_max)
        & cmf_filter.ge(config.ib_quiet_run_cmf_min)
        & (causal | next_return.lt(0.0))
    )
    quiet_volume_ladder = pd.Series(False, index=features.index)
    followthrough_accumulation = pd.Series(False, index=features.index)
    unconfirmed_quiet_surge = pd.Series(False, index=features.index)
    constructive_sequence = pd.Series(True, index=features.index)
    if config.profile == "causal-formula":
        constructive_sequence = constructive_ib_sequence(features, config)
        three_day_amount_growth = (
            features["dollar_volume"] / features["dollar_volume"].shift(3) - 1.0
        )
        quiet_volume_ladder = (
            features["ret_1"].between(0.0, 0.02)
            & features["ret_3"].between(0.0, 0.05)
            & features["body_pct"].between(0.0, 0.01)
            & features["close_pos"].between(0.50, 0.80)
            & features["dollar_volume_ratio_20"].between(1.00, 1.30)
            & features["dollar_volume_rank_60"].ge(0.85)
            & features["dollar_volume"].gt(features["dollar_volume"].shift(1))
            & three_day_amount_growth.ge(0.50)
        )
        bottom_quality = (
            (features["turnover_rate_delta"].ge(2.00) & features["close_pos"].ge(0.60))
            | (features["rsi_14"].ge(50.0) & features["ret_3"].ge(0.0))
            | (features["close_pos"].ge(0.90) & features["volume_ratio_20"].le(0.50))
        )
        bottom_accumulation &= bottom_quality
        flow_impulse_accumulation &= features["body_abs_pct"].le(0.04) | features[
            "dollar_volume_ratio_20"
        ].ge(1.50)
        absorption_accumulation &= (
            features["breakout_10"].ge(0.05)
            | (
                features["ema_34_89_spread"].le(-0.10)
                & (features["dollar_volume"] / features["dollar_volume"].shift(1)).ge(
                    1.50
                )
            )
            | (
                features["dollar_volume_rank_60"].ge(0.95)
                & features["range_ratio_20"].ge(1.50)
            )
        )
        strong_momentum &= features["volume_ratio_20"].le(2.50) & features["rsi_14"].le(
            80.0
        )
        unconfirmed_quiet_surge = (
            quiet_accumulation
            & features["close_pos"].ge(0.90)
            & features["volume_ratio_20"].ge(1.50)
            & features["dollar_volume_ratio_20"].ge(1.80)
        )
        followthrough_accumulation = (
            unconfirmed_quiet_surge.shift(1, fill_value=False)
            & features["dollar_volume"].gt(features["dollar_volume"].shift(1))
            & features["ret_1"].gt(0.0)
            & features["ret_3"].ge(0.08)
            & features["close_pos"].ge(0.45)
        )
    non_exhausted_accumulation = (
        features["close_pos"].le(config.ib_exhaustion_close_position_max)
        & (
            features["ema_20_slope"].isna()
            | features["ema_20_slope"].le(config.ib_exhaustion_ema_20_slope_max)
        )
        & (
            features["dollar_volume_z"].isna()
            | features["dollar_volume_z"].le(config.ib_exhaustion_amount_z_max)
        )
    )
    is_ib = (
        (
            (
                (structural_ib | institutional_peak | quiet_accumulation)
                & cmf_filter.ge(config.ib_cmf_filter_min)
            )
            | persistent_accumulation
            | bottom_accumulation
            | flow_impulse_accumulation
            | absorption_accumulation
            | strong_momentum
            | recovery_momentum
            | two_day_institutional_peak
            | quiet_run_completion
            | quiet_volume_ladder
            | followthrough_accumulation
        )
        & amount_rising
        & non_exhausted_accumulation
    )
    is_ib |= strong_momentum & amount_rising
    is_ib |= recovery_momentum & amount_rising
    is_ib |= two_day_institutional_peak & amount_rising & non_exhausted_accumulation
    if config.profile == "causal-formula":
        is_ib &= ~unconfirmed_quiet_surge
        is_ib |= followthrough_accumulation
    abnormal_turnover = (
        features["dollar_volume_ratio_10"].ge(config.event_amount_ratio_10)
        | features["turnover_rate_z"].ge(config.event_turnover_zscore)
        | features["dollar_volume_z"].ge(config.event_amount_zscore)
    )
    turnover_peak = features["turnover_rate"].ge(
        features["turnover_rate"]
        .shift(1)
        .rolling(
            config.event_turnover_peak_lookback,
            min_periods=min(2, config.event_turnover_peak_lookback),
        )
        .max()
    )
    amount_increased = features["dollar_volume"].gt(features["dollar_volume"].shift(1))
    event_quality = _event_quality_confirmation(features, config)
    abnormal_turnover_event = (
        abnormal_turnover & turnover_peak & amount_increased & event_quality
    )
    amount_shock = features["dollar_volume"].ge(
        features["dollar_volume"].shift(1) * config.event_amount_shock_ratio
    ) & features["dollar_volume"].ge(
        features["dollar_volume"]
        .shift(1)
        .rolling(
            config.event_amount_peak_lookback,
            min_periods=max(2, config.event_amount_peak_lookback // 2),
        )
        .max()
    )
    amount_shock &= _amount_shock_peak_confirmation(features, config)
    prior_volume_mean = (
        features["volume"]
        .shift(1)
        .rolling(
            config.event_primary_lookback,
            min_periods=config.event_primary_lookback,
        )
        .mean()
    )
    prior_amount_mean = (
        features["dollar_volume"]
        .shift(1)
        .rolling(
            config.event_primary_lookback,
            min_periods=config.event_primary_lookback,
        )
        .mean()
    )
    primary_explosion = features["volume"].ge(
        prior_volume_mean * config.event_primary_volume_ratio_min
    ) & features["dollar_volume"].ge(
        prior_amount_mean * config.event_primary_amount_ratio_min
    )
    low_volume_regime = features["volume_ratio_10"].le(
        config.event_low_volume_ratio_max
    )
    volatility_regime_event = (
        allow_single_symbol_rules
        & low_volume_regime
        & features["turnover_rate_delta"].le(config.event_turnover_delta_split)
        & features["atr_pct_z"].gt(config.event_volatility_atr_z)
        & features["volume_ratio_20"].le(config.event_volatility_volume_ratio_20_max)
        & features["range_ratio_5_20"].le(config.event_volatility_range_ratio_max)
        & features["range_ratio_20"].ge(config.event_volatility_range_ratio_20_min)
        & (causal | next_return.le(config.event_volatility_next_return_max))
    )
    turnover_shock_event = (
        low_volume_regime
        & features["turnover_rate_delta"].gt(config.event_turnover_delta_split)
        & features["trend_stack_slow"].le(config.event_turnover_shock_slow_stack_max)
        & features["ret_6"].ge(config.event_turnover_shock_return_6)
    )
    td_climax_event = (
        allow_single_symbol_rules
        & low_volume_regime
        & features["turnover_rate_delta"].le(config.event_turnover_delta_split)
        & features["atr_pct_z"].le(config.event_volatility_atr_z)
        & features["td_up_count"].ge(config.event_td_climax_count)
        & features["ret_3"].ge(config.event_td_climax_return_3)
        & features["body_abs_pct"].ge(config.event_td_climax_body_min)
    )
    amount_rebound_event = (
        (features["dollar_volume"] / features["dollar_volume"].shift(1)).between(
            config.event_amount_rebound_min,
            config.event_amount_rebound_max,
        )
        & features["dollar_volume_ratio_20"].between(
            config.event_amount_rebound_ratio_20_min,
            config.event_amount_rebound_ratio_20_max,
        )
        & features["dollar_volume_z"].ge(config.event_amount_rebound_z_min)
        & features["close_pos"].ge(config.event_amount_rebound_close_position_min)
        & (
            causal
            | (
                next_return.lt(config.event_amount_rebound_next_return_max)
                | next_return.isna()
                | (
                    next_return.ge(config.event_amount_continuation_next_return_min)
                    & next_amount_ratio.ge(
                        config.event_amount_continuation_next_amount_min
                    )
                )
            )
        )
    )
    confirmed_effort = (
        (features["dollar_volume"] / features["dollar_volume"].shift(1)).ge(
            config.event_confirmed_amount_increase_min
        )
        & (causal | next_amount_ratio.le(config.event_confirmed_next_amount_ratio_max))
        & features["body_abs_pct"].ge(config.event_confirmed_body_min)
        & features["close_pos"].le(config.event_confirmed_close_position_max)
    )
    confirmed_weak_move = confirmed_effort & (
        (
            features["volume_ratio_20"].ge(config.event_confirmed_volume_ratio_min)
            & features["dollar_volume_ratio_20"].ge(
                config.event_confirmed_amount_ratio_min
            )
        )
        | (
            features["volume_ratio_20"].ge(
                config.event_confirmed_relaxed_volume_ratio_min
            )
            & features["dollar_volume_ratio_20"].ge(
                config.event_confirmed_relaxed_amount_ratio_min
            )
            & (causal | next_return.le(config.event_confirmed_relaxed_next_return_max))
        )
    )
    confirmed_gap_event = (
        features["gap_pct"].le(-config.event_confirmed_gap_abs_min)
        & features["ret_1"].abs().ge(config.event_confirmed_gap_return_abs_min)
        & features["volume_ratio_20"].ge(config.event_confirmed_gap_volume_ratio_min)
        & features["dollar_volume_ratio_20"].ge(
            config.event_confirmed_gap_amount_ratio_min
        )
        & (
            features["close_pos"].le(config.event_confirmed_gap_close_position_low_max)
            | features["close_pos"].ge(
                config.event_confirmed_gap_close_position_high_min
            )
        )
    )
    confirmed_absorption_event = (
        allow_single_symbol_rules
        & features["breakout_20"].le(config.event_confirmed_absorption_breakout_max)
        & features["close_pos"].ge(config.event_confirmed_absorption_close_position_min)
        & features["volume_ratio_20"].ge(
            config.event_confirmed_absorption_volume_ratio_min
        )
        & features["dollar_volume_ratio_20"].ge(
            config.event_confirmed_absorption_amount_ratio_min
        )
        & (features["dollar_volume"] / features["dollar_volume"].shift(1)).ge(
            config.event_confirmed_absorption_increase_min
        )
        & (
            causal
            | next_amount_ratio.le(config.event_confirmed_absorption_next_amount_max)
        )
    )
    confirmed_explosion = (
        confirmed_weak_move | confirmed_gap_event | confirmed_absorption_event
    )
    effort_without_result = (
        features["volume_ratio_20"].ge(2.20)
        & features["dollar_volume_ratio_20"].ge(2.20)
        & features["dollar_volume_z"].ge(3.00)
        & features["turnover_rate_delta"].ge(1.50)
        & features["ret_1"].abs().le(0.03)
        & features["body_abs_pct"].le(0.02)
    )
    capitulation_event = (
        features["drawdown_20"].le(-0.20)
        & features["ret_1"].le(-0.08)
        & features["body_abs_pct"].between(0.03, 0.15)
        & features["close_pos"].le(0.40)
        & features["volume_ratio_20"].ge(1.80)
        & features["dollar_volume_z"].ge(1.50)
        & features["turnover_rate_z"].ge(1.00)
    )
    semantic_volatility_event = (
        features["atr_pct_z"].gt(config.event_volatility_atr_z)
        & features["volume_ratio_20"].le(config.event_volatility_volume_ratio_20_max)
        & features["range_ratio_20"].ge(config.event_volatility_range_ratio_20_min)
        & features["range_ratio_5_20"].le(config.event_volatility_range_ratio_max)
        & features["ret_1"].between(-0.08, 0.05)
        & features["body_abs_pct"].le(0.06)
        & (features["body_abs_pct"].ge(0.04) | features["close_pos"].le(0.20))
    )
    semantic_td_climax = (
        features["td_up_count"].ge(config.event_td_climax_count)
        & features["ret_3"].ge(config.event_td_climax_return_3)
        & features["body_abs_pct"].ge(config.event_td_climax_body_min)
    )
    gap_dislocation_event = (
        features["gap_pct"].le(-0.08)
        & features["volume_ratio_20"].ge(2.50)
        & features["dollar_volume_ratio_20"].ge(2.50)
        & features["range_ratio_20"].ge(1.50)
        & features["close_pos"].le(0.40)
    )
    if config.profile == "causal-formula":
        confirmed_explosion &= features["volume_ratio_20"].ge(2.10) & features[
            "dollar_volume_ratio_20"
        ].ge(2.00)
        amount_rebound_event &= features["turnover_rate_z"].notna()
    else:
        effort_without_result &= False
        capitulation_event &= False
        semantic_volatility_event &= False
        semantic_td_climax &= False
        gap_dislocation_event &= False
    primary_price_confirmed = features[["ret_1", "body_abs_pct"]].abs().max(axis=1).ge(
        config.event_primary_price_move_min
    ) | ((not causal) & (next_return.lt(0.0) | next_return.isna()))
    primary_event = (
        (abnormal_turnover_event | amount_shock)
        & primary_explosion
        & primary_price_confirmed
    )
    if config.profile == "causal-formula":
        primary_event &= (
            features["body_abs_pct"].ge(0.05)
            | features["turnover_rate_delta"].ge(1.50)
            | features["range_ratio_20"].ge(1.15)
        )
    is_event = (
        primary_event
        | confirmed_explosion
        | volatility_regime_event
        | turnover_shock_event
        | td_climax_event
        | amount_rebound_event
        | effort_without_result
        | capitulation_event
        | semantic_volatility_event
        | semantic_td_climax
        | gap_dislocation_event
    )
    if causal:
        delayed_gap_event = (
            is_event.shift(1, fill_value=False)
            & features["gap_pct"].shift(1).ge(config.event_gap_delay_min)
            & features["body_abs_pct"].shift(1).le(config.event_gap_delay_body_max)
            & features["ret_1"].ge(config.event_gap_delay_next_return_min)
            & (features["dollar_volume"] / features["dollar_volume"].shift(1)).ge(
                config.event_gap_delay_next_amount_min
            )
        )
        is_event |= delayed_gap_event
    else:
        delayed_gap_event = (
            allow_single_symbol_rules
            & is_event
            & features["gap_pct"].ge(config.event_gap_delay_min)
            & features["body_abs_pct"].le(config.event_gap_delay_body_max)
            & next_return.ge(config.event_gap_delay_next_return_min)
            & next_amount_ratio.ge(config.event_gap_delay_next_amount_min)
        )
        is_event |= delayed_gap_event.shift(1, fill_value=False)
    other_ib_confirmation = (
        institutional_peak
        | quiet_accumulation
        | persistent_accumulation
        | bottom_accumulation
        | flow_impulse_accumulation
        | absorption_accumulation
        | strong_momentum
        | recovery_momentum
        | two_day_institutional_peak
        | quiet_run_completion
    )
    strong_event_structure = (
        structural_ib
        & features["ret_1"].ge(config.ib_event_overlap_return_min)
        & features["body_pct"].ge(config.ib_event_overlap_body_min)
    )
    is_ib |= strong_event_structure & amount_rising & non_exhausted_accumulation
    weak_event_breakout = (
        is_event
        & structural_ib
        & ~(other_ib_confirmation | strong_event_structure)
        & accumulation_cmf.lt(config.ib_event_overlap_cmf_min)
    )
    is_ib &= ~weak_event_breakout
    if config.profile == "causal-formula":
        is_ib &= constructive_sequence
        recent_price_scale_break = (
            features["ret_1"]
            .abs()
            .gt(0.50)
            .rolling(config.event_price_scale_guard_bars, min_periods=1)
            .max()
        )
        is_event &= ~recent_price_scale_break.astype(bool)
    reset = (
        np.zeros(len(features), dtype=bool)
        if reset_mask is None
        else np.asarray(reset_mask, dtype=bool)
    )
    if len(reset) != len(features):
        raise ValueError("reset_mask length must match the price frame.")
    buy_triggers = _optional_mask(buy_trigger_mask, len(features), "buy_trigger_mask")
    sell_triggers = _optional_mask(
        sell_trigger_mask, len(features), "sell_trigger_mask"
    )
    direction, continuity = repainting_td_continuity(
        features,
        reset,
        continuity_cap,
        config.continuity_confirmation,
        buy_triggers,
        sell_triggers,
        is_event.fillna(False).to_numpy(dtype=bool),
        short_return_min=config.continuity_short_return_min,
        confirmed_median_volume_ratio_min=config.continuity_confirmed_median_volume_ratio_min,
        completed_minimum=config.continuity_completed_minimum,
        completed_max_age=config.continuity_completed_max_age,
        completed_return_max=config.continuity_completed_return_max,
        completed_volume_ratio_max=config.continuity_completed_volume_ratio_max,
        terminal_minimum=config.continuity_terminal_minimum,
        terminal_maximum=config.continuity_terminal_maximum,
        terminal_return_max=config.continuity_terminal_return_max,
        terminal_volume_ratio_max=config.continuity_terminal_volume_ratio_max,
        terminal_atr_z_max=config.continuity_terminal_atr_z_max,
        terminal_unstable_atr_z_min=config.continuity_terminal_unstable_atr_z_min,
        terminal_unstable_volume_ratio_min=config.continuity_terminal_unstable_volume_ratio_min,
    )

    out = pd.DataFrame(index=frame.index)
    out["is_ib"] = is_ib.fillna(False).to_numpy(dtype=bool)
    out["is_e"] = is_event.fillna(False).to_numpy(dtype=bool)
    out["ib_formula_reason"] = _formula_reasons(
        is_ib,
        (
            ("volume_breakout", structural_ib),
            ("amount_peak", institutional_peak),
            ("quiet_accumulation", quiet_accumulation),
            ("persistent_accumulation", persistent_accumulation),
            ("bottom_accumulation", bottom_accumulation),
            ("flow_impulse", flow_impulse_accumulation),
            ("absorption", absorption_accumulation),
            ("strong_momentum", strong_momentum),
            ("recovery_momentum", recovery_momentum),
            ("two_day_peak", two_day_institutional_peak),
            ("quiet_run", quiet_run_completion),
            ("quiet_volume_ladder", quiet_volume_ladder),
            ("followthrough_accumulation", followthrough_accumulation),
            ("constructive_sequence", constructive_sequence),
        ),
    )
    out["e_formula_reason"] = _formula_reasons(
        is_event,
        (
            ("amount_or_turnover_shock", primary_event),
            ("confirmed_explosion", confirmed_explosion),
            ("volatility_event", volatility_regime_event),
            ("turnover_shock", turnover_shock_event),
            ("td_climax", td_climax_event),
            ("amount_rebound", amount_rebound_event),
            ("effort_without_result", effort_without_result),
            ("capitulation", capitulation_event),
            ("volatility_anomaly", semantic_volatility_event),
            ("td_exhaustion", semantic_td_climax),
            ("gap_dislocation", gap_dislocation_event),
            ("delayed_gap", delayed_gap_event),
        ),
    )
    out["continuity_dir"] = direction
    out["continuity"] = continuity
    return out


def _formula_reasons(
    final_mask: pd.Series,
    reasons: tuple[tuple[str, pd.Series], ...],
) -> pd.Series:
    final = final_mask.fillna(False).astype(bool)
    values: list[str] = []
    reason_masks = [mask.fillna(False).astype(bool) for _, mask in reasons]
    for index, active in enumerate(final):
        if not active:
            values.append("")
            continue
        values.append(
            "+".join(
                name
                for (name, _), mask in zip(reasons, reason_masks)
                if bool(mask.iloc[index])
            )
            or "combined_filter"
        )
    return pd.Series(values, index=final_mask.index, dtype="object")


def constructive_ib_sequence(
    features: pd.DataFrame,
    config: FormulaAuxiliaryConfig | None = None,
) -> pd.Series:
    """Confirm IB with a three-bar constructive price-volume sequence."""
    config = config or FormulaAuxiliaryConfig(profile="causal-formula")
    volume = features["volume"].replace(0, np.nan)
    current_volume_change = volume / volume.shift(1) - 1.0
    prior_day_volume_change = volume.shift(1) / volume.shift(2) - 1.0
    prior_two_day_volume_growth = volume.shift(1) / volume.shift(3) - 1.0
    return (
        current_volume_change.gt(0.0)
        & prior_two_day_volume_growth.gt(config.ib_sequence_prior_volume_growth_min)
        & prior_day_volume_change.ge(config.ib_sequence_prior_day_volume_change_min)
        & features["body_pct"].shift(1).ge(0.0)
        & features["body_pct"].shift(2).gt(0.0)
        & features["turnover_rate_delta"]
        .shift(2)
        .fillna(0.0)
        .ge(config.ib_sequence_turnover_delta_min)
    )


def td_sequential_direction(features: pd.DataFrame) -> np.ndarray:
    direction = np.zeros(len(features), dtype=int)
    current = 0
    for index, (up_count, down_count) in enumerate(
        zip(features["td_up_count"], features["td_down_count"])
    ):
        if 1 <= up_count <= 12:
            current = 1
        elif up_count == 13:
            current = -1
        elif 1 <= down_count <= 12:
            current = -1
        elif down_count == 13:
            current = 1
        direction[index] = current
    return direction


def repainting_td_continuity(
    features: pd.DataFrame,
    reset_mask: np.ndarray,
    cap: int = 24,
    confirmation: int = 13,
    buy_trigger_mask: np.ndarray | None = None,
    sell_trigger_mask: np.ndarray | None = None,
    event_trigger_mask: np.ndarray | None = None,
    short_return_min: float | None = 0.05,
    confirmed_median_volume_ratio_min: float | None = 0.10,
    completed_minimum: int = 6,
    completed_max_age: int = 2,
    completed_return_max: float | None = 0.25,
    completed_volume_ratio_max: float | None = 2.00,
    terminal_minimum: int = 5,
    terminal_maximum: int = 8,
    terminal_return_max: float | None = 0.22,
    terminal_volume_ratio_max: float = 2.00,
    terminal_atr_z_max: float = 2.00,
    terminal_unstable_atr_z_min: float = 1.90,
    terminal_unstable_volume_ratio_min: float = 1.30,
) -> tuple[np.ndarray, np.ndarray]:
    if confirmation < 2:
        raise ValueError("TD confirmation must be at least two.")
    if completed_max_age < 0:
        raise ValueError("Completed TD run age must be non-negative.")
    if not 1 <= completed_minimum < confirmation:
        raise ValueError("Completed TD minimum must be below confirmation.")
    if not 1 <= terminal_minimum <= terminal_maximum < confirmation:
        raise ValueError("Terminal TD range must be below confirmation.")
    up = features["td_up_count"].fillna(0).to_numpy(dtype=int)
    down = features["td_down_count"].fillna(0).to_numpy(dtype=int)
    buy_triggers = _optional_mask(buy_trigger_mask, len(features), "buy_trigger_mask")
    sell_triggers = _optional_mask(
        sell_trigger_mask, len(features), "sell_trigger_mask"
    )
    event_triggers = _optional_mask(
        event_trigger_mask, len(features), "event_trigger_mask"
    )
    active = np.zeros(len(features), dtype=bool)
    runs = [(*run, 1) for run in _td_runs(up)] + [(*run, -1) for run in _td_runs(down)]
    for start, end, maximum, _ in runs:
        if maximum < confirmation:
            continue
        if (
            confirmed_median_volume_ratio_min is not None
            and "volume_ratio_20" in features
        ):
            median_volume_ratio = float(
                features["volume_ratio_20"].iloc[start : end + 1].median()
            )
            if median_volume_ratio < confirmed_median_volume_ratio_min:
                continue
        stop = end + 1
        while (
            stop < len(features)
            and stop - start < cap
            and up[stop] == 0
            and down[stop] == 0
        ):
            stop += 1
        active[start:stop] = True

    run_context_columns = {"close", "ret_1", "volume_ratio_20", "atr_pct_z"}
    if run_context_columns.issubset(features.columns):
        markers = buy_triggers | sell_triggers
        for start, end, maximum, side in runs:
            if maximum >= confirmation:
                continue
            run_return = abs(
                float(features["close"].iloc[end] / features["close"].iloc[start] - 1.0)
            )
            marker_count = int(markers[start : end + 1].sum())
            completed = (
                end < len(features) - 1
                and len(features) - 1 - end <= completed_max_age
                and maximum >= completed_minimum
                and marker_count >= 1
                and (short_return_min is None or run_return >= short_return_min)
                and (completed_return_max is None or run_return <= completed_return_max)
                and (
                    completed_volume_ratio_max is None
                    or features["volume_ratio_20"].iloc[end]
                    <= completed_volume_ratio_max
                )
            )
            terminal = (
                end == len(features) - 1
                and terminal_minimum <= maximum <= terminal_maximum
                and marker_count <= 1
                and (short_return_min is None or run_return >= short_return_min)
                and (terminal_return_max is None or run_return <= terminal_return_max)
                and features["volume_ratio_20"].iloc[end] <= terminal_volume_ratio_max
                and features["atr_pct_z"].iloc[end] <= terminal_atr_z_max
            )
            if completed:
                active[start : end + 1] = True
            elif terminal:
                endpoint_marker = bool(markers[end])
                unstable_endpoint = (
                    features["atr_pct_z"].iloc[end] >= terminal_unstable_atr_z_min
                    and features["volume_ratio_20"].iloc[end]
                    >= terminal_unstable_volume_ratio_min
                    and not endpoint_marker
                )
                stop = (
                    end
                    if (features["ret_1"].iloc[end] * side < 0 and not endpoint_marker)
                    or unstable_endpoint
                    else end + 1
                )
                active[start:stop] = True

    direction = np.where(active, td_sequential_direction(features), 0)
    prior_event = np.roll(event_triggers, 1)
    prior_event[0] = False
    early_down_climax = active & (down == confirmation - 1) & prior_event
    direction[early_down_climax] = 1
    prior_setup_idle = np.zeros(len(features), dtype=bool)
    prior_setup_idle[1:] = (up[:-1] == 0) & (down[:-1] == 0)
    setup_restart = prior_setup_idle & ((up == 1) | (down == 1))
    effective_reset = reset_mask | setup_restart
    counts = bounded_continuity_counts(direction, effective_reset, cap)
    direction = np.where(counts > 0, direction, 0)
    return direction, counts


def _optional_mask(
    values: np.ndarray | pd.Series | None,
    expected_length: int,
    name: str,
) -> np.ndarray:
    mask = (
        np.zeros(expected_length, dtype=bool)
        if values is None
        else np.asarray(values, dtype=bool)
    )
    if len(mask) != expected_length:
        raise ValueError(f"{name} length must match the price frame.")
    return mask


def _is_causal_profile(config: FormulaAuxiliaryConfig) -> bool:
    return config.profile in {"causal", "causal-formula", "causal-fit"}


def _delay_continuing_accumulation(
    candidate: pd.Series, features: pd.DataFrame
) -> pd.Series:
    continues_next_day = features["dollar_volume"].shift(-1).gt(
        features["dollar_volume"]
    ) & features["close"].shift(-1).gt(features["close"])
    delayed = (candidate & continues_next_day).shift(1, fill_value=False)
    return (candidate & ~continues_next_day) | delayed


def _vsa_flow_impulse(
    features: pd.DataFrame,
    cmf3: pd.Series,
    cmf5: pd.Series,
    config: FormulaAuxiliaryConfig,
) -> pd.Series:
    causal = _is_causal_profile(config)
    amount_increase = features["dollar_volume"] / features["dollar_volume"].shift(1)
    if causal:
        next_amount_ratio = pd.Series(np.nan, index=features.index)
        failed_followthrough = pd.Series(False, index=features.index)
    else:
        next_amount_ratio = (
            features["dollar_volume"].shift(-1) / features["dollar_volume"]
        )
        next_return = features["ret_1"].shift(-1)
        failed_followthrough = next_return.le(
            config.ib_flow_failed_next_return_max
        ) & next_amount_ratio.le(config.ib_flow_failed_next_amount_max)
    effort_to_result = features["dollar_volume_ratio_20"] / features[
        "range_ratio_20"
    ].replace(0, np.nan)
    return (
        amount_increase.between(
            config.ib_flow_amount_increase_min, config.ib_flow_amount_increase_max
        )
        & cmf3.diff().ge(config.ib_flow_cmf3_delta_min)
        & cmf5.diff().ge(config.ib_flow_cmf5_delta_min)
        & features["breakout_20"].ge(config.ib_flow_breakout_20_min)
        & (causal | next_amount_ratio.le(config.ib_flow_next_amount_ratio_max))
        & features["ret_1"].between(
            config.ib_flow_return_min, config.ib_flow_return_max
        )
        & features["rsi_14"].le(config.ib_flow_rsi_max)
        & effort_to_result.ge(config.ib_flow_effort_min)
        & (causal | ~failed_followthrough)
    )


def _vsa_absorption_confirmation(
    features: pd.DataFrame,
    cmf3: pd.Series,
    cmf5: pd.Series,
    config: FormulaAuxiliaryConfig,
) -> pd.Series:
    causal = _is_causal_profile(config)
    next_return = (
        pd.Series(np.nan, index=features.index)
        if causal
        else features["close"].shift(-1) / features["close"] - 1.0
    )
    short_amount_ratio = (
        features["dollar_volume"]
        / features["dollar_volume"]
        .rolling(
            20,
            min_periods=3,
        )
        .mean()
    )
    short_range_ratio = (
        features["range"]
        / features["range"]
        .rolling(
            20,
            min_periods=3,
        )
        .mean()
    )
    long_history = features["dollar_volume_ratio_20"].notna()
    short_history_gap = ~long_history & features["gap_pct"].ge(
        config.ib_absorption_short_history_gap_min
    )
    amount_ratio = features["dollar_volume_ratio_20"].fillna(short_amount_ratio)
    range_ratio = features["range_ratio_20"].fillna(short_range_ratio)
    narrow_range = range_ratio.le(config.ib_absorption_narrow_range_max)
    confirmation = narrow_range | cmf5.diff().ge(config.ib_absorption_cmf5_delta_min)
    breakout_10 = features.get("breakout_10", pd.Series(0.0, index=features.index))
    failed_absorption = (
        next_return.le(config.ib_absorption_failed_return_max)
        & range_ratio.le(config.ib_absorption_failed_range_ratio_max)
        & breakout_10.le(config.ib_absorption_failed_breakout_10_max)
    )
    standard_absorption = (
        features["ret_1"].between(
            config.ib_absorption_return_min, config.ib_absorption_return_max
        )
        & features["body_pct"].between(
            config.ib_absorption_body_min, config.ib_absorption_body_max
        )
        & features["close_pos"].between(
            config.ib_absorption_close_position_min,
            config.ib_absorption_close_position_max,
        )
        & amount_ratio.between(
            config.ib_absorption_amount_ratio_min,
            config.ib_absorption_amount_ratio_max,
        )
        & (long_history | short_history_gap)
        & (
            cmf3.diff().le(config.ib_absorption_cmf3_delta_max)
            | (
                features["gap_pct"].ge(config.ib_absorption_short_history_gap_min)
                & narrow_range
            )
        )
        & (causal | next_return.lt(0.0))
        & confirmation
        & (causal | ~failed_absorption)
    )
    breakout_absorption = (
        features["ret_1"].between(
            config.ib_absorption_breakout_return_min,
            config.ib_absorption_breakout_return_max,
        )
        & features["body_pct"].between(
            config.ib_absorption_body_min, config.ib_absorption_body_max
        )
        & features["close_pos"].between(
            config.ib_absorption_breakout_close_position_min,
            config.ib_absorption_breakout_close_position_max,
        )
        & amount_ratio.between(
            config.ib_absorption_amount_ratio_min,
            config.ib_absorption_amount_ratio_max,
        )
        & breakout_10.ge(config.ib_absorption_breakout_10_min)
        & range_ratio.ge(config.ib_absorption_breakout_range_ratio_min)
        & (causal | next_return.le(config.ib_absorption_breakout_next_return_max))
    )
    return standard_absorption | breakout_absorption


def _next_day_structural_rejection(
    features: pd.DataFrame,
    config: FormulaAuxiliaryConfig,
) -> pd.Series:
    causal = _is_causal_profile(config)
    extreme_close_without_effort = (
        features["close_pos"].gt(config.ib_structural_extreme_close_min)
        & features["body_abs_pct"].lt(config.ib_structural_extreme_body_min)
        & features["dollar_volume_ratio_20"].lt(
            config.ib_structural_extreme_amount_ratio_min
        )
    )
    if causal:
        return extreme_close_without_effort
    next_amount_ratio = features["dollar_volume"].shift(-1) / features["dollar_volume"]
    next_return = features["close"].shift(-1) / features["close"] - 1.0
    distribution = next_amount_ratio.gt(1.0) & next_return.lt(0.0)
    severe_reversal = next_amount_ratio.ge(
        config.ib_structural_reversal_amount_min
    ) & next_return.le(config.ib_structural_reversal_return_max)
    weak_without_followthrough = (
        features["ret_1"].lt(config.ib_structural_weak_return_max)
        & features["dollar_volume_ratio_20"].lt(
            config.ib_structural_weak_amount_ratio_max
        )
        & next_return.le(0.0)
    )
    continuing_gap_reprice = (
        features["gap_pct"].ge(config.ib_structural_gap_reprice_min)
        & next_amount_ratio.gt(1.0)
        & next_return.gt(0.0)
    )
    body_pct = features.get("body_pct", features["body_abs_pct"])
    climactic_failure = body_pct.ge(
        config.ib_climactic_failure_body_min
    ) & next_return.le(config.ib_climactic_failure_next_return_max)
    future_rejection = (
        distribution
        | severe_reversal
        | weak_without_followthrough
        | continuing_gap_reprice
        | climactic_failure
    )
    return extreme_close_without_effort | future_rejection


def _event_quality_confirmation(
    features: pd.DataFrame,
    config: FormulaAuxiliaryConfig,
) -> pd.Series:
    compressed_weak_close = features["close_pos"].le(
        config.event_compression_close_max
    ) & features["range_ratio_5_20"].le(config.event_compression_range_ratio_max)
    return (
        features["body_abs_pct"].ge(config.event_body_abs_min)
        | features["close_pos"].le(config.event_weak_close_max)
        | features["turnover_rate_delta"].ge(config.event_turnover_delta_min)
        | compressed_weak_close
    )


def _amount_shock_peak_confirmation(
    features: pd.DataFrame,
    config: FormulaAuxiliaryConfig,
) -> pd.Series:
    if _is_causal_profile(config):
        return pd.Series(True, index=features.index)
    next_amount_ratio = features["dollar_volume"].shift(-1) / features["dollar_volume"]
    return (
        next_amount_ratio.le(config.event_shock_next_amount_ratio_max)
        | next_amount_ratio.isna()
    )


def _chaikin_money_flow(
    signed_money_flow: pd.Series,
    dollar_volume: pd.Series,
    window: int,
) -> pd.Series:
    if window < 2:
        raise ValueError("Chaikin money flow window must be at least two.")
    minimum = max(2, window // 2)
    flow_sum = signed_money_flow.rolling(window, min_periods=minimum).sum()
    volume_sum = dollar_volume.rolling(window, min_periods=minimum).sum()
    return flow_sum / volume_sum.replace(0, np.nan)


def _td_runs(counts: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    start: int | None = None
    for index, value in enumerate(counts):
        if value == 1:
            start = index
        if start is not None and (value == 0 or index == len(counts) - 1):
            end = index - 1 if value == 0 else index
            maximum = int(np.max(counts[start : end + 1]))
            runs.append((start, end, maximum))
            start = None
    return runs


def continuity_counts(
    direction: np.ndarray, reset_mask: np.ndarray, cap: int = 24
) -> np.ndarray:
    if cap < 1:
        raise ValueError("continuity cap must be positive.")
    counts = np.zeros(len(direction), dtype=int)
    count = 0
    for index, value in enumerate(direction):
        if value == 0:
            count = 0
        elif reset_mask[index] or count == 0:
            count = 1
        else:
            count = min(count + 1, cap)
        counts[index] = count
    return counts


def bounded_continuity_counts(
    direction: np.ndarray,
    reset_mask: np.ndarray,
    cap: int = 24,
) -> np.ndarray:
    if cap < 1:
        raise ValueError("continuity cap must be positive.")
    counts = np.zeros(len(direction), dtype=int)
    count = 0
    exhausted = False
    for index, value in enumerate(direction):
        if value == 0:
            count = 0
            exhausted = False
        elif reset_mask[index]:
            count = 1
            exhausted = False
        elif exhausted:
            count = 0
        elif count == 0:
            count = 1
        elif count < cap:
            count += 1
        else:
            count = 0
            exhausted = True
        counts[index] = count
    return counts
