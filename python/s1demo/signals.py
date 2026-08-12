from __future__ import annotations

import pandas as pd


def append_signal(series: pd.Series, label: str) -> pd.Series:
    return series.where(series == "", series + "|") + label


def build_signal_reason(df: pd.DataFrame, prefix: str) -> pd.Series:
    parts = []
    for label, column in (
        ("B", f"{prefix}_b_reason"),
        ("S", f"{prefix}_s_reason"),
        ("IB", f"{prefix}_ib_reason"),
        ("E", f"{prefix}_e_reason"),
    ):
        if column in df:
            parts.append(
                df[column]
                .fillna("")
                .apply(lambda value, label=label: f"{label}:{value}" if value else "")
            )
    if not parts:
        return pd.Series("", index=df.index)
    reason_frame = pd.concat(parts, axis=1)
    return reason_frame.apply(
        lambda row: "|".join(value for value in row if value), axis=1
    )
