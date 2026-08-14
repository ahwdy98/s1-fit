from __future__ import annotations

from datetime import date, timedelta
import json

import pandas as pd

from .monotonic_zigzag import generate_removed_restricted_candidates
from .zigzag_signals import ZigZagSignalConfig, generate_zigzag_signals


def _frame_from_payload(payload: dict[str, object]) -> pd.DataFrame:
    start = date.fromisoformat(str(payload["s"]))
    dates = [start + timedelta(days=int(offset)) for offset in payload["d"]]
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": payload["o"],
            "high": payload["h"],
            "low": payload["l"],
            "close": payload["c"],
            "volume": payload["v"],
            "amount": payload["a"],
            "turnover_rate": payload["t"],
        }
    )


def calculate(payload: dict[str, object]) -> dict[str, object]:
    frame = _frame_from_payload(payload)
    signals = generate_zigzag_signals(
        frame,
        ZigZagSignalConfig(
            max_confirmation_bars=None,
            monotonic_markers=True,
            auxiliary_mode="formula",
            formula_auxiliary_profile="restricted-formula",
        ),
    )
    bars = []
    markers = []
    numbers = []
    for row in signals.itertuples(index=False):
        day = pd.Timestamp(row.date).date().isoformat()
        bars.append(
            {
                "time": day,
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": int(row.volume),
            }
        )
        for label, active, position, color, shape in (
            ("B", row.is_b, "belowBar", "#00796b", "arrowUp"),
            ("S", row.is_s, "aboveBar", "#d32f2f", "arrowDown"),
            ("IB", row.is_ib, "belowBar", "#0277bd", "circle"),
            ("E", row.is_e, "aboveBar", "#ef6c00", "circle"),
        ):
            if active:
                markers.append(
                    {
                        "time": day,
                        "position": position,
                        "color": color,
                        "shape": shape,
                        "text": label,
                        "kind": str(
                            getattr(
                                row,
                                "zigzag_b_kind"
                                if label == "B"
                                else "zigzag_s_kind"
                                if label == "S"
                                else "formula_ib_reason"
                                if label == "IB"
                                else "formula_e_reason",
                                "",
                            )
                        ),
                    }
                )
        count = int(row.continuity or 0)
        direction = int(row.continuity_dir or 0)
        if count > 0 and direction != 0:
            numbers.append(
                {
                    "time": day,
                    "position": "belowBar" if direction > 0 else "aboveBar",
                    "color": "#00897b" if direction > 0 else "#6a1b9a",
                    "shape": "circle",
                    "text": str(count),
                    "value": count,
                    "direction": direction,
                }
            )
    removed_markers = []
    for candidate in generate_removed_restricted_candidates(frame):
        day = pd.Timestamp(frame.iloc[candidate.marker]["date"]).date().isoformat()
        removed_markers.append(
            {
                "time": day,
                "position": "belowBar" if candidate.side == "B" else "aboveBar",
                "color": "#78909c" if candidate.side == "B" else "#9e7777",
                "shape": "circle",
                "text": f"{candidate.side}×",
                "kind": f"removed_{candidate.side.lower()}",
            }
        )
    return {
        "bars": bars,
        "markers": markers,
        "removedMarkers": removed_markers,
        "numbers": numbers,
        "formulaProfile": "restricted-formula",
    }


def calculate_json(payload_json: str) -> str:
    return json.dumps(calculate(json.loads(payload_json)), separators=(",", ":"))
