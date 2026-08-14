from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
import zlib


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
EXPECTED_MARKERS = {
    "AAPL": {("2026-07-29", "S")},
    "AMBA": {("2026-06-01", "B")},
    "KO": {("2026-04-23", "B")},
    "NVTS": {("2026-06-04", "S")},
}


def main() -> None:
    sys.path.insert(0, str(PYTHON_ROOT))
    from s1demo import calculate_json
    from s1demo.monotonic_zigzag import generate_restricted_monotonic_candidates
    from s1demo.runtime import _frame_from_payload

    manifest = json.loads((ROOT / "data" / "manifest.json").read_text("utf-8"))
    shards = sorted((ROOT / "data" / "shards").glob("*.json.gz"))
    if len(shards) != manifest["shards"]:
        raise RuntimeError(
            f"Expected {manifest['shards']} shards, found {len(shards)}."
        )

    total_bytes = sum(path.stat().st_size for path in shards)
    largest = max(shards, key=lambda path: path.stat().st_size)
    print(
        f"Manifest: {len(manifest['symbols']):,} symbols / "
        f"{manifest['rows']:,} rows / {total_bytes / 1024 / 1024:.1f} MiB"
    )
    print(
        f"Visible: {manifest['cutoff']} .. {manifest['latest']} / "
        f"warmup starts {manifest['calculation_cutoff']}"
    )
    print(f"Largest shard: {largest.name} / {largest.stat().st_size / 1024:.1f} KiB")
    if largest.stat().st_size >= 100 * 1024 * 1024:
        raise RuntimeError("A shard exceeds GitHub's 100 MiB normal Git limit.")

    for symbol, expected_markers in EXPECTED_MARKERS.items():
        shard = zlib.crc32(symbol.encode("ascii")) % manifest["shards"]
        with gzip.open(
            ROOT / "data" / "shards" / f"{shard:02x}.json.gz",
            "rt",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)[symbol]
        lengths = {key: len(value) for key, value in payload.items() if isinstance(value, list)}
        if len(set(lengths.values())) != 1:
            raise RuntimeError(f"{symbol} has inconsistent column lengths: {lengths}")
        result = json.loads(calculate_json(json.dumps(payload, separators=(",", ":"))))
        if result.get("formulaProfile") != "restricted-formula":
            raise RuntimeError(f"{symbol} did not use restricted-formula.")
        if not result["bars"] or result["bars"][-1]["time"] > manifest["latest"]:
            raise RuntimeError(f"{symbol} produced an invalid chart result.")
        actual_markers = {
            (marker["time"], marker["text"]) for marker in result["markers"]
        }
        missing = expected_markers - actual_markers
        if missing:
            raise RuntimeError(f"{symbol} is missing expected markers: {sorted(missing)}")
        if symbol == "KO" and ("2026-07-23", "B") in actual_markers:
            raise RuntimeError("KO incorrectly contains a 2026-07-23 B marker.")
        removed_markers = {
            (marker["time"], marker["kind"].removeprefix("removed_").upper())
            for marker in result.get("removedMarkers", [])
        }
        active_bs = {
            (marker["time"], marker["text"])
            for marker in result["markers"]
            if marker["text"] in {"B", "S"}
        }
        if not removed_markers:
            raise RuntimeError(f"{symbol} did not expose any withdrawn B/S markers.")
        overlap = active_bs & removed_markers
        if overlap:
            raise RuntimeError(
                f"{symbol} exposes active markers as withdrawn: {sorted(overlap)}"
            )
        bs_sequence = [
            marker["text"]
            for marker in sorted(result["markers"], key=lambda marker: marker["time"])
            if marker["text"] in {"B", "S"}
        ]
        if any(left == right == "S" for left, right in zip(bs_sequence, bs_sequence[1:])):
            raise RuntimeError(f"{symbol} contains repeated S markers without a B.")
        frame = _frame_from_payload(payload)
        previous: set[tuple[int, str]] = set()
        removed: set[tuple[int, str]] = set()
        for stop in range(4, len(frame) + 1):
            current = {
                (candidate.marker, candidate.side)
                for candidate in generate_restricted_monotonic_candidates(
                    frame.iloc[:stop]
                )
            }
            historical_new = {
                marker for marker in current - previous if marker[0] < stop - 1
            }
            restored = current & removed
            if historical_new or restored:
                raise RuntimeError(
                    f"{symbol} backfilled or restored B/S markers: "
                    f"new={sorted(historical_new)}, restored={sorted(restored)}"
                )
            removed.update(previous - current)
            previous = current
        print(
            f"{symbol}: {len(result['bars'])} bars / "
            f"{len(result['markers'])} markers / {len(result['numbers'])} numbers"
            f" / {len(removed_markers)} withdrawn"
        )

    print("Demo data and formula runtime verification passed.")


if __name__ == "__main__":
    main()
