from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
import zlib


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
SAMPLE_SYMBOLS = ("AAPL", "GOOG", "MP", "KLAC")


def main() -> None:
    sys.path.insert(0, str(PYTHON_ROOT))
    from s1demo import calculate_json

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

    for symbol in SAMPLE_SYMBOLS:
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
        if not result["bars"] or result["bars"][-1]["time"] > manifest["latest"]:
            raise RuntimeError(f"{symbol} produced an invalid chart result.")
        print(
            f"{symbol}: {len(result['bars'])} bars / "
            f"{len(result['markers'])} markers / {len(result['numbers'])} numbers"
        )

    print("Demo data and formula runtime verification passed.")


if __name__ == "__main__":
    main()
