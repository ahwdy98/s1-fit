from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
import gzip
import json
from pathlib import Path
import shutil
import sqlite3
import zlib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent / "xys" / "data" / "bfq_daily.sqlite3"
DEFAULT_OUTPUT = ROOT / "data"
SHARD_COUNT = 256
DEFAULT_VISIBLE_DAYS = 251
DEFAULT_WARMUP_DAYS = 100


def shard_for(symbol: str) -> int:
    return zlib.crc32(symbol.encode("ascii")) % SHARD_COUNT


def clean_number(value: object) -> int | float | None:
    if value is None:
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def encode_symbol(rows: list[sqlite3.Row]) -> dict[str, object]:
    first = date.fromisoformat(rows[0]["trade_date"])
    return {
        "s": first.isoformat(),
        "d": [(date.fromisoformat(row["trade_date"]) - first).days for row in rows],
        "o": [clean_number(row["open"]) for row in rows],
        "h": [clean_number(row["high"]) for row in rows],
        "l": [clean_number(row["low"]) for row in rows],
        "c": [clean_number(row["close"]) for row in rows],
        "v": [int(row["volume"]) for row in rows],
        "a": [clean_number(row["amount"]) for row in rows],
        "t": [clean_number(row["turnover_rate"]) for row in rows],
    }


def build(source: Path, output: Path, visible_days: int, warmup_days: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    shards_dir = output / "shards"
    if shards_dir.exists():
        shutil.rmtree(shards_dir)
    shards_dir.mkdir()

    with sqlite3.connect(source) as connection:
        connection.row_factory = sqlite3.Row
        dates = [
            row[0]
            for row in connection.execute(
                """
                SELECT DISTINCT trade_date
                FROM daily_prices
                ORDER BY trade_date DESC
                LIMIT ?
                """,
                (visible_days + warmup_days,),
            )
        ]
        if not dates:
            raise RuntimeError("The source database contains no daily prices.")
        calculation_cutoff = min(dates)
        visible_cutoff = dates[min(visible_days, len(dates)) - 1]
        latest = max(dates)
        cursor = connection.execute(
            """
            SELECT symbol, trade_date, open, high, low, close,
                   volume, amount, turnover_rate
            FROM daily_prices
            WHERE trade_date >= ?
            ORDER BY symbol, trade_date
            """,
            (calculation_cutoff,),
        )

        handles: dict[int, gzip.GzipFile] = {}
        first_record = defaultdict(lambda: True)
        symbols: list[str] = []
        row_count = 0

        def write_symbol(symbol: str, rows: list[sqlite3.Row]) -> None:
            nonlocal row_count
            shard = shard_for(symbol)
            if shard not in handles:
                handle = gzip.open(
                    shards_dir / f"{shard:02x}.json.gz",
                    "wt",
                    encoding="utf-8",
                    compresslevel=9,
                    newline="",
                )
                handle.write("{")
                handles[shard] = handle
            handle = handles[shard]
            if not first_record[shard]:
                handle.write(",")
            first_record[shard] = False
            handle.write(json.dumps(symbol))
            handle.write(":")
            handle.write(
                json.dumps(encode_symbol(rows), separators=(",", ":"), allow_nan=False)
            )
            symbols.append(symbol)
            row_count += len(rows)

        current_symbol: str | None = None
        current_rows: list[sqlite3.Row] = []
        for row in cursor:
            symbol = str(row["symbol"])
            if current_symbol is not None and symbol != current_symbol:
                write_symbol(current_symbol, current_rows)
                current_rows = []
            current_symbol = symbol
            current_rows.append(row)
        if current_symbol is not None:
            write_symbol(current_symbol, current_rows)

        for handle in handles.values():
            handle.write("}")
            handle.close()

    manifest = {
        "format": 1,
        "visible_trading_days": min(visible_days, len(dates)),
        "warmup_trading_days": max(0, len(dates) - visible_days),
        "calculation_trading_days": len(dates),
        "calculation_cutoff": calculation_cutoff,
        "cutoff": visible_cutoff,
        "latest": latest,
        "symbols": sorted(symbols),
        "rows": row_count,
        "shards": SHARD_COUNT,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":")), encoding="utf-8"
    )
    total_bytes = sum(path.stat().st_size for path in shards_dir.glob("*.gz"))
    print(
        f"Built {len(symbols):,} symbols / {row_count:,} rows / "
        f"{len(handles)} shards / {total_bytes / 1024 / 1024:.1f} MiB"
    )
    print(f"Calculation range: {calculation_cutoff} .. {latest}")
    print(f"Visible range: {visible_cutoff} .. {latest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static S1 demo data shards.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--visible-days", type=int, default=DEFAULT_VISIBLE_DAYS)
    parser.add_argument("--warmup-days", type=int, default=DEFAULT_WARMUP_DAYS)
    args = parser.parse_args()
    if args.visible_days < 1 or args.warmup_days < 0:
        parser.error("visible-days must be positive and warmup-days cannot be negative")
    build(args.source, args.output, args.visible_days, args.warmup_days)


if __name__ == "__main__":
    main()
