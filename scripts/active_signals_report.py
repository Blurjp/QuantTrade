#!/usr/bin/env python3
"""Print a compact table of active region signals for a daily summary."""

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def format_cell(value, width):
    text = str(value)
    if len(text) > width:
        return text[: width - 1] + "."
    return text.ljust(width)


def main() -> int:
    parser = argparse.ArgumentParser(description="Active region signal snapshot")
    parser.add_argument("--date", required=True, help="Daily summary date to inspect")
    parser.add_argument("--output", default="outputs", help="Output directory")
    args = parser.parse_args()

    summary = load_json(Path(args.output) / args.date / "daily_summary.json")
    if not summary:
        print("Daily summary not found")
        return 1

    print("Active Region Signals")
    print("=" * 79)
    headers = [
        format_cell("Region", 24),
        format_cell("Action", 8),
        format_cell("Raw", 8),
        format_cell("Conf", 6),
        format_cell("Score", 8),
        format_cell("Signal", 21),
    ]
    print(" ".join(headers))
    print("-" * 79)

    for region_id in sorted(summary.get("signals", {}).keys()):
        signal = summary["signals"][region_id]
        score = signal.get("vote_score", signal.get("ndvi_change", ""))
        row = [
            format_cell(region_id, 24),
            format_cell(signal.get("trading_action", "FLAT"), 8),
            format_cell(signal.get("raw_trading_action", signal.get("trading_action", "FLAT")), 8),
            format_cell(signal.get("confidence", "Low"), 6),
            format_cell(f"{score:.3f}" if isinstance(score, (int, float)) else score, 8),
            format_cell(signal.get("signal", "No data"), 21),
        ]
        print(" ".join(row))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
