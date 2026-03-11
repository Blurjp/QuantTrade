#!/usr/bin/env python3
"""Print the latest Brazil soy live signal and backtest snapshot."""

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Brazil soy signal report")
    parser.add_argument("--date", default="2026-03-08", help="Daily summary date to inspect")
    parser.add_argument("--output", default="outputs", help="Output directory")
    args = parser.parse_args()

    output_base = Path(args.output)
    daily_summary = load_json(output_base / args.date / "daily_summary.json")
    backtest = load_json(output_base / "backtest" / "brazil_soy_Soybeans_backtest.json")

    print("Brazil Soy Report")
    print("=" * 40)

    if daily_summary:
        signal = daily_summary.get("signals", {}).get("brazil_soy", {})
        result = next((item for item in daily_summary.get("results", []) if item.get("region") == "brazil_soy"), {})
        detection = result.get("detection", {})
        details = detection.get("details", [{}])
        latest = details[0] if details else {}

        print(f"Daily date: {args.date}")
        print(f"Daily status: {result.get('status', 'unknown')}")
        print(f"Signal: {signal.get('signal', 'N/A')}")
        print(f"Action: {signal.get('trading_action', 'N/A')}")
        print(f"Confidence: {signal.get('confidence', 'N/A')}")
        print(f"NDVI current: {signal.get('ndvi_current', latest.get('ndvi_mean', 'N/A'))}")
        print(f"NDVI baseline: {signal.get('ndvi_baseline', 'N/A')}")
        print(f"NDVI change: {signal.get('ndvi_change', 'N/A')}")
        print(f"Scene: {detection.get('metadata', {}).get('scene_id', 'N/A')}")
        print()
    else:
        print("Daily summary: missing")
        print()

    if backtest:
        stats = backtest.get("backtest", {})
        directions = stats.get("by_direction", {})
        print("Backtest")
        print(f"Window: {backtest.get('start_date')} -> {backtest.get('end_date')}")
        print(f"Overall accuracy: {stats.get('overall_accuracy', 0) * 100:.1f}%")
        print(f"Directional signals: {stats.get('total_directional_signals', 0)}")
        for direction in ("long", "short"):
            direction_stats = directions.get(direction)
            if direction_stats:
                print(
                    f"{direction}: {direction_stats.get('accuracy', 0) * 100:.1f}% "
                    f"({direction_stats.get('count', 0)} signals, avg {direction_stats.get('avg_return', 0) * 100:.2f}%)"
                )
    else:
        print("Backtest: missing")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
