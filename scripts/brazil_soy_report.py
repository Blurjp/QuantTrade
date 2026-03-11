#!/usr/bin/env python3
"""Print the latest Brazil soy live signal and backtest snapshot."""

import argparse
import json
from pathlib import Path


BRAZIL_WEIGHTS = {
    "brazil_soy_north": 0.35,
    "brazil_soy_central": 0.40,
    "brazil_soy_southeast": 0.25,
}


def vote_from_action(action: str) -> float:
    if action == "LONG":
        return 1.0
    if action == "SHORT":
        return -1.0
    return 0.0


def compute_meta_backtest(backtests):
    total_weight = 0.0
    weighted_vote = 0.0

    for region_id, backtest in backtests.items():
        weight = BRAZIL_WEIGHTS.get(region_id, 0.0)
        directions = backtest.get("backtest", {}).get("by_direction", {})
        long_acc = directions.get("long", {}).get("accuracy", 0.0)
        short_acc = directions.get("short", {}).get("accuracy", 0.0)
        long_count = directions.get("long", {}).get("count", 0)
        short_count = directions.get("short", {}).get("count", 0)

        if long_count >= 3 and long_acc >= short_acc:
            edge = max(long_acc - 0.5, 0.0)
            weighted_vote += weight * edge
            total_weight += weight
        elif short_count >= 3:
            edge = max(short_acc - 0.5, 0.0)
            weighted_vote -= weight * edge
            total_weight += weight

    if total_weight == 0:
        return None

    vote_score = weighted_vote / total_weight
    if vote_score > 0.05:
        stance = "LONG"
    elif vote_score < -0.05:
        stance = "SHORT"
    else:
        stance = "FLAT"

    return {"stance": stance, "vote_score": vote_score}


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
    backtest_dir = output_base / "backtest"
    region_ids = [
        "brazil_soy_north",
        "brazil_soy_central",
        "brazil_soy_southeast",
        "brazil_soy",
    ]

    print("Brazil Soy Report")
    print("=" * 40)

    if daily_summary:
        print(f"Daily date: {args.date}")
        for region_id in region_ids:
            signal = daily_summary.get("signals", {}).get(region_id)
            result = next((item for item in daily_summary.get("results", []) if item.get("region") == region_id), None)
            if not signal and not result:
                continue

            detection = (result or {}).get("detection", {})
            details = detection.get("details", [{}])
            latest = details[0] if details else {}

            print(f"{region_id}:")
            print(f"  status: {(result or {}).get('status', 'missing')}")
            print(f"  signal: {(signal or {}).get('signal', 'N/A')}")
            print(f"  action: {(signal or {}).get('trading_action', 'N/A')}")
            print(f"  confidence: {(signal or {}).get('confidence', 'N/A')}")
            print(f"  ndvi current: {(signal or {}).get('ndvi_current', latest.get('ndvi_mean', 'N/A'))}")
            print(f"  ndvi baseline: {(signal or {}).get('ndvi_baseline', 'N/A')}")
            print(f"  ndvi change: {(signal or {}).get('ndvi_change', 'N/A')}")
            print(f"  scene: {detection.get('metadata', {}).get('scene_id', 'N/A')}")
        print()
    else:
        print("Daily summary: missing")
        print()

    print("Backtests")
    found_backtest = False
    backtests = {}
    for region_id in region_ids:
        backtest = load_json(backtest_dir / f"{region_id}_Soybeans_backtest.json")
        if not backtest:
            continue
        found_backtest = True
        backtests[region_id] = backtest
        stats = backtest.get("backtest", {})
        directions = stats.get("by_direction", {})
        print(f"{region_id}: {backtest.get('start_date')} -> {backtest.get('end_date')}")
        print(f"  overall accuracy: {stats.get('overall_accuracy', 0) * 100:.1f}%")
        print(f"  directional signals: {stats.get('total_directional_signals', 0)}")
        for direction in ("long", "short"):
            direction_stats = directions.get(direction)
            if direction_stats:
                print(
                    f"  {direction}: {direction_stats.get('accuracy', 0) * 100:.1f}% "
                    f"({direction_stats.get('count', 0)} signals, avg {direction_stats.get('avg_return', 0) * 100:.2f}%)"
                )
    if not found_backtest:
        print("No Brazil soy backtests found")

    meta_signal = daily_summary.get("signals", {}).get("brazil_soy_meta") if daily_summary else None
    meta_backtest = compute_meta_backtest(backtests)
    if meta_signal or meta_backtest:
        print()
        print("Brazil Soy Meta")
        if meta_signal:
            print(f"live action: {meta_signal.get('trading_action')}")
            print(f"raw action: {meta_signal.get('raw_trading_action', meta_signal.get('trading_action'))}")
            print(f"live confidence: {meta_signal.get('confidence')}")
            print(f"live vote score: {meta_signal.get('vote_score', 0):.3f}")
        if meta_backtest:
            print(f"backtest stance: {meta_backtest['stance']}")
            print(f"backtest vote score: {meta_backtest['vote_score']:.3f}")

    if daily_summary:
        print()
        print("All Active Regions")
        print("-" * 40)
        for region_id in sorted(daily_summary.get("signals", {}).keys()):
            signal = daily_summary["signals"][region_id]
            print(
                f"{region_id}: {signal.get('trading_action', 'FLAT')} | "
                f"{signal.get('confidence', 'Low')} | {signal.get('signal', 'No data')}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
