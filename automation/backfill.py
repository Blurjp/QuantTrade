"""
Historical backfill runner.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from pipeline.regions import resolve_region_output_base, resolve_region_paths
from pipeline.run import run_single_day


def run_region_backfill(
    region_id: str,
    start_date: date,
    end_date: date,
    output_base: str = "outputs",
    force: bool = False,
) -> list[dict]:
    aoi_path, gate_path = resolve_region_paths(region_id)
    results = []
    current = start_date
    region_root = Path(resolve_region_output_base(output_base, region_id))

    while current <= end_date:
        day_dir = region_root / current.isoformat()
        metrics_path = day_dir / "metrics" / "daily.parquet"
        if metrics_path.exists() and not force:
            current += timedelta(days=1)
            continue

        results.append(
            run_single_day(
                target_date=current,
                aoi_path=aoi_path,
                gate_path=gate_path,
                output_base=output_base,
                region_id=region_id,
            )
        )
        current += timedelta(days=1)

    return results


def main():
    parser = argparse.ArgumentParser(description="QuantTrade backfill runner")
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    results = run_region_backfill(
        region_id=args.region,
        start_date=date.fromisoformat(args.start),
        end_date=date.fromisoformat(args.end),
        output_base=args.output,
        force=args.force,
    )
    print(results)


if __name__ == "__main__":
    main()
