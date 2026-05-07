"""
Historical data backfill runner.

Supports two backfill modes:
1. SAR pipeline backfill (Hormuz-style, uses pipeline/run.py)
2. Multi-target backfill (uses satellite data for various regions)

Usage:
    # SAR pipeline backfill
    python scripts/run_backfill.py --region hormuz --start 2024-01-01 --end 2024-01-31

    # Multi-target backfill
    python scripts/run_backfill.py --targets hormuz panama_canal --start 2024-01-01 --end 2024-01-31
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import json
from typing import List, Dict

from pipeline.regions import resolve_region_output_base, resolve_region_paths, load_registry
from pipeline.run import run_single_day


def run_sar_backfill(
    region_id: str,
    start_date: date,
    end_date: date,
    output_base: str = "outputs",
    force: bool = False,
) -> List[Dict]:
    """
    Run SAR pipeline backfill for a region.

    Uses the original pipeline/run.py logic for Hormuz-style regions.

    Args:
        region_id: Region identifier
        start_date: Start date
        end_date: End date
        output_base: Output directory
        force: Force reprocessing even if data exists

    Returns:
        List of daily run results
    """
    aoi_path, gate_path = resolve_region_paths(region_id)
    results = []
    current = start_date
    region_root = Path(resolve_region_output_base(output_base, region_id))

    print(f"\n{'='*60}")
    print(f"SAR Backfill: {region_id}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"{'='*60}\n")

    while current <= end_date:
        day_dir = region_root / current.isoformat()
        metrics_path = day_dir / "metrics" / "daily.parquet"

        if metrics_path.exists() and not force:
            print(f"  {current.isoformat()}: Already exists, skipping")
            current += timedelta(days=1)
            continue

        print(f"  Processing {current.isoformat()}...")
        try:
            result = run_single_day(
                target_date=current,
                aoi_path=aoi_path,
                gate_path=gate_path,
                output_base=output_base,
                region_id=region_id,
            )
            results.append(result)
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "date": current.isoformat(),
                "status": "error",
                "message": str(e),
            })

        current += timedelta(days=1)

    return results


def run_multi_backfill(
    targets: List[str],
    start_date: str,
    end_date: str,
    output_base: str = "outputs",
    max_scenes: int = 20,
) -> Dict:
    """
    Run multi-target backfill using satellite data.

    This uses Planetary Computer directly to backfill historical data
    for regions that don't need the full SAR pipeline processing.

    Args:
        targets: List of region IDs to backfill
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_base: Output directory
        max_scenes: Maximum scenes to process per region

    Returns:
        Summary of backfill results
    """
    from pipeline.backfill_multi import run_multi_backfill as _run_multi

    results = _run_multi(
        targets=targets,
        start_date=start_date,
        end_date=end_date,
        output_base=output_base,
    )

    return results


def run_auto_backfill(
    regions_filter: List[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    output_base: str = "outputs",
) -> Dict:
    """
    Automatically backfill all active regions.

    Uses the auto_aoi configs to backfill detected regions.

    Args:
        regions_filter: Optional list of region IDs to process
        start_date: Start date (default: 90 days ago)
        end_date: End date (default: today)
        output_base: Output directory

    Returns:
        Summary of backfill results
    """
    from datetime import datetime, timedelta

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    # Find all auto_aoi configs
    configs_dir = Path("configs")
    auto_aoi_files = list(configs_dir.glob("aoi_auto_*.geojson"))

    if not auto_aoi_files:
        print("No auto_aoi configs found")
        return {}

    # Extract region names from filenames
    regions = []
    for f in auto_aoi_files:
        region_name = f.stem.replace("aoi_auto_", "")
        if regions_filter is None or region_name in regions_filter:
            regions.append(region_name)

    if not regions:
        print(f"No matching regions found (filter: {regions_filter})")
        return {}

    print(f"Auto-backfilling {len(regions)} regions: {regions}")

    return run_multi_backfill(
        targets=regions,
        start_date=start_date,
        end_date=end_date,
        output_base=output_base,
    )


def main():
    parser = argparse.ArgumentParser(
        description="QuantTrade backfill runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SAR pipeline backfill
  python scripts/run_backfill.py --mode sar --region hormuz --start 2024-01-01 --end 2024-01-31

  # Multi-target backfill
  python scripts/run_backfill.py --mode multi --targets hormuz panama_canal --start 2024-01-01 --end 2024-01-31

  # Auto backfill (uses aoi_auto_* configs)
  python scripts/run_backfill.py --mode auto --regions indiana kentucky
        """
    )

    parser.add_argument(
        "--mode",
        choices=["sar", "multi", "auto"],
        default="auto",
        help="Backfill mode: sar (pipeline), multi (satellite), auto (detect regions)"
    )

    # SAR mode args
    parser.add_argument("--region", type=str, help="Region ID (for SAR mode)")

    # Multi mode args
    parser.add_argument("--targets", nargs="*", help="Region IDs to backfill (for multi mode)")

    # Auto mode args
    parser.add_argument("--regions", nargs="*", help="Region names to filter (for auto mode)")

    # Common args
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force reprocessing (SAR mode only)")

    args = parser.parse_args()

    if args.mode == "sar":
        if not args.region:
            parser.error("--region required for SAR mode")
        if not args.start or not args.end:
            parser.error("--start and --end required for SAR mode")

        results = run_sar_backfill(
            region_id=args.region,
            start_date=date.fromisoformat(args.start),
            end_date=date.fromisoformat(args.end),
            output_base=args.output,
            force=args.force,
        )

        # Save summary
        summary = {
            "mode": "sar",
            "region": args.region,
            "start_date": args.start,
            "end_date": args.end,
            "results": results,
        }
        output_path = Path(args.output) / "backfill"
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / f"{args.region}_backfill_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )

        print(f"\n{'='*60}")
        print("SAR BACKFILL COMPLETE")
        print(f"Processed: {len(results)} days")
        print(f"Summary saved to: {output_path / f'{args.region}_backfill_summary.json'}")
        print(f"{'='*60}")

    elif args.mode == "multi":
        if not args.targets:
            parser.error("--targets required for multi mode")
        if not args.start or not args.end:
            parser.error("--start and --end required for multi mode")

        results = run_multi_backfill(
            targets=args.targets,
            start_date=args.start,
            end_date=args.end,
            output_base=args.output,
        )

        print(f"\n{'='*60}")
        print("MULTI-TARGET BACKFILL COMPLETE")
        print(f"Targets: {args.targets}")
        print(f"{'='*60}")

    elif args.mode == "auto":
        results = run_auto_backfill(
            regions_filter=args.regions,
            start_date=args.start,
            end_date=args.end,
            output_base=args.output,
        )

        if results:
            print(f"\n{'='*60}")
            print("AUTO BACKFILL COMPLETE")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
