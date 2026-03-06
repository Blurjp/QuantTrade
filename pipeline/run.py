"""
Main Pipeline Runner

Orchestrates the full SAR throughput pipeline.
"""

import argparse
import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr
import numpy as np

from pipeline.manifest import run_manifest_builder, load_aoi
from pipeline.loader import (
    load_scenes_from_manifest,
    iter_scenes,
    compute_coverage_score,
    sign_items
)
from pipeline.preprocess import preprocess_scene
from pipeline.detection import run_detection_pipeline
from pipeline.tracking import link_detections, save_tracklets
from pipeline.crossings import load_gate_line, infer_crossings, save_crossings, fallback_scene_crossings
from pipeline.metrics import aggregate_daily_metrics, save_metrics


def run_calibration_step(
    output_base: str,
    ais_path: str,
    gate_path: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> dict:
    """
    Run calibration step against AIS data.

    Args:
        output_base: Base output directory
        ais_path: Path to AIS Parquet data
        gate_path: Path to gate line GeoJSON
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Calibration results dict
    """
    from pipeline.calibration import run_calibration_workflow

    # Find satellite metrics
    metrics_path = Path(output_base) / "metrics" / "daily.parquet"

    if not metrics_path.exists():
        # Try to aggregate from daily crossings
        print("Aggregating metrics from daily crossings...")
        all_metrics = []

        for day_dir in sorted(Path(output_base).iterdir()):
            if day_dir.is_dir() and day_dir.name.startswith("20"):
                crossings_file = day_dir / "crossings" / "crossings.parquet"
                if crossings_file.exists():
                    crossings = pd.read_parquet(crossings_file)
                    if len(crossings) > 0:
                        # Get manifest for coverage
                        manifest_path = day_dir / "manifests" / "manifest.parquet"
                        manifest = pd.read_parquet(manifest_path) if manifest_path.exists() else None

                        aoi = load_aoi()
                        aoi_geom = aoi["features"][0]["geometry"]

                        metrics = aggregate_daily_metrics(crossings, manifest, aoi_geom)
                        all_metrics.append(metrics)

        if all_metrics:
            combined = pd.concat(all_metrics, ignore_index=True)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(metrics_path, index=False)
            print(f"Saved metrics to {metrics_path}")
        else:
            print("No metrics found for calibration")
            return {}

    # Load gate coordinates
    with open(gate_path) as f:
        gate_data = json.load(f)

    # Handle both Feature and FeatureCollection formats
    if gate_data['type'] == 'FeatureCollection':
        gate_coords = gate_data['features'][0]['geometry']['coordinates']
    elif gate_data['type'] == 'Feature':
        gate_coords = gate_data['geometry']['coordinates']
    else:
        gate_coords = gate_data['coordinates']

    # Run calibration
    return run_calibration_workflow(
        sat_metrics_path=str(metrics_path),
        ais_data_path=ais_path,
        gate_coords=gate_coords,
        output_path=str(Path(output_base) / "calibration"),
        start_date=start_date,
        end_date=end_date
    )


def run_single_day(
    target_date: date,
    aoi_path: str = "configs/aoi_hormuz.geojson",
    gate_path: str = "configs/gate_line.geojson",
    output_base: str = "outputs",
    return_detections: bool = False
) -> dict:
    """
    Run pipeline for a single day.

    This is the core execution loop that:
    1. Builds manifest and saves STAC items
    2. Loads scenes (with Planetary Computer signing)
    3. For each scene: preprocess → detect → save
    4. Aggregates daily metrics with coverage score
    5. Writes run report

    Args:
        target_date: Date to process
        aoi_path: Path to AOI GeoJSON
        gate_path: Path to gate line GeoJSON
        output_base: Base output directory
        return_detections: If True, return detections DataFrame in report

    Returns:
        Run report dict (with 'detections_df' key if return_detections=True)
    """
    start_time = time.time()
    report = {
        "date": target_date.isoformat(),
        "status": "started",
        "scenes_processed": 0,
        "total_detections": 0,
        "errors": []
    }

    print(f"\n{'='*60}")
    print(f"Hormuz SAR Throughput Pipeline - Single Day")
    print(f"Date: {target_date}")
    print(f"{'='*60}\n")

    # Create output directories
    output_dir = Path(output_base)
    date_dir = output_dir / target_date.strftime("%Y-%m-%d")
    detections_dir = date_dir / "detections"
    qa_dir = date_dir / "qa"

    for d in [detections_dir, qa_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load AOI for coverage computation
    aoi = load_aoi(aoi_path)
    aoi_geom = aoi["features"][0]["geometry"]

    # Step 1: Build manifest and save STAC items
    print("Step 1: Building manifest...")
    step_start = time.time()

    manifest, items_path = run_manifest_builder(
        start_date=target_date,
        end_date=target_date,
        aoi_path=aoi_path,
        output_path=str(date_dir / "manifests")
    )

    report["manifest_build_time_s"] = round(time.time() - step_start, 2)
    report["scenes_found"] = len(manifest)

    if len(manifest) == 0:
        print("No scenes found. Exiting.")
        report["status"] = "no_scenes"
        _write_report(report, qa_dir)
        return report

    # Step 2: Compute coverage score
    print("\nStep 2: Computing coverage score...")
    from pipeline.loader import load_stac_items
    items = load_stac_items(str(items_path))

    coverage = compute_coverage_score(items, aoi_geom)
    report["coverage"] = coverage
    print(f"Coverage score: {coverage['coverage_score']:.2%}")
    print(f"Scenes covering AOI: {coverage['num_scenes']}")

    # Step 3: Process each scene
    print("\nStep 3: Processing scenes...")
    step_start = time.time()

    all_detections = []
    scene_reports = []

    # Load items for iteration
    items = load_stac_items(str(items_path))
    signed_items = sign_items(items)

    for i, item in enumerate(signed_items):
        scene_id = item.id
        datetime_str = item.datetime.isoformat()

        print(f"\n  [{i+1}/{len(signed_items)}] Processing {scene_id}...")

        try:
            # Load single scene with explicit CRS and resolution
            from odc import stac

            # Get AOI bounds for cropping
            aoi_coords = aoi_geom.get("coordinates", [[]])[0]
            lons = [c[0] for c in aoi_coords]
            lats = [c[1] for c in aoi_coords]
            bbox = [min(lons), min(lats), max(lons), max(lats)]

            ds = stac.load(
                [item],
                bands=["vh", "vv"],
                chunks={"x": 1024, "y": 1024},
                crs="EPSG:4326",
                resolution=0.0001,  # ~10m in degrees
                bbox=bbox,
                preserve_original_order=True
            )

            scene_report = {
                "scene_id": scene_id,
                "datetime": datetime_str,
                "status": "loaded"
            }

            # Preprocess
            ds_preprocessed, qc_metrics = preprocess_scene(ds)
            scene_report["qc_metrics"] = qc_metrics

            # Detect (CFAR baseline)
            detections_df = run_detection_pipeline(
                ds_preprocessed,
                scene_id=scene_id,
                datetime_str=datetime_str,
                output_path=str(detections_dir)
            )

            scene_report["detections"] = len(detections_df)
            scene_report["status"] = "success"

            all_detections.append(detections_df)
            report["scenes_processed"] += 1
            report["total_detections"] += len(detections_df)

            # Save scene-level QC
            qc_file = qa_dir / f"{scene_id}_qc.json"
            with open(qc_file, "w") as f:
                json.dump(scene_report, f, indent=2, default=str)

        except Exception as e:
            error_msg = f"Error processing {scene_id}: {str(e)}"
            print(f"    ERROR: {error_msg}")
            report["errors"].append(error_msg)
            scene_reports.append({"scene_id": scene_id, "status": "error", "error": str(e)})

    report["scene_processing_time_s"] = round(time.time() - step_start, 2)

    # Step 4: Aggregate daily detections
    print(f"\nStep 4: Aggregating daily detections...")
    if all_detections:
        daily_detections = pd.concat(all_detections, ignore_index=True)
        daily_file = detections_dir / "daily_detections.parquet"
        daily_detections.to_parquet(daily_file, index=False)
        print(f"Saved {len(daily_detections)} total detections to {daily_file}")
    else:
        daily_detections = pd.DataFrame()
        print("No detections to aggregate.")

    # Step 5: Tracklet linking (if detections exist)
    print(f"\nStep 5: Tracklet linking...")
    if len(daily_detections) > 0:
        try:
            # Use tighter parameters for ship tracklets:
            # Ships move ~15-20 knots = In 6 hours: ~110-220 km
            # Use 5km threshold for same-day linking
            tracklets = link_detections(
                daily_detections,
                max_distance_km=5.0,  # 5km - reasonable ship movement in 6 hours
                max_time_gap_hours=6.0,  # 6 hours between scenes
                min_tracklet_length=2
            )
            save_tracklets(tracklets, str(date_dir / "tracklets"))
            report["tracklets"] = len(tracklets)
        except Exception as e:
            print(f"  Tracklet linking failed: {e}")
            report["errors"].append(f"Tracklet linking: {str(e)}")
    else:
        report["tracklets"] = 0

    # Step 6: Gate crossing inference
    print(f"\nStep 6: Gate crossing inference...")
    gate_line = load_gate_line(gate_path)

    # Use fallback scene crossings for same-day data
    if len(daily_detections) > 0:
        try:
            crossings = fallback_scene_crossings(daily_detections, gate_line)
            save_crossings(crossings, str(date_dir / "crossings"))
            # Count crossings (fallback format has estimated_gc_in/out)
            if len(crossings) > 0:
                report["crossings"] = {
                    "in": int(crossings["estimated_gc_in"].iloc[0]) if "estimated_gc_in" in crossings.columns else 0,
                    "out": int(crossings["estimated_gc_out"].iloc[0]) if "estimated_gc_out" in crossings.columns else 0
                }
        except Exception as e:
            print(f"  Gate crossing inference failed: {e}")
            report["errors"].append(f"Gate crossing: {str(e)}")

    # Step 7: Finalize report
    report["status"] = "completed" if not report["errors"] else "completed_with_errors"
    report["total_time_s"] = round(time.time() - start_time, 2)

    _write_report(report, qa_dir)

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"Scenes processed: {report['scenes_processed']}")
    print(f"Total detections: {report['total_detections']}")
    print(f"Coverage: {coverage['coverage_score']:.2%}")
    print(f"Total time: {report['total_time_s']:.1f}s")
    if report["errors"]:
        print(f"Errors: {len(report['errors'])}")
    print(f"{'='*60}\n")

    if return_detections:
        report["detections_df"] = daily_detections

    return report


def _write_report(report: dict, qa_dir: Path):
    """Write run report to JSON file."""
    report_file = qa_dir / "run_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Run report saved to {report_file}")


def run_pipeline(
    start_date: date,
    end_date: date,
    aoi_path: str = "configs/aoi_hormuz.geojson",
    gate_path: str = "configs/gate_line.geojson",
    output_base: str = "outputs",
    ais_path: Optional[str] = None
):
    """
    Run full pipeline for date range with multi-day tracklet linking.

    This function:
    1. Processes each day independently (detection, daily tracklets)
    2. Collects all detections across days
    3. Runs global tracklet linking across all days
    4. Runs gate crossing inference on global tracklets

    Args:
        start_date: Start date
        end_date: End date
        aoi_path: Path to AOI GeoJSON
        gate_path: Path to gate line GeoJSON
        output_base: Base output directory
    """
    print(f"\n{'='*60}")
    print(f"Hormuz SAR Throughput Pipeline")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"{'='*60}\n")

    current_date = start_date
    reports = []
    all_detections = []

    # Step 1: Process each day and collect detections
    while current_date <= end_date:
        report = run_single_day(
            target_date=current_date,
            aoi_path=aoi_path,
            gate_path=gate_path,
            output_base=output_base,
            return_detections=True  # Collect detections for global linking
        )
        reports.append(report)

        # Collect detections if available
        if "detections_df" in report and len(report["detections_df"]) > 0:
            all_detections.append(report["detections_df"])
            print(f"  Collected {len(report['detections_df'])} detections from {current_date}")

        current_date = date(
            current_date.year,
            current_date.month,
            current_date.day + 1
        )

    # Step 2: Global tracklet linking across all days
    print(f"\n{'='*60}")
    print("Step 2: Global Tracklet Linking (Multi-Day)")
    print(f"{'='*60}\n")

    if all_detections:
        combined_detections = pd.concat(all_detections, ignore_index=True)
        print(f"Total detections across all days: {len(combined_detections)}")

        # Run global tracklet linking with parameters suitable for multi-day tracking
        # Ships can move significant distances over multiple days
        try:
            global_tracklets = link_detections(
                combined_detections,
                max_distance_km=50.0,  # 50km - allows for multi-day ship movement
                max_time_gap_hours=72.0,  # 72 hours - up to 3 days between observations
                min_tracklet_length=2
            )

            # Save global tracklets
            global_output_dir = Path(output_base) / "global_tracklets"
            global_output_dir.mkdir(parents=True, exist_ok=True)
            save_tracklets(global_tracklets, str(global_output_dir))
            print(f"Saved {len(global_tracklets)} global tracklets to {global_output_dir}")

            # Step 3: Gate crossing inference on global tracklets
            print(f"\n{'='*60}")
            print("Step 3: Gate Crossing Inference (Global Tracklets)")
            print(f"{'='*60}\n")

            gate_line = load_gate_line(gate_path)
            global_crossings = fallback_scene_crossings(combined_detections, gate_line)
            save_crossings(global_crossings, str(global_output_dir / "crossings"))
            print(f"Saved global crossings to {global_output_dir / 'crossings'}")

        except Exception as e:
            print(f"Global tracklet linking failed: {e}")
    else:
        print("No detections collected across days.")

    # Step 4: Calibration (if AIS data provided)
    calibration_result = None
    if ais_path:
        print(f"\n{'='*60}")
        print("Step 4: AIS Calibration")
        print(f"{'='*60}\n")

        try:
            calibration_result = run_calibration_step(
                output_base=output_base,
                ais_path=ais_path,
                gate_path=gate_path,
                start_date=start_date,
                end_date=end_date
            )
        except Exception as e:
            print(f"Calibration failed: {e}")

    # Summary
    total_detections = sum(r.get("total_detections", 0) for r in reports)
    total_scenes = sum(r.get("scenes_processed", 0) for r in reports)

    print(f"\n{'='*60}")
    print("Multi-day pipeline complete!")
    print(f"Days processed: {len(reports)}")
    print(f"Total scenes: {total_scenes}")
    print(f"Total detections: {total_detections}")
    if all_detections:
        print(f"Combined detections: {len(pd.concat(all_detections, ignore_index=True))}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Hormuz SAR Throughput Pipeline")
    parser.add_argument("--date", type=str, help="Single date (YYYY-MM-DD)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--aoi", type=str, default="configs/aoi_hormuz.geojson")
    parser.add_argument("--gate", type=str, default="configs/gate_line.geojson")
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--ais", type=str, help="Path to AIS data Parquet for calibration")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run calibration step only (requires --ais)")

    args = parser.parse_args()

    if args.calibrate:
        # Run calibration only
        if not args.ais:
            parser.error("--ais required for calibration")
        start = date.fromisoformat(args.start) if args.start else None
        end = date.fromisoformat(args.end) if args.end else None
        run_calibration_step(args.output, args.ais, args.gate, start, end)
    elif args.date:
        # Single day
        d = date.fromisoformat(args.date)
        run_single_day(d, args.aoi, args.gate, args.output)
    elif args.start and args.end:
        # Date range
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        run_pipeline(start, end, args.aoi, args.gate, args.output, args.ais)
    else:
        parser.error("Either --date or both --start and --end required")


if __name__ == "__main__":
    main()
