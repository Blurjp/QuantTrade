"""
Main Pipeline Runner

Orchestrates the full SAR throughput pipeline.
"""

import argparse
from datetime import date, datetime
from pathlib import Path

from pipeline.manifest import run_manifest_builder
from pipeline.loader import load_manifest, load_scenes
from pipeline.preprocess import preprocess_scene
from pipeline.detection import run_detection_pipeline
from pipeline.tracking import link_detections, save_tracklets
from pipeline.crossings import load_gate_line, infer_crossings, save_crossings
from pipeline.metrics import aggregate_daily_metrics, save_metrics, load_aoi
from pipeline.calibration import (
    fit_bias_model,
    apply_bias_correction,
    save_calibration_report
)

def run_pipeline(
    start_date: date,
    end_date: date,
    aoi_path: str = "configs/aoi_hormuz.geojson",
    gate_path: str = "configs/gate_line.geojson",
    output_base: str = "outputs"
):
    """
    Run full pipeline for date range.

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

    # Step 1: Build manifest
    print("Step 1: Building manifest...")
    manifest = run_manifest_builder(
        start_date=start_date,
        end_date=end_date,
        aoi_path=aoi_path,
        output_path=f"{output_base}/manifests"
    )

    if len(manifest) == 0:
        print("No scenes found. Exiting.")
        return

    # Step 2: Load scenes
    print("\nStep 2: Loading scenes...")
    scenes = load_scenes(manifest, aoi_path=aoi_path)

    # Step 3-4: Preprocess + Detect (mock for now)
    print("\nStep 3-4: Preprocessing and detection...")
    # In production, would iterate through scenes
    # For now, create mock detections

    # Step 5: Tracklet linking
    print("\nStep 5: Tracklet linking...")
    # Mock detections for demonstration
    # tracklets = link_detections(detections_df)
    # save_tracklets(tracklets, f"{output_base}/tracklets")

    # Step 6: Gate crossing inference
    print("\nStep 6: Gate crossing inference...")
    gate_line = load_gate_line(gate_path)
    # crossings = infer_crossings(tracklets, gate_line)
    # save_crossings(crossings, f"{output_base}/crossings")

    # Step 7: Daily metrics
    print("\nStep 7: Aggregating daily metrics...")
    # metrics = aggregate_daily_metrics(crossings, manifest, load_aoi(aoi_path))
    # save_metrics(metrics, f"{output_base}/metrics")

    # Step 8: Calibration (if AIS data available)
    print("\nStep 8: Calibration...")
    # coefficients, performance = fit_bias_model(metrics, ais_metrics)
    # corrected_metrics = apply_bias_correction(metrics, coefficients)
    # save_calibration_report(coefficients, performance, f"{output_base}/validation")

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"Manifest: {len(manifest)} scenes")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="Hormuz SAR Throughput Pipeline")
    parser.add_argument("--date", type=str, help="Single date (YYYY-MM-DD)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--aoi", type=str, default="configs/aoi_hormuz.geojson")
    parser.add_argument("--gate", type=str, default="configs/gate_line.geojson")
    parser.add_argument("--output", type=str, default="outputs")

    args = parser.parse_args()

    if args.date:
        # Single day
        d = date.fromisoformat(args.date)
        run_pipeline(d, d, args.aoi, args.gate, args.output)
    elif args.start and args.end:
        # Date range
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        run_pipeline(start, end, args.aoi, args.gate, args.output)
    else:
        parser.error("Either --date or both --start and --end required")

if __name__ == "__main__":
    main()
