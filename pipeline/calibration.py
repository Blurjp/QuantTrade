"""
AIS Calibration & Bias Correction

Validates and calibrates satellite-derived metrics against AIS data.
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def load_ais_data(ais_path: str) -> pd.DataFrame:
    """
    Load AIS data.

    Expected columns:
    - datetime
    - latitude
    - longitude
    - mmsi (vessel ID)
    - speed
    - heading
    """
    return pd.read_parquet(ais_path)

def compute_ais_crossings(
    ais_df: pd.DataFrame,
    gate_coords: list
) -> pd.DataFrame:
    """
    Compute AIS-based gate crossings.

    Args:
        ais_df: AIS data
        gate_coords: Gate line coordinates [[lon1, lat1], [lon2, lat2]]

    Returns:
        AIS crossings DataFrame
    """
    if len(ais_df) == 0:
        return pd.DataFrame()

    from shapely.geometry import LineString, Point

    gate_line = LineString(gate_coords)

    crossings = []

    # Group by vessel
    for mmsi, vessel_df in ais_df.groupby("mmsi"):
        vessel_df = vessel_df.sort_values("datetime")

        # Track positions
        points = [Point(row["longitude"], row["latitude"])
                  for _, row in vessel_df.iterrows()]

        # Detect gate crossings
        for i in range(1, len(points)):
            # Check if segment crosses gate
            segment = LineString([points[i-1], points[i]])

            if segment.crosses(gate_line):
                crossing_point = segment.intersection(gate_line)

                # Determine direction based on heading or position change
                prev_row = vessel_df.iloc[i-1]
                heading = prev_row.get("heading", 0)

                # Simplified direction inference
                # (would need proper heading-based logic)
                direction = "in" if heading < 180 else "out"

                crossings.append({
                    "mmsi": mmsi,
                    "datetime": vessel_df.iloc[i]["datetime"],
                    "crossing_lon": crossing_point.x,
                    "crossing_lat": crossing_point.y,
                    "direction": direction
                })

    return pd.DataFrame(crossings)

def aggregate_ais_daily(crossings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate AIS crossings to daily counts.

    Args:
        crossings_df: AIS crossings

    Returns:
        Daily AIS metrics
    """
    if len(crossings_df) == 0:
        return pd.DataFrame()

    crossings_df["date"] = pd.to_datetime(crossings_df["datetime"]).dt.date

    daily = []

    for date_val, group in crossings_df.groupby("date"):
        gc_in = len(group[group["direction"] == "in"])
        gc_out = len(group[group["direction"] == "out"])

        daily.append({
            "date": date_val.isoformat(),
            "ais_gc_in": gc_in,
            "ais_gc_out": gc_out,
            "ais_gc_total": gc_in + gc_out
        })

    return pd.DataFrame(daily)

def fit_bias_model(
    sat_metrics_df: pd.DataFrame,
    ais_metrics_df: pd.DataFrame
) -> Tuple[dict, dict]:
    """
    Fit bias correction model.

    Args:
        sat_metrics_df: Satellite-derived metrics
        ais_metrics_df: AIS-derived metrics

    Returns:
        Model coefficients and performance metrics
    """
    # Merge on date
    merged = pd.merge(sat_metrics_df, ais_metrics_df, on="date", how="inner")

    if len(merged) < 3:
        print("Warning: Insufficient data for calibration (need at least 3 days)")
        return {}, {}

    # Features: satellite metrics + coverage + metadata
    X = merged[["gc_total", "coverage_score"]].values
    y = merged["ais_gc_total"].values

    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X)

    # Performance
    performance = {
        "r2": r2_score(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
        "n_samples": len(merged)
    }

    # Model coefficients
    coefficients = {
        "intercept": float(model.intercept_),
        "gc_total_coef": float(model.coef_[0]),
        "coverage_coef": float(model.coef_[1])
    }

    return coefficients, performance

def apply_bias_correction(
    metrics_df: pd.DataFrame,
    coefficients: dict
) -> pd.DataFrame:
    """
    Apply bias correction to metrics.

    Args:
        metrics_df: Original metrics
        coefficients: Bias model coefficients

    Returns:
        Corrected metrics
    """
    if not coefficients:
        return metrics_df

    df = metrics_df.copy()

    # Apply correction
    df["throughput_index_corrected"] = (
        coefficients.get("intercept", 0) +
        coefficients.get("gc_total_coef", 1) * df["gc_total"] +
        coefficients.get("coverage_coef", 0) * df["coverage_score"]
    )

    # Normalize
    max_corrected = df["throughput_index_corrected"].max()
    if max_corrected > 0:
        df["throughput_index_corrected"] = df["throughput_index_corrected"] / max_corrected

    # Add bias factor
    df["bias_factor"] = df["throughput_index_corrected"] / df["throughput_index_total"].replace(0, 1)

    return df

def save_calibration_report(
    coefficients: dict,
    performance: dict,
    output_path: str = "outputs/validation"
) -> str:
    """
    Save calibration report.

    Args:
        coefficients: Model coefficients
        performance: Performance metrics
        output_path: Output directory

    Returns:
        Report file path
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": {
            "type": "linear_regression",
            "coefficients": coefficients
        },
        "performance": performance
    }

    output_file = output_dir / "calibration_report.json"

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved calibration report to {output_file}")

    return str(output_file)

def run_calibration_workflow(
    sat_metrics_path: str,
    ais_data_path: str,
    gate_coords: list,
    output_path: str = "outputs/calibration",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> dict:
    """
    Run full calibration workflow.

    Args:
        sat_metrics_path: Path to satellite metrics Parquet
        ais_data_path: Path to AIS Parquet data
        gate_coords: Gate line coordinates
        output_path: Output directory for calibration report
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Calibration results dict
    """
    from pipeline.ais import load_ais_data, prepare_ais_for_calibration

    print("=" * 60)
    print("AIS Calibration Workflow")
    print("=" * 60)

    # Step 1: Load satellite metrics
    print("\nStep 1: Loading satellite metrics...")
    sat_metrics = pd.read_parquet(sat_metrics_path)
    print(f"  Loaded {len(sat_metrics)} days of satellite metrics")

    # Ensure date column is string format for merging
    if 'date' in sat_metrics.columns:
        sat_metrics['date'] = pd.to_datetime(sat_metrics['date']).dt.strftime('%Y-%m-%d')

    # Step 2: Load and prepare AIS data
    print("\nStep 2: Loading AIS data...")

    # Get AOI bounds from gate coords
    lons = [c[0] for c in gate_coords]
    lats = [c[1] for c in gate_coords]
    aoi_bounds = (min(lons) - 1, min(lats) - 1, max(lons) + 1, max(lats) + 1)

    ais_df = load_ais_data(
        ais_data_path,
        start_date=start_date,
        end_date=end_date,
        aoi_bounds=aoi_bounds
    )

    # Step 3: Compute AIS gate crossings
    print("\nStep 3: Computing AIS gate crossings...")
    ais_metrics = prepare_ais_for_calibration(ais_df, gate_coords)
    print(f"  Computed crossings for {len(ais_metrics)} days")

    # Step 4: Fit bias model
    print("\nStep 4: Fitting bias correction model...")
    coefficients, performance = fit_bias_model(sat_metrics, ais_metrics)

    if coefficients:
        print(f"  R² = {performance['r2']:.3f}")
        print(f"  MAE = {performance['mae']:.1f} crossings")
        print(f"  Model: y = {coefficients['intercept']:.2f} + "
              f"{coefficients['gc_total_coef']:.3f}*gc_total + "
              f"{coefficients['coverage_coef']:.3f}*coverage")
    else:
        print("  Warning: Could not fit bias model (insufficient data)")

    # Step 5: Apply correction and save
    print("\nStep 5: Applying bias correction...")
    if coefficients:
        corrected_metrics = apply_bias_correction(sat_metrics, coefficients)
        corrected_path = Path(output_path) / "corrected_metrics.parquet"
        corrected_path.parent.mkdir(parents=True, exist_ok=True)
        corrected_metrics.to_parquet(corrected_path, index=False)
        print(f"  Saved corrected metrics to {corrected_path}")

    # Step 6: Save calibration report
    print("\nStep 6: Saving calibration report...")
    report_path = save_calibration_report(coefficients, performance, output_path)

    print("\n" + "=" * 60)
    print("Calibration complete!")
    print("=" * 60)

    return {
        "coefficients": coefficients,
        "performance": performance,
        "report_path": report_path
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="AIS Calibration & Bias Correction")
    parser.add_argument("--sat", type=str, required=True,
                        help="Path to satellite metrics Parquet")
    parser.add_argument("--ais", type=str, required=True,
                        help="Path to AIS data Parquet")
    parser.add_argument("--gate", type=str, default="configs/gate_line.geojson",
                        help="Path to gate line GeoJSON")
    parser.add_argument("--output", type=str, default="outputs/calibration",
                        help="Output directory")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    # Load gate line
    with open(args.gate) as f:
        gate_data = json.load(f)

    gate_coords = gate_data['features'][0]['geometry']['coordinates']

    # Parse dates
    start_date = date.fromisoformat(args.start) if args.start else None
    end_date = date.fromisoformat(args.end) if args.end else None

    # Run calibration
    run_calibration_workflow(
        sat_metrics_path=args.sat,
        ais_data_path=args.ais,
        gate_coords=gate_coords,
        output_path=args.output,
        start_date=start_date,
        end_date=end_date
    )
