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

    if len(merged) < 10:
        print("Warning: Insufficient data for calibration")
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

if __name__ == "__main__":
    print("AIS Calibration module")
    print("Usage: python -m pipeline.calibration --sat <sat_metrics> --ais <ais_data>")
