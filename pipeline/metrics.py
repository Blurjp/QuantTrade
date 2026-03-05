"""
Daily Metrics Aggregation

Aggregates crossings into daily throughput metrics.
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import shape

def load_crossings(crossings_path: str) -> pd.DataFrame:
    """Load crossings from parquet."""
    return pd.read_parquet(crossings_path)

def load_aoi(aoi_path: str = "configs/aoi_hormuz.geojson") -> dict:
    """Load AOI geometry."""
    with open(aoi_path) as f:
        return json.load(f)

def compute_coverage_score(
    manifest_df: pd.DataFrame,
    aoi_geom: dict
) -> float:
    """
    Compute daily coverage score.

    Args:
        manifest_df: Manifest DataFrame for one day
        aoi_geom: AOI geometry

    Returns:
        Coverage score (0-1)
    """
    if len(manifest_df) == 0:
        return 0.0

    # Extract scene footprints
    scene_geoms = []
    for _, row in manifest_df.iterrows():
        geom = shape(json.loads(row["geometry"]))
        scene_geoms.append(geom)

    # Compute union of footprints
    from shapely.ops import unary_union
    union_geom = unary_union(scene_geoms)

    # Compute intersection with AOI
    aoi_shape = shape(aoi_geom["features"][0]["geometry"])

    if union_geom.is_empty:
        return 0.0

    intersection = union_geom.intersection(aoi_shape)

    # Coverage score
    coverage = intersection.area / aoi_shape.area

    return float(min(coverage, 1.0))

def aggregate_daily_metrics(
    crossings_df: pd.DataFrame,
    manifest_df: Optional[pd.DataFrame] = None,
    aoi_geom: Optional[dict] = None
) -> pd.DataFrame:
    """
    Aggregate crossings into daily metrics.

    Args:
        crossings_df: Crossings DataFrame
        manifest_df: Optional manifest for coverage scoring
        aoi_geom: Optional AOI geometry

    Returns:
        Daily metrics DataFrame
    """
    if len(crossings_df) == 0:
        return pd.DataFrame()

    # Parse datetime
    crossings_df["date"] = pd.to_datetime(crossings_df["datetime"]).dt.date

    # Group by date
    daily = []

    for date_val, group in crossings_df.groupby("date"):
        # Count crossings by direction
        gc_in = len(group[group["direction"] == "in"])
        gc_out = len(group[group["direction"] == "out"])

        # Size-weighted throughput (use bbox area as proxy)
        swt_in = group[group["direction"] == "in"]["area_km2"].sum() if "area_km2" in group.columns else gc_in
        swt_out = group[group["direction"] == "out"]["area_km2"].sum() if "area_km2" in group.columns else gc_out

        # Coverage score
        coverage_score = 1.0  # Default
        if manifest_df is not None and aoi_geom is not None:
            day_manifest = manifest_df[
                pd.to_datetime(manifest_df["datetime"]).dt.date == date_val
            ]
            coverage_score = compute_coverage_score(day_manifest, aoi_geom)

        daily_metrics = {
            "date": date_val.isoformat(),
            "gc_in": gc_in,
            "gc_out": gc_out,
            "gc_total": gc_in + gc_out,
            "swt_in": swt_in,
            "swt_out": swt_out,
            "swt_total": swt_in + swt_out,
            "coverage_score": coverage_score,
            "num_crossings": len(group)
        }

        daily.append(daily_metrics)

    df = pd.DataFrame(daily)

    # Compute throughput index (normalized)
    if len(df) > 0:
        max_gc = df["gc_total"].max()
        if max_gc > 0:
            df["throughput_index_total"] = df["gc_total"] / max_gc
            df["throughput_index_in"] = df["gc_in"] / max_gc
            df["throughput_index_out"] = df["gc_out"] / max_gc
        else:
            df["throughput_index_total"] = 0.0
            df["throughput_index_in"] = 0.0
            df["throughput_index_out"] = 0.0

    return df

def save_metrics(
    metrics_df: pd.DataFrame,
    output_path: str = "outputs/metrics"
) -> str:
    """
    Save metrics to parquet.

    Args:
        metrics_df: Metrics DataFrame
        output_path: Output directory

    Returns:
        Output file path
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(metrics_df) == 0:
        return ""

    output_file = output_dir / "daily.parquet"

    # Append or overwrite
    if output_file.exists():
        existing = pd.read_parquet(output_file)
        metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
        metrics_df = metrics_df.drop_duplicates(subset=["date"], keep="last")

    metrics_df.to_parquet(output_file, index=False)
    print(f"Saved {len(metrics_df)} daily metrics to {output_file}")

    return str(output_file)

if __name__ == "__main__":
    print("Daily Metrics Aggregation module")
    print("Usage: python -m pipeline.metrics --crossings <crossings_path>")
