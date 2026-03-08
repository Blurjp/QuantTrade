"""
Daily Metrics Aggregation

Aggregates observation metadata and optional crossings into daily throughput metrics.
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from shapely.geometry import shape
from shapely.ops import unary_union


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
    """Compute daily coverage score."""
    if len(manifest_df) == 0:
        return 0.0

    aoi_shape = shape(aoi_geom)
    scene_geoms = []

    for _, row in manifest_df.iterrows():
        geometry_value = row.get("geometry")
        if not geometry_value:
            continue
        scene_geoms.append(shape(json.loads(geometry_value)))

    if not scene_geoms or aoi_shape.is_empty or aoi_shape.area == 0:
        return 0.0

    union_geom = unary_union(scene_geoms)
    intersection = union_geom.intersection(aoi_shape)
    return float(min(intersection.area / aoi_shape.area, 1.0))


def summarize_observation_day(
    manifest_df: pd.DataFrame,
    load_log_df: Optional[pd.DataFrame] = None,
    aoi_geom: Optional[dict] = None
) -> dict:
    """Summarize scene coverage and observation metadata for one day."""
    manifest_df = manifest_df.copy() if manifest_df is not None else pd.DataFrame()
    load_log_df = load_log_df.copy() if load_log_df is not None else pd.DataFrame()

    if len(manifest_df) == 0:
        return {
            "coverage_score": 0.0,
            "num_scenes": 0,
            "loaded_scenes": int(len(load_log_df)),
            "max_scene_gap_hours": None,
            "orbit_summary": "",
            "incidence_angle_min": None,
            "incidence_angle_max": None,
            "incidence_angle_mean": None,
        }

    datetimes = pd.to_datetime(manifest_df["datetime"], utc=True, errors="coerce").dropna().sort_values()
    max_gap_hours = None
    if len(datetimes) > 1:
        gaps = datetimes.diff().dropna().dt.total_seconds() / 3600.0
        if len(gaps) > 0:
            max_gap_hours = float(gaps.max())

    incidence = pd.to_numeric(manifest_df.get("incidence_angle"), errors="coerce").dropna()
    orbits = (
        manifest_df.get("orbit", pd.Series(dtype="object"))
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .sort_index()
    )
    orbit_summary = ",".join(f"{orbit}:{count}" for orbit, count in orbits.items())

    return {
        "coverage_score": compute_coverage_score(manifest_df, aoi_geom) if aoi_geom else 0.0,
        "num_scenes": int(len(manifest_df)),
        "loaded_scenes": int((load_log_df.get("status") == "loaded").sum()) if "status" in load_log_df else int(len(load_log_df)),
        "max_scene_gap_hours": round(max_gap_hours, 3) if max_gap_hours is not None else None,
        "orbit_summary": orbit_summary,
        "incidence_angle_min": float(incidence.min()) if len(incidence) > 0 else None,
        "incidence_angle_max": float(incidence.max()) if len(incidence) > 0 else None,
        "incidence_angle_mean": float(incidence.mean()) if len(incidence) > 0 else None,
    }


def aggregate_daily_metrics(
    target_date: str,
    manifest_df: pd.DataFrame,
    load_log_df: Optional[pd.DataFrame] = None,
    crossings_df: Optional[pd.DataFrame] = None,
    aoi_geom: Optional[dict] = None
) -> pd.DataFrame:
    """
    Aggregate one day's observations and optional crossings into daily metrics.
    """
    observation_summary = summarize_observation_day(
        manifest_df=manifest_df,
        load_log_df=load_log_df,
        aoi_geom=aoi_geom,
    )

    gc_in = 0
    gc_out = 0
    swt_in = 0.0
    swt_out = 0.0
    num_crossings = 0

    if crossings_df is not None and len(crossings_df) > 0:
        num_crossings = int(len(crossings_df))
        is_fallback = "estimated_gc_in" in crossings_df.columns and "direction" not in crossings_df.columns

        if is_fallback:
            gc_in = int(crossings_df["estimated_gc_in"].fillna(0).sum())
            gc_out = int(crossings_df["estimated_gc_out"].fillna(0).sum())
            if "area_km2" in crossings_df.columns:
                swt_in = float(crossings_df["area_km2"].fillna(0).sum())
                swt_out = swt_in
        else:
            gc_in = int((crossings_df["direction"] == "in").sum())
            gc_out = int((crossings_df["direction"] == "out").sum())
            if "area_km2" in crossings_df.columns:
                swt_in = float(crossings_df.loc[crossings_df["direction"] == "in", "area_km2"].fillna(0).sum())
                swt_out = float(crossings_df.loc[crossings_df["direction"] == "out", "area_km2"].fillna(0).sum())

    record = {
        "date": target_date,
        "gc_in": gc_in,
        "gc_out": gc_out,
        "gc_total": gc_in + gc_out,
        "swt_in": swt_in,
        "swt_out": swt_out,
        "swt_total": swt_in + swt_out,
        "coverage_score": observation_summary["coverage_score"],
        "num_scenes": observation_summary["num_scenes"],
        "loaded_scenes": observation_summary["loaded_scenes"],
        "max_scene_gap_hours": observation_summary["max_scene_gap_hours"],
        "orbit_summary": observation_summary["orbit_summary"],
        "incidence_angle_min": observation_summary["incidence_angle_min"],
        "incidence_angle_max": observation_summary["incidence_angle_max"],
        "incidence_angle_mean": observation_summary["incidence_angle_mean"],
        "num_crossings": num_crossings,
    }

    record["throughput_index_total"] = float(record["gc_total"] > 0)
    record["throughput_index_in"] = float(record["gc_in"] > 0)
    record["throughput_index_out"] = float(record["gc_out"] > 0)

    return pd.DataFrame([record])


def save_metrics(
    metrics_df: pd.DataFrame,
    output_path: str = "outputs/metrics"
) -> str:
    """Save metrics to parquet."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "daily.parquet"
    metrics_df.to_parquet(output_file, index=False)
    print(f"Saved {len(metrics_df)} daily metrics to {output_file}")
    return str(output_file)


if __name__ == "__main__":
    print("Daily Metrics Aggregation module")
    print("Usage: python -m pipeline.metrics --crossings <crossings_path>")
