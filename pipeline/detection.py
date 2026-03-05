"""
Ship Detection (Raster Vision)

Detects ships in SAR imagery using CV models.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

def detect_ships_baseline(
    ds,
    threshold: float = 0.5,
    min_size: int = 10,
    max_size: int = 500
) -> gpd.GeoDataFrame:
    """
    Baseline ship detection using thresholding.

    This is a simple baseline. For production, use Raster Vision
    with a fine-tuned model.

    Args:
        ds: xarray Dataset
        threshold: Detection threshold
        min_size: Minimum bbox size in pixels
        max_size: Maximum bbox size in pixels

    Returns:
        GeoDataFrame with detections
    """
    # Placeholder for actual detection
    # In production, this would use Raster Vision inference

    detections = []

    # Mock detections for demonstration
    # Real implementation would:
    # 1. Tile the scene
    # 2. Run model inference
    # 3. Post-process (NMS, filtering)
    # 4. Convert to GeoDataFrame

    print("Running baseline ship detection...")

    return gpd.GeoDataFrame(detections, crs="EPSG:4326")

def filter_detections(
    gdf: gpd.GeoDataFrame,
    water_mask_gdf: Optional[gpd.GeoDataFrame] = None,
    min_score: float = 0.3,
    min_area_km2: float = 0.001,
    max_area_km2: float = 1.0
) -> gpd.GeoDataFrame:
    """
    Filter detections based on score, area, and location.

    Args:
        gdf: Detections GeoDataFrame
        water_mask_gdf: Optional water mask to filter land detections
        min_score: Minimum detection score
        min_area_km2: Minimum bbox area in km²
        max_area_km2: Maximum bbox area in km²

    Returns:
        Filtered GeoDataFrame
    """
    if len(gdf) == 0:
        return gdf

    # Filter by score
    gdf = gdf[gdf["score"] >= min_score]

    # Filter by area
    gdf["area_km2"] = gdf.geometry.area * 111**2  # Rough km² conversion
    gdf = gdf[(gdf["area_km2"] >= min_area_km2) & (gdf["area_km2"] <= max_area_km2)]

    # Filter by water mask if provided
    if water_mask_gdf is not None:
        # Keep only detections within water mask
        gdf = gpd.sjoin(gdf, water_mask_gdf, how="inner", predicate="within")

    return gdf

def detections_to_parquet(
    gdf: gpd.GeoDataFrame,
    scene_id: str,
    datetime_str: str,
    output_path: str = "outputs/detections"
) -> pd.DataFrame:
    """
    Save detections to parquet.

    Args:
        gdf: Detections GeoDataFrame
        scene_id: Scene identifier
        datetime_str: Acquisition datetime
        output_path: Output directory

    Returns:
        DataFrame with detection records
    """
    records = []

    for _, row in gdf.iterrows():
        record = {
            "scene_id": scene_id,
            "datetime": datetime_str,
            "bbox_geom_wkt": row.geometry.wkt,
            "score": row.get("score", 0.5),
            "centroid_lon": row.geometry.centroid.x,
            "centroid_lat": row.geometry.centroid.y,
            "area_km2": row.get("area_km2", 0.0)
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Save to parquet
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime_str[:10] if datetime_str else "unknown"
    output_file = output_dir / f"{date_str}_{scene_id}.parquet"

    df.to_parquet(output_file, index=False)
    print(f"Saved {len(df)} detections to {output_file}")

    return df

def run_detection_pipeline(
    ds,
    scene_id: str,
    datetime_str: str,
    output_path: str = "outputs/detections"
) -> pd.DataFrame:
    """
    Full detection pipeline for a single scene.

    Args:
        ds: xarray Dataset
        scene_id: Scene identifier
        datetime_str: Acquisition datetime
        output_path: Output directory

    Returns:
        Detection DataFrame
    """
    # Detect ships
    detections_gdf = detect_ships_baseline(ds)

    # Filter detections
    filtered_gdf = filter_detections(detections_gdf)

    # Save to parquet
    df = detections_to_parquet(filtered_gdf, scene_id, datetime_str, output_path)

    return df

if __name__ == "__main__":
    print("Ship Detection module")
    print("Usage: python -m pipeline.detection --scene <scene_path>")
