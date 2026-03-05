"""
Ship Detection (CFAR Baseline)

Detects ships in SAR imagery using Constant False Alarm Rate (CFAR) algorithm.
This is a robust baseline that doesn't require training data.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from scipy.ndimage import label, binary_dilation, binary_erosion
from shapely.geometry import box
import rasterio
from rasterio.transform import from_bounds


def cfar_detector(
    data: np.ndarray,
    guard_size: int = 5,
    window_size: int = 21,
    k: float = 3.0,
    min_target_size: int = 2,
    max_target_size: int = 100
) -> np.ndarray:
    """
    CFAR (Constant False Alarm Rate) detector for SAR imagery.

    Simplified implementation using global statistics with adaptive local threshold.

    Args:
        data: 2D numpy array (log-scaled SAR intensity)
        guard_size: Size of guard window (not used in simplified version)
        window_size: Size of local window for adaptive threshold
        k: Threshold multiplier (higher = fewer false alarms)
        min_target_size: Minimum target size in pixels
        max_target_size: Maximum target size in pixels

    Returns:
        Binary mask of detected targets
    """
    # Handle NaN values - replace with a low value
    data = np.nan_to_num(data, nan=np.nanmin(data) if np.any(~np.isnan(data)) else -50)

    # Global statistics for initial threshold
    global_median = np.median(data)
    global_std = np.std(data)
    global_mad = np.median(np.abs(data - global_median))

    # Use MAD-based std estimate (more robust)
    robust_std = 1.4826 * global_mad if global_mad > 0 else global_std

    # Initial threshold
    threshold = global_median + k * robust_std

    print(f"  CFAR stats: median={global_median:.2f}, std={global_std:.2f}, threshold={threshold:.2f}")

    # Create initial detection mask
    mask = data > threshold

    # Morphological cleanup - light, to reduce noise while preserving detections
    mask = binary_dilation(mask, iterations=1)

    return mask


def detect_ships_cfar(
    ds: xr.Dataset,
    band: str = "vv",
    guard_size: int = 5,
    window_size: int = 21,
    k: float = 3.5,
    min_area_px: int = 4,
    max_area_px: int = 500,
    log_scale: bool = True
) -> gpd.GeoDataFrame:
    """
    Detect ships using CFAR algorithm on SAR data.

    Args:
        ds: xarray Dataset with SAR bands
        band: Band to use for detection (vv or vh)
        guard_size: CFAR guard window size
        window_size: CFAR background window size
        k: CFAR threshold multiplier
        min_area_px: Minimum detection area in pixels
        max_area_px: Maximum detection area in pixels
        log_scale: Whether to log-scale the data

    Returns:
        GeoDataFrame with detected ship bboxes
    """
    print(f"Running CFAR detection on {band} band...")

    # Extract data
    if band not in ds:
        print(f"Warning: {band} band not found in dataset")
        return gpd.GeoDataFrame(crs="EPSG:4326")

    data = ds[band].values

    # Handle multi-dimensional data (time, etc.)
    if data.ndim > 2:
        # Take first time slice if multiple
        data = data[0]

    # Log scale if needed
    if log_scale:
        # Avoid log(0) by clipping to small positive value
        data = np.clip(data, 1e-10, None)
        data = 10 * np.log10(data)

    # Mask invalid values
    valid_mask = np.isfinite(data)
    data = np.where(valid_mask, data, np.nanmin(data[valid_mask]) if valid_mask.any() else -50)

    # Run CFAR
    detection_mask = cfar_detector(
        data,
        guard_size=guard_size,
        window_size=window_size,
        k=k
    )

    # Connected components
    labeled_array, num_features = label(detection_mask)

    print(f"Found {num_features} connected components")

    if num_features == 0:
        return gpd.GeoDataFrame(columns=["geometry", "score"], geometry="geometry", crs="EPSG:4326")

    # Extract bboxes
    detections = []

    # Get spatial coordinates (handle different naming conventions)
    if 'longitude' in ds[band].coords and 'latitude' in ds[band].coords:
        x_coords = ds[band].coords['longitude'].values
        y_coords = ds[band].coords['latitude'].values
    elif 'x' in ds[band].coords and 'y' in ds[band].coords:
        x_coords = ds[band].coords['x'].values
        y_coords = ds[band].coords['y'].values
    else:
        print("Warning: Cannot find spatial coordinates")
        return gpd.GeoDataFrame(columns=["geometry", "score"], geometry="geometry", crs="EPSG:4326")

    # Get pixel sizes (assume regular grid)
    if len(x_coords) > 1 and len(y_coords) > 1:
        pixel_width = abs(float(x_coords[1] - x_coords[0]))
        pixel_height = abs(float(y_coords[1] - y_coords[0]))
    else:
        pixel_width = 0.0001  # fallback
        pixel_height = 0.0001

    for region_id in range(1, num_features + 1):
        region_mask = labeled_array == region_id

        # Count pixels
        area_px = np.sum(region_mask)

        # Filter by size
        if area_px < min_area_px or area_px > max_area_px:
            continue

        # Find bbox in pixel coordinates
        rows, cols = np.where(region_mask)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()

        # Convert to geo coordinates
        # Handle coordinate direction (y may be decreasing)
        if len(y_coords) > 1 and y_coords[0] > y_coords[-1]:
            # Y coordinates are decreasing (common for raster)
            min_y = float(y_coords[max_row])
            max_y = float(y_coords[min_row])
        else:
            min_y = float(y_coords[min_row])
            max_y = float(y_coords[max_row])

        min_x = float(x_coords[min_col])
        max_x = float(x_coords[max_col])

        # Create bbox geometry
        bbox_geom = box(min_x, min_y, max_x, max_y)

        # Compute centroid
        centroid_x = (min_x + max_x) / 2
        centroid_y = (min_y + max_y) / 2

        # Compute area in km² (rough approximation)
        area_km2 = abs((max_x - min_x) * (max_y - min_y) * 111**2)

        # Compute mean intensity in detection region
        mean_intensity = float(np.mean(data[region_mask]))

        detections.append({
            "geometry": bbox_geom,
            "score": min(1.0, max(0.0, (mean_intensity + 30) / 30)),  # Normalize score
            "centroid_lon": centroid_x,
            "centroid_lat": centroid_y,
            "area_px": int(area_px),
            "area_km2": round(area_km2, 4),
            "mean_intensity_db": round(mean_intensity, 2),
            "bbox_width_px": int(max_col - min_col + 1),
            "bbox_height_px": int(max_row - min_row + 1)
        })

    print(f"Filtered to {len(detections)} valid detections")

    if not detections:
        return gpd.GeoDataFrame(columns=["geometry", "score"], geometry="geometry", crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(detections, geometry="geometry", crs="EPSG:4326")
    return gdf


def detect_ships_baseline(
    ds,
    threshold: float = 0.5,
    min_size: int = 10,
    max_size: int = 500
) -> gpd.GeoDataFrame:
    """
    Baseline ship detection using CFAR algorithm.

    This is the primary detection method. For production ML-based
    detection, replace this function's implementation.

    Args:
        ds: xarray Dataset
        threshold: Detection threshold (maps to CFAR k parameter)
        min_size: Minimum bbox size in pixels
        max_size: Maximum bbox size in pixels

    Returns:
        GeoDataFrame with detections
    """
    # Map threshold to CFAR k parameter
    # threshold 0.3 -> k=2.5 (more sensitive), threshold 0.5 -> k=3.5, threshold 0.7 -> k=4.5 (less sensitive)
    k = 2.0 + (threshold * 3.0)

    return detect_ships_cfar(
        ds,
        band="vv",
        guard_size=5,
        window_size=21,
        k=k,
        min_area_px=min_size,
        max_area_px=max_size,
        log_scale=True
    )


def filter_detections(
    gdf: gpd.GeoDataFrame,
    water_mask_gdf: Optional[gpd.GeoDataFrame] = None,
    min_score: float = 0.2,
    min_area_km2: float = 0.0005,
    max_area_km2: float = 0.5,
    min_aspect_ratio: float = 0.1,
    max_aspect_ratio: float = 10.0
) -> gpd.GeoDataFrame:
    """
    Filter detections based on score, area, aspect ratio, and location.

    Args:
        gdf: Detections GeoDataFrame
        water_mask_gdf: Optional water mask to filter land detections
        min_score: Minimum detection score
        min_area_km2: Minimum bbox area in km²
        max_area_km2: Maximum bbox area in km²
        min_aspect_ratio: Minimum width/height ratio
        max_aspect_ratio: Maximum width/height ratio

    Returns:
        Filtered GeoDataFrame
    """
    if len(gdf) == 0:
        return gdf

    # Filter by score
    gdf = gdf[gdf["score"] >= min_score].copy()

    # Filter by area
    if "area_km2" in gdf.columns:
        gdf = gdf[(gdf["area_km2"] >= min_area_km2) & (gdf["area_km2"] <= max_area_km2)]

    # Filter by aspect ratio
    if "bbox_width_px" in gdf.columns and "bbox_height_px" in gdf.columns:
        gdf["aspect_ratio"] = gdf["bbox_width_px"] / gdf["bbox_height_px"].clip(lower=1)
        gdf = gdf[
            (gdf["aspect_ratio"] >= min_aspect_ratio) &
            (gdf["aspect_ratio"] <= max_aspect_ratio)
        ]

    # Filter by water mask if provided
    if water_mask_gdf is not None and len(water_mask_gdf) > 0:
        # Keep only detections within water mask
        try:
            gdf = gpd.sjoin(gdf, water_mask_gdf, how="inner", predicate="intersects")
        except Exception as e:
            print(f"Warning: Water mask filtering failed: {e}")

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

    for idx, row in gdf.iterrows():
        record = {
            "scene_id": scene_id,
            "datetime": datetime_str,
            "detection_id": f"{scene_id}_{idx}",
            "bbox_geom_wkt": row.geometry.wkt,
            "score": row.get("score", 0.5),
            "centroid_lon": row.geometry.centroid.x,
            "centroid_lat": row.geometry.centroid.y,
            "area_km2": row.get("area_km2", 0.0),
            "mean_intensity_db": row.get("mean_intensity_db", 0.0)
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Save to parquet
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime_str[:10] if datetime_str else "unknown"
    output_file = output_dir / f"{date_str}_{scene_id}.parquet"

    if len(df) > 0:
        df.to_parquet(output_file, index=False)
        print(f"Saved {len(df)} detections to {output_file}")
    else:
        print(f"No detections to save for {scene_id}")

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
    # Detect ships using CFAR
    detections_gdf = detect_ships_baseline(ds)

    # Filter detections
    filtered_gdf = filter_detections(detections_gdf)

    # Save to parquet
    df = detections_to_parquet(filtered_gdf, scene_id, datetime_str, output_path)

    return df


if __name__ == "__main__":
    print("Ship Detection module - CFAR Baseline")
    print("Usage: python -m pipeline.detection --scene <scene_path>")
    print("\nFor pipeline execution, use:")
    print("  python -m pipeline.run --date YYYY-MM-DD")
