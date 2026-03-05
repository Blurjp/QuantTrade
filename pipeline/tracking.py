"""
Tracklet Linking

Links detections across scenes to form tracklets.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point

def load_detections(detection_path: str) -> pd.DataFrame:
    """Load detections from parquet."""
    return pd.read_parquet(detection_path)

def compute_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Compute distance in km between two points (Haversine).

    Args:
        lat1, lon1: First point
        lat2, lon2: Second point

    Returns:
        Distance in km
    """
    R = 6371  # Earth radius in km

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)

    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c

def link_detections(
    detections_df: pd.DataFrame,
    max_distance_km: float = 50.0,
    max_time_gap_hours: float = 24.0,
    min_tracklet_length: int = 2
) -> pd.DataFrame:
    """
    Link detections into tracklets using nearest-neighbor heuristic.

    Args:
        detections_df: Detections DataFrame
        max_distance_km: Maximum distance for linking
        max_time_gap_hours: Maximum time gap for linking
        min_tracklet_length: Minimum detections per tracklet

    Returns:
        Tracklets DataFrame
    """
    if len(detections_df) == 0:
        return pd.DataFrame()

    # Sort by time
    df = detections_df.sort_values("datetime").reset_index(drop=True)

    # Assign tracklet IDs
    tracklet_id = 0
    tracklet_ids = []

    for i, row in df.iterrows():
        if i == 0:
            tracklet_ids.append(tracklet_id)
            continue

        # Find nearest previous detection
        prev_rows = df.iloc[:i]

        # Filter by time gap
        current_time = pd.to_datetime(row["datetime"])
        prev_times = pd.to_datetime(prev_rows["datetime"])
        time_ok = (current_time - prev_times) < timedelta(hours=max_time_gap_hours)

        if not time_ok.any():
            tracklet_id += 1
            tracklet_ids.append(tracklet_id)
            continue

        # Find nearest within time window
        candidates = prev_rows[time_ok]
        distances = [
            compute_distance(row["centroid_lat"], row["centroid_lon"],
                           c["centroid_lat"], c["centroid_lon"])
            for _, c in candidates.iterrows()
        ]

        min_dist = min(distances) if distances else float("inf")

        if min_dist <= max_distance_km:
            # Link to same tracklet as nearest neighbor
            nearest_idx = candidates.index[distances.index(min_dist)]
            tracklet_ids.append(tracklet_ids[nearest_idx])
        else:
            tracklet_id += 1
            tracklet_ids.append(tracklet_id)

    df["tracklet_id"] = tracklet_ids

    # Filter short tracklets
    tracklet_counts = df["tracklet_id"].value_counts()
    valid_tracklets = tracklet_counts[tracklet_counts >= min_tracklet_length].index
    df = df[df["tracklet_id"].isin(valid_tracklets)]

    return df

def save_tracklets(
    tracklets_df: pd.DataFrame,
    output_path: str = "outputs/tracklets"
) -> str:
    """
    Save tracklets to parquet.

    Args:
        tracklets_df: Tracklets DataFrame
        output_path: Output directory

    Returns:
        Output file path
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(tracklets_df) == 0:
        return ""

    date_str = tracklets_df.iloc[0]["datetime"][:10]
    output_file = output_dir / f"{date_str}.parquet"

    tracklets_df.to_parquet(output_file, index=False)
    print(f"Saved {len(tracklets_df)} tracklet detections to {output_file}")

    return str(output_file)

if __name__ == "__main__":
    print("Tracklet Linking module")
    print("Usage: python -m pipeline.tracking --detections <detections_path>")
