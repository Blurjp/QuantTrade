"""
Gate Crossing Inference

Detects crossings through the gate line from tracklets.
"""

import json
from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

def load_gate_line(gate_path: str = "configs/gate_line.geojson") -> LineString:
    """Load gate line from GeoJSON."""
    with open(gate_path) as f:
        data = json.load(f)

    coords = data["geometry"]["coordinates"]
    return LineString(coords)

def compute_side_of_gate(point: Point, gate_line: LineString) -> int:
    """
    Determine which side of the gate a point is on.

    Returns:
        1 or -1 (indicating side)
    """
    # Use cross product to determine side
    gate_coords = list(gate_line.coords)
    gate_vec = np.array([gate_coords[1][0] - gate_coords[0][0],
                        gate_coords[1][1] - gate_coords[0][1]])

    point_vec = np.array([point.x - gate_coords[0][0],
                         point.y - gate_coords[0][1]])

    cross = gate_vec[0] * point_vec[1] - gate_vec[1] * point_vec[0]

    return 1 if cross > 0 else -1

def infer_crossings(
    tracklets_df: pd.DataFrame,
    gate_line: LineString
) -> pd.DataFrame:
    """
    Infer gate crossings from tracklets.

    A crossing occurs when a tracklet moves from one side of the gate
    to the other.

    Args:
        tracklets_df: Tracklets DataFrame
        gate_line: Gate line geometry

    Returns:
        Crossings DataFrame
    """
    if len(tracklets_df) == 0:
        return pd.DataFrame()

    crossings = []

    for tracklet_id, group in tracklets_df.groupby("tracklet_id"):
        # Sort by time
        group = group.sort_values("datetime")

        # Get centroids
        points = [Point(row["centroid_lon"], row["centroid_lat"])
                  for _, row in group.iterrows()]

        # Compute sides
        sides = [compute_side_of_gate(p, gate_line) for p in points]

        # Detect side changes
        for i in range(1, len(sides)):
            if sides[i] != sides[i-1]:
                # Crossing detected
                row = group.iloc[i]

                crossing = {
                    "tracklet_id": tracklet_id,
                    "datetime": row["datetime"],
                    "crossing_lon": points[i].x,
                    "crossing_lat": points[i].y,
                    "direction": "in" if sides[i-1] == -1 else "out",
                    "prev_side": sides[i-1],
                    "curr_side": sides[i]
                }
                crossings.append(crossing)

    df = pd.DataFrame(crossings)
    return df

def fallback_scene_crossings(
    detections_df: pd.DataFrame,
    gate_line: LineString,
    distance_threshold_km: float = 5.0
) -> pd.DataFrame:
    """
    Fallback crossing estimation using scene-level spatial density.

    Used when tracklets are too sparse.

    Args:
        detections_df: Detections DataFrame
        gate_line: Gate line geometry
        distance_threshold_km: Distance threshold for "near gate"

    Returns:
        Estimated crossings DataFrame
    """
    if len(detections_df) == 0:
        return pd.DataFrame()

    # Find detections near gate
    gate_buffer = gate_line.buffer(distance_threshold_km / 111.0)  # Rough degree conversion

    near_gate = []
    for _, row in detections_df.iterrows():
        point = Point(row["centroid_lon"], row["centroid_lat"])
        if point.within(gate_buffer):
            near_gate.append(row)

    if not near_gate:
        return pd.DataFrame()

    near_gate_df = pd.DataFrame(near_gate)

    # Estimate crossings based on spatial density and time ordering
    # This is lower quality than tracklet-based crossings
    crossings = []

    for datetime_str, group in near_gate_df.groupby("datetime"):
        # Split by side of gate
        points = [Point(row["centroid_lon"], row["centroid_lat"])
                  for _, row in group.iterrows()]
        sides = [compute_side_of_gate(p, gate_line) for p in points]

        n_in = sides.count(-1)
        n_out = sides.count(1)

        crossings.append({
            "datetime": datetime_str,
            "estimated_gc_in": n_in,
            "estimated_gc_out": n_out,
            "quality": "low"  # Flag as lower quality
        })

    return pd.DataFrame(crossings)

def save_crossings(
    crossings_df: pd.DataFrame,
    output_path: str = "outputs/crossings"
) -> str:
    """
    Save crossings to parquet.

    Args:
        crossings_df: Crossings DataFrame
        output_path: Output directory

    Returns:
        Output file path
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(crossings_df) == 0:
        return ""

    date_str = crossings_df.iloc[0]["datetime"][:10]
    output_file = output_dir / f"{date_str}.parquet"

    crossings_df.to_parquet(output_file, index=False)
    print(f"Saved {len(crossings_df)} crossings to {output_file}")

    return str(output_file)

if __name__ == "__main__":
    print("Gate Crossing Inference module")
    print("Usage: python -m pipeline.crossings --tracklets <tracklets_path>")
