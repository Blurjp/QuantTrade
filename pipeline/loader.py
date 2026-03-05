"""
Scene Loader (ODC-STAC → xarray)

Loads Sentinel-1 scenes from STAC items into xarray.Dataset.
"""

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import xarray as xr
import numpy as np
from odc import stac

def load_manifest(manifest_path: str) -> pd.DataFrame:
    """Load manifest from parquet."""
    return pd.read_parquet(manifest_path)

def load_scenes(
    manifest: pd.DataFrame,
    bands: List[str] = ["vh", "vv"],
    chunks: dict = {"x": 2048, "y": 2048},
    aoi_path: Optional[str] = None
) -> xr.Dataset:
    """
    Load Sentinel-1 scenes from STAC into xarray.Dataset.

    Args:
        manifest: Manifest DataFrame with STAC items
        bands: Bands to load (vh, vv)
        chunks: Dask chunk sizes
        aoi_path: Optional AOI for spatial filtering

    Returns:
        xarray.Dataset with loaded scenes
    """
    # Reconstruct STAC items from manifest
    # For now, use mock data structure
    # In production, would fetch full STAC items from API

    print(f"Loading {len(manifest)} scenes...")

    # Mock implementation for now
    # Real implementation would use:
    # items = [reconstruct_stac_item(row) for _, row in manifest.iterrows()]
    # ds = stac.load(items, bands=bands, chunks=chunks)

    # Create mock dataset
    scenes = []

    for _, row in manifest.iterrows():
        scene_data = {
            "item_id": row["item_id"],
            "datetime": row["datetime"],
            "geometry": json.loads(row["geometry"])
        }
        scenes.append(scene_data)

    print(f"Loaded {len(scenes)} scene metadata")

    # Return manifest for now
    # Real implementation would return xarray.Dataset
    return manifest

def load_single_scene(
    item_id: str,
    stac_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
    bands: List[str] = ["vh", "vv"]
) -> xr.Dataset:
    """
    Load a single Sentinel-1 scene by item ID.

    Args:
        item_id: STAC item ID
        stac_url: STAC API endpoint
        bands: Bands to load

    Returns:
        xarray.Dataset
    """
    from pystac_client import Client

    client = Client.open(stac_url)

    # Search for specific item
    search = client.search(
        collections=["sentinel-1-grd"],
        ids=[item_id]
    )

    items = list(search.items())

    if not items:
        raise ValueError(f"Item not found: {item_id}")

    # Load with ODC-STAC
    ds = stac.load(items, bands=bands, chunks={"x": 2048, "y": 2048})

    return ds

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest parquet")
    parser.add_argument("--aoi", type=str, help="Optional AOI GeoJSON for filtering")

    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    scenes = load_scenes(manifest, aoi_path=args.aoi)

    print(f"\nLoaded scenes:")
    print(f"Total: {len(scenes)}")
