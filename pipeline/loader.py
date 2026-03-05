"""
Scene Loader (ODC-STAC → xarray)

Loads Sentinel-1 scenes from STAC items into xarray.Dataset.
Handles Planetary Computer signing for asset access.
"""

import json
import time
from pathlib import Path
from typing import List, Optional, Tuple, Iterator

import pandas as pd
import xarray as xr
import numpy as np
from odc import stac
import planetary_computer as pc
from pystac import Item

from pipeline.manifest import load_stac_items


def load_manifest(manifest_path: str) -> pd.DataFrame:
    """Load manifest from parquet."""
    return pd.read_parquet(manifest_path)


def sign_items(items: List[Item]) -> List[Item]:
    """
    Sign STAC items for Planetary Computer access.

    This is CRITICAL - without signing, you'll get 403s when loading assets.

    Args:
        items: List of pystac Items

    Returns:
        List of signed Items with valid asset hrefs
    """
    print(f"Signing {len(items)} STAC items...")
    signed_items = []
    for item in items:
        signed_item = pc.sign(item)
        signed_items.append(signed_item)
    print(f"Signed {len(signed_items)} items")
    return signed_items


def load_scenes_from_items(
    items: List[Item],
    bands: List[str] = ["vh", "vv"],
    chunks: dict = {"x": 2048, "y": 2048},
    aoi_geom: Optional[dict] = None,
    sign: bool = True
) -> xr.Dataset:
    """
    Load Sentinel-1 scenes from STAC items into xarray.Dataset.

    Args:
        items: List of pystac Items (will be signed if sign=True)
        bands: Bands to load (vh, vv)
        chunks: Dask chunk sizes
        aoi_geom: Optional AOI geometry for spatial cropping
        sign: Whether to sign items (default True for Planetary Computer)

    Returns:
        xarray.Dataset with loaded scenes
    """
    if sign:
        items = sign_items(items)

    print(f"Loading {len(items)} scenes with odc-stac...")

    # Build bbox from AOI if provided
    bbox = None
    if aoi_geom:
        coords = aoi_geom.get("coordinates", [[]])[0]
        if coords:
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            bbox = [min(lons), min(lats), max(lons), max(lats)]

    # Load with odc-stac
    # Note: groupby="solar_day" can merge scenes from same day
    # For ship detection, we want individual scenes, so we don't group
    ds = stac.load(
        items,
        bands=bands,
        chunks=chunks,
        bbox=bbox,
        preserve_original_order=True
    )

    print(f"Loaded dataset with shape: {dict(ds.sizes)}")
    return ds


def load_scenes_from_manifest(
    items_path: str,
    bands: List[str] = ["vh", "vv"],
    chunks: dict = {"x": 2048, "y": 2048},
    aoi_path: Optional[str] = None
) -> Tuple[xr.Dataset, List[Item]]:
    """
    Load Sentinel-1 scenes from saved STAC items file.

    Args:
        items_path: Path to NDJSON file with STAC items
        bands: Bands to load (vh, vv)
        chunks: Dask chunk sizes
        aoi_path: Optional AOI GeoJSON for spatial filtering

    Returns:
        Tuple of (xarray.Dataset, list of signed Items)
    """
    # Load STAC items from file
    items = load_stac_items(items_path)

    # Load AOI if provided
    aoi_geom = None
    if aoi_path:
        with open(aoi_path) as f:
            aoi = json.load(f)
            aoi_geom = aoi["features"][0]["geometry"]

    # Load scenes
    ds = load_scenes_from_items(items, bands=bands, chunks=chunks, aoi_geom=aoi_geom)

    # Sign items for return (so caller has access to metadata)
    signed_items = sign_items(items)

    return ds, signed_items


def iter_scenes(
    items: List[Item],
    bands: List[str] = ["vh", "vv"],
    chunks: dict = {"x": 2048, "y": 2048},
    sign: bool = True
) -> Iterator[Tuple[xr.Dataset, Item]]:
    """
    Iterate over scenes one at a time (memory-efficient for large batches).

    Args:
        items: List of pystac Items
        bands: Bands to load
        chunks: Dask chunk sizes
        sign: Whether to sign items

    Yields:
        Tuple of (single-scene Dataset, signed Item)
    """
    if sign:
        items = sign_items(items)

    for item in items:
        ds = stac.load(
            [item],
            bands=bands,
            chunks=chunks,
            preserve_original_order=True
        )
        yield ds, item


def compute_coverage_score(
    items: List[Item],
    aoi_geom: dict
) -> dict:
    """
    Compute coverage metrics for the scene collection.

    Args:
        items: List of STAC items
        aoi_geom: AOI geometry dict

    Returns:
        Dict with coverage metrics
    """
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union

    # Build AOI polygon
    aoi_shape = shape(aoi_geom)
    aoi_area = aoi_shape.area

    # Build union of scene footprints intersecting AOI
    scene_footprints = []
    for item in items:
        footprint = shape(item.geometry)
        intersection = footprint.intersection(aoi_shape)
        if not intersection.is_empty:
            scene_footprints.append(intersection)

    if not scene_footprints:
        return {
            "coverage_score": 0.0,
            "num_scenes": len(items),
            "aoi_area_km2": aoi_area * 111**2  # rough km2 conversion
        }

    # Union of all footprints
    covered = unary_union(scene_footprints)
    coverage_score = covered.area / aoi_area if aoi_area > 0 else 0

    return {
        "coverage_score": round(min(coverage_score, 1.0), 4),
        "num_scenes": len(items),
        "covered_area_km2": covered.area * 111**2,
        "aoi_area_km2": aoi_area * 111**2
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=str, required=True, help="Path to STAC items NDJSON")
    parser.add_argument("--aoi", type=str, help="Optional AOI GeoJSON for filtering")

    args = parser.parse_args()

    ds, items = load_scenes_from_manifest(args.items, aoi_path=args.aoi)

    print(f"\nLoaded scenes:")
    print(f"Total items: {len(items)}")
    print(f"Dataset: {ds}")
