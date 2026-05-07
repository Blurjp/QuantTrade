"""
Satellite Data Loader (ODC-STAC → xarray)

Loads satellite imagery from STAC items into xarray.Dataset.
Handles Planetary Computer signing for asset access.

NOTE: This module was moved from pipeline/loader.py during P0.3 refactoring.
The old import path `from pipeline.loader import ...` still works for backward compatibility.
"""
from __future__ import annotations

import functools
import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple, Iterator, Callable, TypeVar

import pandas as pd
import xarray as xr
import numpy as np
from odc import stac
import planetary_computer as pc
from pystac import Item

from data.loaders.manifest import load_stac_items

# Configure logging
logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        retryable_exceptions: Tuple of exception types to retry on

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after error: {e}. Waiting {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries} retries exhausted for {func.__name__}"
                        )
            raise last_exception
        return wrapper
    return decorator


def load_manifest(manifest_path: str) -> pd.DataFrame:
    """Load manifest from parquet."""
    return pd.read_parquet(manifest_path)


def sign_items(items: List[Item], max_retries: int = 3, base_delay: float = 1.0) -> List[Item]:
    """
    Sign STAC items for Planetary Computer access with retry logic.

    This is CRITICAL - without signing, you'll get 403s when loading assets.

    Args:
        items: List of pystac Items
        max_retries: Maximum retry attempts per item (default 3)
        base_delay: Base delay for exponential backoff in seconds (default 1.0)

    Returns:
        List of signed Items with valid asset hrefs
    """
    print(f"Signing {len(items)} STAC items...")
    signed_items = []
    failed_items = []

    for item in items:
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                signed_item = pc.sign(item)
                signed_items.append(signed_item)
                break
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                # Retry on 403, 5xx, or connection errors
                should_retry = (
                    '403' in error_str or
                    '500' in error_str or
                    '502' in error_str or
                    '503' in error_str or
                    '504' in error_str or
                    'connection' in error_str or
                    'timeout' in error_str
                )

                if should_retry and attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt), 30.0)
                    print(f"    Retry {attempt + 1}/{max_retries} for {item.id} after error: {e}. Waiting {delay:.1f}s...")
                    time.sleep(delay)
                elif attempt == max_retries:
                    print(f"    ERROR: Failed to sign {item.id} after {max_retries} retries: {e}")
                    failed_items.append((item.id, str(e)))

    print(f"Signed {len(signed_items)} items")
    if failed_items:
        print(f"Warning: {len(failed_items)} items failed to sign")
        for item_id, error in failed_items:
            logger.warning(f"Failed to sign item {item_id}: {error}")

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


__all__ = [
    "retry_with_backoff",
    "load_manifest",
    "sign_items",
    "load_scenes_from_items",
    "load_scenes_from_manifest",
    "iter_scenes",
    "compute_coverage_score",
]
