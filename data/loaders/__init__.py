"""
Data loaders for various sources.

Loaders are responsible for fetching raw data from external sources:
- Satellite imagery (Sentinel, Landsat, MODIS, VIIRS)
- Market data (Yahoo Finance, EIA, etc.)
- API data sources
"""

from pathlib import Path

from data.loaders.manifest import (
    load_aoi,
    search_sentinel1,
    build_manifest,
    save_stac_items,
    load_stac_items,
    run_manifest_builder,
)

__all__ = [
    "load_aoi",
    "search_sentinel1",
    "build_manifest",
    "save_stac_items",
    "load_stac_items",
    "run_manifest_builder",
]
