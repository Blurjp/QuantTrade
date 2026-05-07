"""
STAC Search → Manifest Builder

Queries STAC APIs (Planetary Computer) for Sentinel-1 GRD scenes over AOI.

NOTE: This module is now a backward-compat wrapper.
      New code should import from data.loaders.manifest instead.
"""

# Backward-compat re-export
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
