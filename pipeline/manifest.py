"""
STAC Search → Manifest Builder

Queries STAC APIs (Planetary Computer) for Sentinel-1 GRD scenes over AOI.
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional, List

import geopandas as gpd
import pandas as pd
from pystac_client import Client

def load_aoi(config_path: str = "configs/aoi_hormuz.geojson") -> dict:
    """Load AOI polygon from GeoJSON config."""
    with open(config_path) as f:
        return json.load(f)

def search_sentinel1(
    aoi: dict,
    start_date: date,
    end_date: date,
    collection: str = "sentinel-1-grd",
    stac_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
) -> List[dict]:
    """
    Search STAC for Sentinel-1 GRD scenes over AOI.

    Args:
        aoi: GeoJSON FeatureCollection with AOI polygon
        start_date: Start date
        end_date: End date
        collection: STAC collection name
        stac_url: STAC API endpoint

    Returns:
        List of STAC items
    """
    # Extract AOI geometry
    aoi_geom = aoi["features"][0]["geometry"]

    # Connect to STAC
    client = Client.open(stac_url)

    # Search
    search = client.search(
        collections=[collection],
        intersects=aoi_geom,
        datetime=f"{start_date.isoformat()}/{end_date.isoformat()}",
        query={
            "sar:instrument_mode": {"eq": "IW"},
            "sar:polarizations": {"contains": "VV"}
        }
    )

    items = list(search.items())
    print(f"Found {len(items)} Sentinel-1 scenes")
    return items

def build_manifest(
    items: List[dict],
    output_path: str = "outputs/manifests"
) -> pd.DataFrame:
    """
    Build manifest DataFrame from STAC items.

    Args:
        items: List of STAC items
        output_path: Directory to save manifest

    Returns:
        Manifest DataFrame
    """
    records = []

    for item in items:
        props = item.properties

        record = {
            "item_id": item.id,
            "datetime": props.get("datetime"),
            "geometry": json.dumps(item.geometry),
            "orbit": props.get("sat:orbit_state", "unknown"),
            "polarizations": ",".join(props.get("sar:polarizations", [])),
            "instrument_mode": props.get("sar:instrument_mode"),
            "resolution": props.get("sar:resolution_range"),
            "incidence_angle": props.get("sar:incidence_angle"),
            "provider": "planetary-computer"
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Save to parquet
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(df) > 0:
        date_str = df.iloc[0]["datetime"][:10]
        output_file = output_dir / f"{date_str}.parquet"
        df.to_parquet(output_file, index=False)
        print(f"Saved manifest: {output_file}")

    return df

def run_manifest_builder(
    start_date: date,
    end_date: date,
    aoi_path: str = "configs/aoi_hormuz.geojson",
    output_path: str = "outputs/manifests"
) -> pd.DataFrame:
    """
    Full manifest builder pipeline.

    Args:
        start_date: Start date
        end_date: End date
        aoi_path: Path to AOI GeoJSON
        output_path: Directory to save manifests

    Returns:
        Manifest DataFrame
    """
    # Load AOI
    aoi = load_aoi(aoi_path)

    # Search STAC
    items = search_sentinel1(aoi, start_date, end_date)

    # Build manifest
    manifest = build_manifest(items, output_path)

    return manifest

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--aoi", type=str, default="configs/aoi_hormuz.geojson")
    parser.add_argument("--output", type=str, default="outputs/manifests")

    args = parser.parse_args()

    manifest = run_manifest_builder(
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
        args.aoi,
        args.output
    )

    print(f"\nManifest summary:")
    print(manifest.head())
    print(f"\nTotal scenes: {len(manifest)}")
