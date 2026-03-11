"""
Multi-target historical data backfill.

Backfills detection data for all monitoring targets.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
from typing import Dict, List
import numpy as np


def load_region_config(region_id: str) -> dict:
    """Load region configuration from registry."""
    registry_path = Path("configs/regions/registry_v2.json")
    if not registry_path.exists():
        registry_path = Path("configs/regions/registry.json")
    
    with open(registry_path) as f:
        registry = json.load(f)
    
    return registry.get("regions", {}).get(region_id)


def load_aoi_geometry(aoi_path: str):
    """Load AOI geometry from GeoJSON."""
    import geopandas as gpd
    from shapely.geometry import shape
    
    gdf = gpd.read_file(aoi_path)
    
    # Ensure WGS84 CRS
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    # Get union of all features
    union = gdf.union_all()
    
    # Return as GeoJSON-like dict for rasterio
    return {
        "type": "Polygon" if union.geom_type == "Polygon" else "MultiPolygon",
        "coordinates": union.__geo_interface__["coordinates"]
    }


def backfill_sar_region(
    region_id: str,
    start_date: str,
    end_date: str,
    output_base: str = "outputs",
    max_scenes: int = 20,
) -> Dict:
    """
    Backfill SAR-based monitoring (chokepoints, ports).
    
    Uses Sentinel-1 RTC imagery with simple bright pixel counting.
    """
    region = load_region_config(region_id)
    if not region:
        return {"status": "error", "message": f"Region {region_id} not found"}
    
    aoi_geom = load_aoi_geometry(region["aoi_file"])
    
    # Get bounds from geometry
    from shapely.geometry import shape, box
    aoi_shape = shape(aoi_geom)
    aoi_bounds = aoi_shape.bounds
    
    results = {
        "region": region_id,
        "name": region.get("name", region_id),
        "start_date": start_date,
        "end_date": end_date,
        "type": region.get("type", "unknown"),
        "scenes_processed": 0,
        "total_detections": 0,
        "daily_stats": [],
    }
    
    # Search for Sentinel-1 scenes
    import planetary_computer
    import pystac_client
    import rasterio
    from rasterio.mask import mask as rasterio_mask
    from pyproj import Transformer
    from shapely.ops import transform
    
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    search = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=aoi_bounds,
        datetime=f"{start_date}/{end_date}",
    )
    
    items = list(search.items())
    print(f"Found {len(items)} Sentinel-1 scenes")
    
    items_to_process = items[:max_scenes]
    
    for item in items_to_process:
        try:
            scene_date = item.datetime.strftime("%Y-%m-%d")
            signed_item = planetary_computer.sign(item)
            
            with rasterio.open(signed_item.assets["vh"].href) as src:
                raster_crs = src.crs
                
                # Transform AOI to raster CRS
                transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
                aoi_projected = transform(transformer.transform, aoi_shape)
                
                # Check overlap
                raster_bbox = box(*src.bounds)
                if not aoi_projected.intersects(raster_bbox):
                    print(f"  {scene_date}: No overlap")
                    continue
                
                intersection = aoi_projected.intersection(raster_bbox)
                out_image, _ = rasterio_mask(src, [intersection.__geo_interface__], crop=True)
                data = out_image[0].astype(float)
            
            # Simple ship detection: count bright pixels
            if data.size > 0:
                threshold = np.percentile(data[data > 0], 95) if np.any(data > 0) else 0
                bright_pixels = np.sum(data > threshold)
                detections = int(bright_pixels / 100)
            else:
                detections = 0
            
            results["daily_stats"].append({
                "date": scene_date,
                "detections": detections,
            })
            results["total_detections"] += detections
            results["scenes_processed"] += 1
            
            print(f"  {scene_date}: {detections}")
            time.sleep(0.2)
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Save
    output_path = Path(output_base) / "backfill"
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / f"{region_id}_backfill.json").write_text(json.dumps(results, indent=2))
    
    return results


def backfill_optical_region(
    region_id: str,
    start_date: str,
    end_date: str,
    output_base: str = "outputs",
    max_scenes: int = 15,
) -> Dict:
    """
    Backfill optical monitoring (agriculture, retail, storage).
    
    Uses Sentinel-2 imagery with NDVI/brightness analysis.
    """
    region = load_region_config(region_id)
    if not region:
        return {"status": "error", "message": f"Region {region_id} not found"}
    
    aoi_geom = load_aoi_geometry(region["aoi_file"])
    
    # Get bounds from geometry
    from shapely.geometry import shape
    aoi_shape = shape(aoi_geom)
    aoi_bounds = aoi_shape.bounds
    
    results = {
        "region": region_id,
        "name": region.get("name", region_id),
        "start_date": start_date,
        "end_date": end_date,
        "type": region.get("type", "unknown"),
        "scenes_processed": 0,
        "weekly_stats": [],
    }
    
    import planetary_computer
    import pystac_client
    import rasterio
    from rasterio.mask import mask as rasterio_mask
    from pyproj import Transformer
    from shapely.ops import transform
    
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=aoi_bounds,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": 30}},
    )
    
    items = list(search.items())
    print(f"Found {len(items)} Sentinel-2 scenes")
    
    # Sample weekly
    items_to_process = items[::max(1, len(items) // max_scenes)][:max_scenes]
    
    for item in items_to_process:
        try:
            scene_date = item.datetime.strftime("%Y-%m-%d")
            signed_item = planetary_computer.sign(item)
            
            # Load Red band first to get CRS
            with rasterio.open(signed_item.assets["B04"].href) as red_src:
                raster_crs = red_src.crs
                
                # Transform AOI to raster CRS
                transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
                aoi_projected = transform(transformer.transform, aoi_shape)
                
                # Check if AOI overlaps with raster
                from shapely.geometry import box
                raster_bbox = box(*red_src.bounds)
                
                if not aoi_projected.intersects(raster_bbox):
                    print(f"  {scene_date}: No overlap with scene")
                    continue
                
                # Get intersection for masking
                intersection = aoi_projected.intersection(raster_bbox)
                intersection_geojson = intersection.__geo_interface__
                
                red, _ = rasterio_mask(red_src, [intersection_geojson], crop=True)
            
            # Load NIR band
            with rasterio.open(signed_item.assets["B08"].href) as nir_src:
                nir, _ = rasterio_mask(nir_src, [intersection_geojson], crop=True)
            
            red = red[0].astype(float) / 10000
            nir = nir[0].astype(float) / 10000
            
            # Calculate NDVI
            denom = nir + red
            ndvi = np.where(denom > 0, (nir - red) / denom, 0)
            ndvi_mean = float(np.mean(ndvi[ndvi != 0])) if np.any(ndvi != 0) else 0
            
            results["weekly_stats"].append({
                "date": scene_date,
                "ndvi_mean": ndvi_mean,
            })
            results["scenes_processed"] += 1
            
            print(f"  {scene_date}: NDVI={ndvi_mean:.3f}")
            time.sleep(0.2)
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Save
    output_path = Path(output_base) / "backfill"
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / f"{region_id}_backfill.json").write_text(json.dumps(results, indent=2))
    
    return results


def run_multi_backfill(
    targets: List[str],
    start_date: str,
    end_date: str,
    output_base: str = "outputs",
) -> Dict:
    """Run backfill for multiple targets."""
    results = {
        "start_date": start_date,
        "end_date": end_date,
        "targets": targets,
        "backfill_results": {},
    }
    
    # SAR targets (chokepoint, port_logistics)
    sar_types = {"chokepoint", "port_logistics"}
    
    for target in targets:
        region = load_region_config(target)
        if not region:
            print(f"⚠️  Unknown region: {target}")
            continue
        
        region_type = region.get("type")
        region_name = region.get("name", target)
        
        print(f"\n{'='*60}")
        print(f"Backfilling: {region_name} ({target})")
        print(f"Type: {region_type}")
        print(f"{'='*60}\n")
        
        if region_type in sar_types:
            result = backfill_sar_region(target, start_date, end_date, output_base)
        else:
            result = backfill_optical_region(target, start_date, end_date, output_base)
        
        results["backfill_results"][target] = result
    
    # Save summary
    output_path = Path(output_base) / "backfill"
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "multi_backfill_summary.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    
    print(f"\n{'='*60}")
    print("BACKFILL COMPLETE")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-target backfill")
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", default="outputs")
    
    args = parser.parse_args()
    run_multi_backfill(args.targets, args.start, args.end, args.output)
