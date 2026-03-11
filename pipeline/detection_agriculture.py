"""
NDVI calculation for agricultural crop health monitoring.

Uses Sentinel-2 red and NIR bands to calculate vegetation indices.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
from datetime import date, timedelta
import pandas as pd


def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Calculate Normalized Difference Vegetation Index.
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        red: Red band (Sentinel-2 B4)
        nir: Near-infrared band (Sentinel-2 B8)
    
    Returns:
        NDVI array with values from -1 to 1
    """
    # Avoid division by zero
    denominator = nir + red
    ndvi = np.where(denominator != 0, (nir - red) / denominator, 0)
    
    return ndvi


def calculate_evi(red: np.ndarray, nir: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """
    Calculate Enhanced Vegetation Index.
    
    EVI is less sensitive to atmosphere and canopy background than NDVI.
    
    EVI = 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
    """
    numerator = nir - red
    denominator = nir + 6 * red - 7.5 * blue + 1
    
    evi = np.where(denominator != 0, 2.5 * numerator / denominator, 0)
    
    return evi


def classify_crop_health(ndvi: np.ndarray) -> np.ndarray:
    """
    Classify NDVI values into health categories.
    
    Categories:
    - < 0.2: Bare soil / stressed
    - 0.2 - 0.4: Sparse vegetation
    - 0.4 - 0.6: Moderate vegetation
    - 0.6 - 0.8: Healthy vegetation
    - > 0.8: Very healthy / dense vegetation
    """
    classes = np.zeros_like(ndvi, dtype=int)
    
    classes[ndvi < 0.2] = 0  # Bare/stressed
    classes[(ndvi >= 0.2) & (ndvi < 0.4)] = 1  # Sparse
    classes[(ndvi >= 0.4) & (ndvi < 0.6)] = 2  # Moderate
    classes[(ndvi >= 0.6) & (ndvi < 0.8)] = 3  # Healthy
    classes[ndvi >= 0.8] = 4  # Very healthy
    
    return classes


def process_sentinel2_for_ndvi(
    aoi_path: str,
    target_date: str,
    output_base: str = "outputs",
    cloud_threshold: float = 0.2,
) -> Dict:
    """
    Process Sentinel-2 scene to calculate NDVI for an agricultural AOI.
    
    Args:
        aoi_path: Path to AOI GeoJSON
        target_date: Date to analyze (YYYY-MM-DD)
        output_base: Output directory
        cloud_threshold: Max cloud cover fraction allowed
    
    Returns:
        Dictionary with NDVI statistics
    """
    import planetary_computer
    import pystac_client
    from pystac.extensions.eo import EOExtension as EO
    import rasterio
    from rasterio.mask import mask as rasterio_mask
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import transform
    
    # Load AOI
    aoi_gdf = gpd.read_file(aoi_path)
    aoi_geom = aoi_gdf.union_all()
    aoi_bounds = aoi_geom.bounds
    
    # Search Sentinel-2
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=aoi_bounds,
        datetime=f"{target_date}/{target_date}",
        query={"eo:cloud_cover": {"lt": cloud_threshold * 100}},
    )
    
    items = list(search.items())
    
    if not items:
        return {
            "status": "no_data",
            "date": target_date,
            "aoi": aoi_path,
            "message": f"No Sentinel-2 scenes found for {target_date}",
        }
    
    # Use the first (least cloudy) item
    item = min(items, key=lambda i: EO.ext(i).cloud_cover or 100.0)
    
    # Load Red (B4) and NIR (B8) bands
    red_href = item.assets["B04"].href
    nir_href = item.assets["B08"].href
    
    results = {
        "status": "success",
        "date": target_date,
        "aoi": aoi_path,
        "scene_id": item.id,
        "cloud_cover": EO.ext(item).cloud_cover,
        "aoi_name": aoi_gdf.iloc[0].get("name", "unknown") if len(aoi_gdf) > 0 else "unknown",
    }
    output_path = Path(output_base) / target_date / "agriculture"
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        with rasterio.open(red_href) as red_src, rasterio.open(nir_href) as nir_src:
            transformer = Transformer.from_crs("EPSG:4326", red_src.crs, always_xy=True)
            aoi_projected = transform(transformer.transform, aoi_geom)
            raster_bbox = box(*red_src.bounds)

            if not aoi_projected.intersects(raster_bbox):
                results["status"] = "error"
                results["error"] = "Input shapes do not overlap raster."
                result_file = output_path / "ndvi_stats.json"
                result_file.write_text(json.dumps(results, indent=2))
                return results

            intersection = aoi_projected.intersection(raster_bbox)

            # Crop to AOI
            out_image_red, out_transform = rasterio_mask(
                red_src, [intersection.__geo_interface__], crop=True
            )
            out_image_nir, _ = rasterio_mask(
                nir_src, [intersection.__geo_interface__], crop=True
            )
            
            red = out_image_red[0].astype(float)
            nir = out_image_nir[0].astype(float)
            
            # Scale to reflectance (Sentinel-2 L2A is already scaled, but ensure float)
            red = red / 10000.0
            nir = nir / 10000.0
            
            # Calculate NDVI
            ndvi = calculate_ndvi(red, nir)
            
            # Calculate statistics
            valid_ndvi = ndvi[ndvi != 0]  # Exclude no-data
            
            results["ndvi_stats"] = {
                "mean": float(np.mean(valid_ndvi)),
                "median": float(np.median(valid_ndvi)),
                "std": float(np.std(valid_ndvi)),
                "min": float(np.min(valid_ndvi)),
                "max": float(np.max(valid_ndvi)),
                "p25": float(np.percentile(valid_ndvi, 25)),
                "p75": float(np.percentile(valid_ndvi, 75)),
                "valid_pixels": int(len(valid_ndvi)),
                "total_pixels": int(ndvi.size),
            }
            
            # Classify health
            health_classes = classify_crop_health(ndvi)
            class_counts = np.bincount(health_classes.flatten(), minlength=5)
            
            results["health_distribution"] = {
                "bare_stressed": float(class_counts[0] / ndvi.size),
                "sparse": float(class_counts[1] / ndvi.size),
                "moderate": float(class_counts[2] / ndvi.size),
                "healthy": float(class_counts[3] / ndvi.size),
                "very_healthy": float(class_counts[4] / ndvi.size),
            }
            
            # Save NDVI raster
            ndvi_file = output_path / "ndvi.tif"
            with rasterio.open(
                ndvi_file,
                'w',
                driver='GTiff',
                height=ndvi.shape[0],
                width=ndvi.shape[1],
                count=1,
                dtype=ndvi.dtype,
                crs=red_src.crs,
                transform=out_transform,
            ) as dst:
                dst.write(ndvi, 1)
            
            results["ndvi_file"] = str(ndvi_file)
    
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
    
    # Save results
    result_file = output_path / "ndvi_stats.json"
    result_file.write_text(json.dumps(results, indent=2))
    
    return results


def calculate_ndvi_anomaly(
    current_ndvi: float,
    historical_ndvi: pd.DataFrame,
    target_week: int,
) -> Dict:
    """
    Calculate NDVI anomaly vs historical baseline.
    
    Args:
        current_ndvi: Current NDVI mean value
        historical_ndvi: DataFrame with historical NDVI data
        target_week: Week number (1-52) for seasonal comparison
    
    Returns:
        Dictionary with anomaly metrics
    """
    # Filter historical data for same week
    same_week = historical_ndvi[
        pd.to_datetime(historical_ndvi['date']).dt.week == target_week
    ]
    
    if len(same_week) < 3:
        return {
            "anomaly": 0,
            "zscore": 0,
            "confidence": "low",
            "message": "Insufficient historical data",
        }
    
    baseline_mean = same_week['ndvi_mean'].mean()
    baseline_std = same_week['ndvi_mean'].std()
    
    if baseline_std > 0:
        zscore = (current_ndvi - baseline_mean) / baseline_std
    else:
        zscore = 0
    
    anomaly = current_ndvi - baseline_mean
    anomaly_pct = anomaly / baseline_mean if baseline_mean > 0 else 0
    
    # Determine confidence
    if len(same_week) >= 5:
        confidence = "high"
    elif len(same_week) >= 3:
        confidence = "medium"
    else:
        confidence = "low"
    
    return {
        "anomaly": anomaly,
        "anomaly_pct": anomaly_pct,
        "zscore": zscore,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "confidence": confidence,
        "historical_years": len(same_week),
    }


def estimate_yield_deviation(ndvi_anomaly: float, crop_type: str = "corn") -> Dict:
    """
    Estimate yield deviation from NDVI anomaly.
    
    Based on research showing correlation between NDVI and yield.
    
    Args:
        ndvi_anomaly: NDVI anomaly value
        crop_type: Type of crop
    
    Returns:
        Dictionary with yield estimate
    """
    # Simplified relationship: ~1% NDVI change ≈ ~1-2% yield change
    # Real models are more complex and crop-specific
    
    yield_factors = {
        "corn": 1.5,  # NDVI-yield correlation factor
        "soybeans": 1.3,
        "wheat": 1.2,
    }
    
    factor = yield_factors.get(crop_type, 1.0)
    yield_deviation = ndvi_anomaly * factor
    
    return {
        "crop": crop_type,
        "ndvi_anomaly": ndvi_anomaly,
        "yield_deviation_pct": yield_deviation * 100,
        "direction": "above_trend" if yield_deviation > 0 else "below_trend",
        "confidence": "medium",  # Would need calibration
    }


if __name__ == "__main__":
    print("NDVI Agricultural Monitoring Module")
    print("=" * 40)
    print()
    print("Usage:")
    print("  from pipeline.detection_agriculture import process_sentinel2_for_ndvi")
    print("  result = process_sentinel2_for_ndvi('configs/aoi_agri_iowa.geojson', '2026-06-15')")
    print()
    print("NDVI interpretation:")
    print("  < 0.2: Bare soil or stressed crops")
    print("  0.2-0.4: Sparse vegetation")
    print("  0.4-0.6: Moderate vegetation")
    print("  0.6-0.8: Healthy vegetation")
    print("  > 0.8: Very healthy/dense vegetation")
