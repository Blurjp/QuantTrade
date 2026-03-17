"""
Detection dispatcher for multi-type monitoring.

Routes detection requests to the appropriate monitoring module based on type.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import pandas as pd


def run_detection(
    monitoring_type: str,
    aoi_path: str,
    target_date: str,
    output_base: str = "outputs",
    **kwargs
) -> Dict[str, Any]:
    """
    Run detection for a given monitoring type and region.

    This is a dispatcher that routes to the appropriate detection module
    based on the monitoring type.

    Args:
        monitoring_type: Type of monitoring (e.g., "agriculture", "oil_storage", "chokepoint")
        aoi_path: Path to AOI GeoJSON file
        target_date: Date to process (YYYY-MM-DD)
        output_base: Base output directory
        **kwargs: Additional parameters for specific detection types

    Returns:
        Dictionary with detection results
    """
    normalized_type = monitoring_type.lower().replace("-", "_").replace(" ", "_")

    # Route to appropriate detection module
    if normalized_type in ("agriculture", "agricultural"):
        return _run_agriculture_detection(aoi_path, target_date, output_base, **kwargs)
    elif normalized_type in ("oil_storage", "oil_storage"):
        return _run_oil_storage_detection(aoi_path, target_date, output_base, **kwargs)
    elif normalized_type in ("chokepoint", "port_logistics"):
        return _run_chokepoint_detection(aoi_path, target_date, output_base, **kwargs)
    elif normalized_type in ("auto_inventory", "autoinventory"):
        return _run_auto_inventory_detection(aoi_path, target_date, output_base, **kwargs)
    else:
        # Return a placeholder result for unknown types
        return {
            "date": target_date,
            "type": monitoring_type,
            "status": "unsupported",
            "message": f"No detection implementation for type: {monitoring_type}",
            "count": 0,
            "details": [],
            "metadata": {"status": "success"},
        }


def _run_agriculture_detection(
    aoi_path: str,
    target_date: str,
    output_base: str,
    **kwargs
) -> Dict[str, Any]:
    """Run agricultural vegetation health detection."""
    try:
        from pipeline.vegetation_health import VegetationHealthMonitor

        # Extract region ID from AOI path
        region_id = Path(aoi_path).stem.replace("aoi_", "").replace("aoi-auto-", "")

        monitor = VegetationHealthMonitor(output_base=output_base)

        # Try to find the matching region in the monitor's regions
        region_id_mapping = {
            "brazil_soy": "brazil_cerrado",
            "brazil_soy_central": "brazil_cerrado",
            "argentina_pampas": "argentina_pampas",
            "usa_corn_belt": "usa_corn_soybeans",
            "usa_soybeans": "usa_corn_soybeans",
        }

        mapped_region = region_id_mapping.get(region_id, region_id)

        # Try to generate signal
        if mapped_region in monitor.regions:
            data = monitor.fetch_ndvi_data(mapped_region, target_date)
            if data:
                return {
                    "date": target_date,
                    "type": "agriculture",
                    "status": "success",
                    "region": region_id,
                    "count": 1,
                    "details": [data],
                    "metadata": {"status": "success", "data_source": data.get("data_source", "unknown")},
                }

        # Fallback: return simulated/placeholder data
        return _get_placeholder_agriculture_data(target_date, region_id)

    except ImportError:
        return _get_placeholder_agriculture_data(target_date, Path(aoi_path).stem)
    except Exception as e:
        return {
            "date": target_date,
            "type": "agriculture",
            "status": "error",
            "message": str(e),
            "count": 0,
            "details": [],
            "metadata": {"status": "error"},
        }


def _run_oil_storage_detection(
    aoi_path: str,
    target_date: str,
    output_base: str,
    **kwargs
) -> Dict[str, Any]:
    """Run oil storage tank level detection."""
    try:
        from pipeline.detection_storage import analyze_cushing_storage

        region_id = Path(aoi_path).stem.replace("aoi_", "")

        result = analyze_cushing_storage(
            aoi_path=aoi_path,
            target_date=target_date,
            output_base=output_base,
        )

        return {
            "date": target_date,
            "type": "oil_storage",
            "status": result.get("status", "success"),
            "region": region_id,
            "count": 1,
            "details": [{
                "date": target_date,
                "fill_pct": result.get("fill_pct"),
                "tanks_detected": result.get("tanks_detected", 0),
            }],
            "metadata": {"status": "success", "method": result.get("method", "tank_detection")},
        }

    except ImportError:
        return _get_placeholder_storage_data(target_date, Path(aoi_path).stem)
    except Exception as e:
        return {
            "date": target_date,
            "type": "oil_storage",
            "status": "error",
            "message": str(e),
            "count": 0,
            "details": [],
            "metadata": {"status": "error"},
        }


def _run_chokepoint_detection(
    aoi_path: str,
    target_date: str,
    output_base: str,
    **kwargs
) -> Dict[str, Any]:
    """Run chokepoint/port logistics detection."""
    region_id = Path(aoi_path).stem.replace("aoi_", "")

    # Placeholder for now - would integrate with ship detection
    return {
        "date": target_date,
        "type": "chokepoint",
        "status": "simulated",
        "region": region_id,
        "count": 15,  # Placeholder vessel count
        "details": [{
            "date": target_date,
            "detections": 15,
            "throughput_index": 0.75,
        }],
        "metadata": {"status": "success", "note": "Simulated data"},
    }


def _run_auto_inventory_detection(
    aoi_path: str,
    target_date: str,
    output_base: str,
    **kwargs
) -> Dict[str, Any]:
    """Run auto inventory detection (using NDVI as proxy)."""
    region_id = Path(aoi_path).stem.replace("aoi_", "").replace("auto_", "")

    # This uses NDVI as a proxy for parking lot fullness
    # Higher NDVI in parking areas = fewer cars = lower inventory
    # Lower NDVI = more cars = higher inventory

    return _get_placeholder_auto_inventory_data(target_date, region_id)


def _get_placeholder_agriculture_data(target_date: str, region_id: str) -> Dict[str, Any]:
    """Generate placeholder agriculture data for testing."""
    import numpy as np

    np.random.seed(hash(target_date + region_id) % 2**32)

    # Simulate NDVI
    baseline_ndvi = 0.65
    seasonal_var = 0.1 * np.sin(2 * np.pi * (datetime.strptime(target_date, "%Y-%m-%d").timetuple().tm_yday - 120) / 365)
    noise = np.random.normal(0, 0.05)
    ndvi = baseline_ndvi + seasonal_var + noise

    return {
        "date": target_date,
        "type": "agriculture",
        "status": "simulated",
        "region": region_id,
        "count": 1,
        "details": [{
            "date": target_date,
            "ndvi_mean": round(ndvi, 3),
            "valid_pixels": 10000,
        }],
        "metadata": {"status": "success", "note": "Simulated data", "data_source": "placeholder"},
    }


def _get_placeholder_storage_data(target_date: str, region_id: str) -> Dict[str, Any]:
    """Generate placeholder storage data for testing."""
    import numpy as np

    np.random.seed(hash(target_date + region_id) % 2**32)

    fill_pct = 50 + np.random.normal(0, 15)
    fill_pct = max(10, min(90, fill_pct))

    return {
        "date": target_date,
        "type": "oil_storage",
        "status": "simulated",
        "region": region_id,
        "count": 1,
        "details": [{
            "date": target_date,
            "fill_pct": round(fill_pct, 1),
            "tanks_detected": 100,
        }],
        "metadata": {"status": "success", "note": "Simulated data"},
    }


def _get_placeholder_auto_inventory_data(target_date: str, region_id: str) -> Dict[str, Any]:
    """Generate placeholder auto inventory data for testing."""
    import numpy as np

    np.random.seed(hash(target_date + region_id) % 2**32)

    # Use NDVI as proxy (lower = more cars = higher inventory)
    baseline_ndvi = 0.10  # Parking lots have low NDVI
    noise = np.random.normal(0, 0.02)
    ndvi = baseline_ndvi + noise

    return {
        "date": target_date,
        "type": "auto_inventory",
        "status": "simulated",
        "region": region_id,
        "count": 1,
        "details": [{
            "date": target_date,
            "ndvi_mean": round(max(0.05, min(0.25, ndvi)), 3),
            "valid_pixels": 5000,
        }],
        "metadata": {"status": "success", "note": "Simulated data - using NDVI as inventory proxy"},
    }


__all__ = ["run_detection"]
