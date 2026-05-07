"""
Vehicle detection for parking lots and auto inventory.

Uses YOLOv8 for vehicle detection in optical satellite imagery.
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import date
import geopandas as gpd
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)


class VehicleDetector:
    """
    Vehicle detection using YOLOv8 or simple brightness threshold.
    
    For production: Use YOLOv8 trained on satellite parking lot imagery.
    For quick start: Use simple brightness threshold on Sentinel-2.
    """
    
    def __init__(self, model_path: str = None, use_yolo: bool = False):
        """
        Initialize vehicle detector.
        
        Args:
            model_path: Path to YOLOv8 model weights
            use_yolo: If True, use YOLO. If False, use simple threshold.
        """
        self.use_yolo = use_yolo
        self.model = None
        
        if use_yolo and model_path:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
            except ImportError:
                print("Warning: ultralytics not installed. Falling back to simple detection.")
                self.use_yolo = False
    
    def detect_vehicles_yolo(self, image: np.ndarray) -> List[Dict]:
        """Detect vehicles using YOLOv8."""
        if self.model is None:
            return []
        
        results = self.model(image)
        detections = []
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls in [2, 3, 5, 7]:  # COCO: car, motorcycle, bus, truck
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "type": "vehicle",
                        "class": r.names[cls],
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                    })
        
        return detections
    
    def detect_vehicles_simple(self, image: np.ndarray, parking_mask: np.ndarray = None) -> List[Dict]:
        """
        Simple vehicle detection using brightness threshold.
        
        Works on parking lots where cars appear darker than asphalt.
        """
        if len(image.shape) == 3:
            # Convert to grayscale
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Apply parking lot mask if provided
        if parking_mask is not None:
            gray = gray * parking_mask
        
        # Find dark spots (vehicles on asphalt)
        # Cars typically 4-6m long, in 10m Sentinel-2 that's ~0.5 pixels
        # For high-res (1-3m), use connected components
        
        from scipy import ndimage
        
        # Threshold for dark spots
        threshold = np.percentile(gray[gray > 0], 30) if parking_mask is not None else np.percentile(gray, 30)
        dark_spots = gray < threshold
        
        # Label connected components
        labeled, num_features = ndimage.label(dark_spots)
        
        detections = []
        for i in range(1, num_features + 1):
            region = (labeled == i)
            area = np.sum(region)
            
            # Filter by size (typical vehicle is 5-20 sq meters)
            # In pixels: depends on resolution
            if 1 < area < 50:  # Rough filter
                cy, cx = ndimage.center_of_mass(region)
                detections.append({
                    "type": "vehicle",
                    "class": "vehicle",
                    "confidence": 0.5,
                    "center": [cx, cy],
                    "area_pixels": int(area),
                })
        
        return detections
    
    def detect(self, image: np.ndarray, parking_mask: np.ndarray = None) -> List[Dict]:
        """Run vehicle detection."""
        if self.use_yolo:
            return self.detect_vehicles_yolo(image)
        else:
            return self.detect_vehicles_simple(image, parking_mask)


def count_vehicles_in_parking_lots(
    aoi_path: str,
    image_date: str,
    output_base: str = "outputs",
    detector: VehicleDetector = None,
) -> Dict:
    """
    Count vehicles in parking lots from satellite imagery.
    
    Args:
        aoi_path: Path to AOI GeoJSON with parking lot polygons
        image_date: Date to analyze
        output_base: Output directory
        detector: VehicleDetector instance
    
    Returns:
        Dictionary with vehicle counts per parking lot
    """
    if detector is None:
        detector = VehicleDetector(use_yolo=False)

    try:
        from data.loaders.manifest import load_stac_items
        import planetary_computer
        import rasterio
        from rasterio.mask import mask as rasterio_mask
        from shapely.geometry import mapping
        _sat_deps = True
    except ImportError:
        _sat_deps = False

    # Load AOI
    aoi = gpd.read_file(aoi_path)
    
    results = {
        "date": image_date,
        "aoi": aoi_path,
        "lots": [],
        "total_vehicles": 0,
    }
    
    # For each parking lot in AOI
    for idx, lot in aoi.iterrows():
        lot_name = lot.get("name", f"lot_{idx}")
        lot_geom = lot.geometry
        
        lot_result = {
            "name": lot_name,
            "polygon": lot_geom.bounds,
            "vehicle_count": 0,
            "confidence": "low",
            "note": "Sentinel-2 fetch failed",
        }
        
        if not _sat_deps:
            lot_result["note"] = "Missing satellite dependencies"
        else:
            try:
                items = load_stac_items(
                    lot_geom,
                    image_date,
                    collection="sentinel-2-l2a",
                    max_cloud_cover=30,
                )

                if not items:
                    lot_result["note"] = "No Sentinel-2 scenes available"
                    results["lots"].append(lot_result)
                    results["total_vehicles"] += lot_result["vehicle_count"]
                    continue

                signed_item = planetary_computer.sign(items[0])
                asset = signed_item.assets.get("visual") or signed_item.assets.get("B04")

                if not asset:
                    lot_result["note"] = "No visual/B04 asset in scene"
                    results["lots"].append(lot_result)
                    results["total_vehicles"] += lot_result["vehicle_count"]
                    continue

                with rasterio.open(asset.href) as src:
                    out_image, out_transform = rasterio_mask(
                        src, [mapping(lot_geom)], crop=True
                    )

                image = np.transpose(out_image, (1, 2, 0))

                if image.shape[0] < 10 or image.shape[1] < 10:
                    lot_result["note"] = "Cropped image too small"
                    results["lots"].append(lot_result)
                    results["total_vehicles"] += lot_result["vehicle_count"]
                    continue

                detections = detector.detect(image)

                lot_result = {
                    "name": lot_name,
                    "polygon": lot_geom.bounds,
                    "vehicle_count": len(detections),
                    "confidence": "high" if detector.use_yolo else "medium",
                    "detections": detections,
                    "note": "success",
                }
            except Exception as e:
                logger.warning(
                    "Vehicle detection failed for lot %s: %s", lot_name, e
                )
                lot_result["note"] = str(e)
        
        results["lots"].append(lot_result)
        results["total_vehicles"] += lot_result["vehicle_count"]
    
    # Save results
    output_path = Path(output_base) / image_date / "parking"
    output_path.mkdir(parents=True, exist_ok=True)
    
    result_file = output_path / "vehicle_count.json"
    result_file.write_text(json.dumps(results, indent=2, default=str))
    
    return results


def get_retail_baseline(ticker: str, lookback_weeks: int = 52) -> Dict:
    """
    Get baseline parking traffic for retail stores.
    
    Uses historical satellite data or alternative data sources.
    """
    TICKER_REGION_MAP = {"WMT": "walmart_hq", "COST": "costco_hq", "TGT": "target_hq"}
    region_id = TICKER_REGION_MAP.get(ticker.upper())
    
    if not region_id:
        return {"weekly_avg_visits": 0, "source": "unknown_ticker"}
    
    vehicle_counts = []
    output_base = Path("outputs")
    dirs_scanned = 0

    for date_dir in sorted(output_base.iterdir(), reverse=True):
        if not date_dir.is_dir() or not date_dir.name[0:4].isdigit():
            continue

        dirs_scanned += 1
        if dirs_scanned > MAX_DATE_DIRS_TO_SCAN:
            break
        
        region_dir = date_dir / "parking" / region_id
        if not region_dir.exists():
            continue
            
        for count_file in region_dir.glob("vehicle_count.json"):
            try:
                import json
                data = json.loads(count_file.read_text())
                vehicle_counts.append(data.get("total_vehicles", 0))
            except (json.JSONDecodeError, KeyError):
                continue
        
        if len(vehicle_counts) >= lookback_weeks * 7:
            break
    
    if not vehicle_counts:
        return {"weekly_avg_visits": 0, "source": "no_historical_data"}
    
    return {
        "weekly_avg_visits": float(np.mean(vehicle_counts)),
        "std_visits": float(np.std(vehicle_counts)) if len(vehicle_counts) > 1 else 0,
        "sample_count": len(vehicle_counts),
        "min": min(vehicle_counts),
        "max": max(vehicle_counts),
        "source": "satellite_historical",
    }


if __name__ == "__main__":
    # Test vehicle detector
    print("Vehicle Detection Module")
    print("=" * 40)
    print()
    print("Detection methods:")
    print("  1. YOLOv8 (requires model + ultralytics)")
    print("  2. Simple brightness threshold (quick start)")
    print()
    print("Usage:")
    print("  from pipeline.detection_vehicles import VehicleDetector")
    print("  detector = VehicleDetector(use_yolo=False)")
    print("  detections = detector.detect(image)")
