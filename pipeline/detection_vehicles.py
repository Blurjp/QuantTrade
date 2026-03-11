"""
Vehicle detection for parking lots and auto inventory.

Uses YOLOv8 for vehicle detection in optical satellite imagery.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import date
import geopandas as gpd
from shapely.geometry import Point, Polygon


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
    import rasterio
    from rasterio.mask import mask as rasterio_mask
    
    if detector is None:
        detector = VehicleDetector(use_yolo=False)
    
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
        
        # TODO: Load Sentinel-2 scene and crop to lot
        # For now, return placeholder
        lot_result = {
            "name": lot_name,
            "polygon": lot_geom.bounds,
            "vehicle_count": 0,
            "confidence": "low",
            "note": "Pending Sentinel-2 data integration",
        }
        
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
    # TODO: Implement baseline calculation
    # Could use:
    # 1. Historical Sentinel-2 data (free)
    # 2. Google Popular Times (scraped)
    # 3. Foot traffic data (paid: SafeGraph, Placer.ai)
    
    baselines = {
        "WMT": {
            "weekly_avg_visits": 5000,
            "peak_day": "Saturday",
            "peak_hour": 14,
            "source": "placeholder",
        },
        "COST": {
            "weekly_avg_visits": 3000,
            "peak_day": "Saturday",
            "peak_hour": 12,
            "source": "placeholder",
        },
        "TGT": {
            "weekly_avg_visits": 4000,
            "peak_day": "Saturday",
            "peak_hour": 14,
            "source": "placeholder",
        },
    }
    
    return baselines.get(ticker, {"weekly_avg_visits": 0, "source": "unknown"})


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
