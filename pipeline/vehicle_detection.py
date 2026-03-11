"""
Vehicle Detection Module using YOLOv8
Detects and counts vehicles in satellite/aerial imagery for retail parking analysis
"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  ultralytics not installed. Install with: pip install ultralytics")

try:
    import rasterio
    from rasterio.windows import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


class VehicleDetector:
    """
    Vehicle detection using YOLOv8 for parking lot analysis
    
    Note: YOLOv8 is trained on ground-level imagery, not satellite.
    For satellite imagery, this provides a rough estimate and should be
    calibrated with ground truth data or used with high-resolution imagery
    (Planet 3m or better).
    """
    
    def __init__(self, model_size: str = "n", confidence: float = 0.25):
        """
        Initialize vehicle detector
        
        Args:
            model_size: YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge)
            confidence: Detection confidence threshold
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed")
        
        self.model = YOLO(f"yolov8{model_size}.pt")
        self.confidence = confidence
        
        # Vehicle classes in COCO dataset
        self.vehicle_classes = {
            2: "car",
            3: "motorcycle", 
            5: "bus",
            7: "truck"
        }
    
    def detect_from_image(self, image_path: str) -> Dict:
        """
        Detect vehicles in an image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict with detection results
        """
        # Run inference
        results = self.model(image_path, conf=self.confidence, verbose=False)
        
        # Parse results
        vehicles = []
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                
                # Only count vehicle classes
                if cls_id in self.vehicle_classes:
                    vehicles.append({
                        "class": self.vehicle_classes[cls_id],
                        "confidence": float(boxes.conf[i]),
                        "bbox": boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
                    })
        
        # Summary
        car_count = sum(1 for v in vehicles if v["class"] == "car")
        truck_count = sum(1 for v in vehicles if v["class"] == "truck")
        bus_count = sum(1 for v in vehicles if v["class"] == "bus")
        motorcycle_count = sum(1 for v in vehicles if v["class"] == "motorcycle")
        
        return {
            "total_vehicles": len(vehicles),
            "cars": car_count,
            "trucks": truck_count,
            "buses": bus_count,
            "motorcycles": motorcycle_count,
            "detections": vehicles
        }
    
    def detect_from_geotiff(self, tiff_path: str, bounds: Tuple[float, float, float, float] = None) -> Dict:
        """
        Detect vehicles in a GeoTIFF file
        
        Args:
            tiff_path: Path to GeoTIFF file
            bounds: Optional bounds (minx, miny, maxx, maxy) to crop
            
        Returns:
            Dict with detection results
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError("rasterio not installed")
        
        # Read GeoTIFF
        with rasterio.open(tiff_path) as src:
            if bounds:
                # Crop to bounds
                window = from_bounds(*bounds, src.transform)
                image = src.read(window=window)
            else:
                image = src.read()
            
            # Convert to RGB format expected by YOLO
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            elif image.shape[0] > 3:
                image = np.transpose(image[:3], (1, 2, 0))
            
            # Normalize to 0-255
            image = (image / image.max() * 255).astype(np.uint8)
        
        # Run detection
        results = self.model(image, conf=self.confidence, verbose=False)
        
        # Parse results (same as detect_from_image)
        vehicles = []
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                
                if cls_id in self.vehicle_classes:
                    vehicles.append({
                        "class": self.vehicle_classes[cls_id],
                        "confidence": float(boxes.conf[i]),
                        "bbox": boxes.xyxy[i].tolist()
                    })
        
        return {
            "total_vehicles": len(vehicles),
            "cars": sum(1 for v in vehicles if v["class"] == "car"),
            "trucks": sum(1 for v in vehicles if v["class"] == "truck"),
            "buses": sum(1 for v in vehicles if v["class"] == "bus"),
            "motorcycles": sum(1 for v in vehicles if v["class"] == "motorcycle"),
            "detections": vehicles
        }
    
    def analyze_parking_lot(self, detection_result: Dict, total_spots: int = None) -> Dict:
        """
        Analyze parking lot occupancy from detection results
        
        Args:
            detection_result: Output from detect_from_image or detect_from_geotiff
            total_spots: Total number of parking spots (if known)
            
        Returns:
            Dict with parking lot analysis
        """
        vehicle_count = detection_result["total_vehicles"]
        
        # Estimate occupancy
        # Note: This is approximate because:
        # 1. YOLO may miss vehicles (especially in satellite imagery)
        # 2. Large vehicles may count as multiple spots
        # 3. Some spots may be occupied by non-vehicle objects
        
        if total_spots:
            occupancy_pct = (vehicle_count / total_spots) * 100
            occupancy_pct = min(occupancy_pct, 100)  # Cap at 100%
        else:
            # Without total spots, we can only count vehicles
            occupancy_pct = None
        
        # Signal generation
        # High occupancy (>80%) = bullish for retail sales
        # Low occupancy (<40%) = bearish for retail sales
        
        if occupancy_pct is not None:
            if occupancy_pct > 80:
                signal = "long"
                rationale = "High parking lot occupancy suggests strong customer traffic"
            elif occupancy_pct < 40:
                signal = "short"
                rationale = "Low parking lot occupancy suggests weak customer traffic"
            else:
                signal = "neutral"
                rationale = "Normal parking lot occupancy"
        else:
            signal = "neutral"
            rationale = "Cannot determine occupancy without total spot count"
        
        return {
            "vehicle_count": vehicle_count,
            "total_spots": total_spots,
            "occupancy_pct": round(occupancy_pct, 1) if occupancy_pct else None,
            "signal": signal,
            "rationale": rationale,
            "detection_confidence": np.mean([v["confidence"] for v in detection_result["detections"]]) if detection_result["detections"] else 0
        }


def detect_vehicles_in_parking_lot(image_path: str, model_size: str = "n") -> Dict:
    """
    Convenience function to detect vehicles in a parking lot image
    
    Args:
        image_path: Path to image file
        model_size: YOLO model size
        
    Returns:
        Dict with vehicle count and analysis
    """
    if not YOLO_AVAILABLE:
        return {
            "error": "ultralytics not installed",
            "total_vehicles": 0,
            "signal": "neutral"
        }
    
    detector = VehicleDetector(model_size=model_size)
    detections = detector.detect_from_image(image_path)
    analysis = detector.analyze_parking_lot(detections)
    
    return {
        **detections,
        **analysis
    }


def batch_detect_vehicles(image_paths: List[str], model_size: str = "n") -> List[Dict]:
    """
    Detect vehicles in multiple images
    
    Args:
        image_paths: List of image file paths
        model_size: YOLO model size
        
    Returns:
        List of detection results
    """
    if not YOLO_AVAILABLE:
        return [{"error": "ultralytics not installed"} for _ in image_paths]
    
    detector = VehicleDetector(model_size=model_size)
    
    results = []
    for path in image_paths:
        try:
            detections = detector.detect_from_image(path)
            analysis = detector.analyze_parking_lot(detections)
            results.append({
                "image": path,
                **detections,
                **analysis
            })
        except Exception as e:
            results.append({
                "image": path,
                "error": str(e)
            })
    
    return results


if __name__ == "__main__":
    # Test vehicle detection
    print("🚗 Testing Vehicle Detection Module")
    print("="*60)
    
    if not YOLO_AVAILABLE:
        print("❌ ultralytics not installed")
        print("   Install with: pip install ultralytics")
    else:
        print("✅ YOLOv8 available")
        print()
        print("Note: YOLOv8 is trained on ground-level imagery.")
        print("For satellite imagery, use high-resolution data (Planet 3m or better)")
        print("or calibrate with ground truth data.")
        print()
        print("To test:")
        print("  1. Download a sample image:")
        print("     python -c \"from pipeline.vehicle_detection import detect_vehicles_in_parking_lot; print(detect_vehicles_in_parking_lot('path/to/image.jpg'))\"")
        print()
        print("  2. Or use with Sentinel-2 data:")
        print("     from pipeline.vehicle_detection import VehicleDetector")
        print("     detector = VehicleDetector()")
        print("     results = detector.detect_from_geotiff('sentinel2.tif')")
