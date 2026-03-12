"""
Detection modules for different monitoring types.

Each module implements a standard interface:
- detect(aoi_path, date, output_base) -> detections
"""

from pathlib import Path
from typing import Dict, List, Optional
import json


class DetectionResult:
    """Standard detection result format."""
    
    def __init__(
        self,
        detection_type: str,
        date: str,
        count: int,
        details: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None
    ):
        self.detection_type = detection_type
        self.date = date
        self.count = count
        self.details = details or []
        self.metadata = metadata or {}
    
    def to_dict(self) -> dict:
        return {
            "detection_type": self.detection_type,
            "date": self.date,
            "count": self.count,
            "details": self.details,
            "metadata": self.metadata,
        }


def detect_ships_sar(aoi_path: str, target_date: str, output_base: str = "outputs") -> DetectionResult:
    """
    Detect ships using Sentinel-1 SAR CFAR.
    
    Used for: chokepoints, ports
    """
    # Run detection
    # ... (existing logic)
    
    return DetectionResult(
        detection_type="ships_sar",
        date=target_date,
        count=0,  # Will be filled by actual detection
        metadata={"data_source": "Sentinel-1", "method": "CFAR"},
    )


def detect_vehicles_optical(aoi_path: str, target_date: str, output_base: str = "outputs") -> DetectionResult:
    """
    Detect vehicles in parking lots using optical imagery.
    
    Used for: retail_parking, auto_inventory
    
    Note: Requires cloud-free Sentinel-2 or commercial imagery.
    """
    # TODO: Implement vehicle detection
    # Options:
    # 1. YOLO model trained on parking lot imagery
    # 2. Simple brightness threshold on asphalt
    # 3. Pre-trained object detection model
    
    detections = []
    
    # Placeholder: Load Sentinel-2 and run detection
    # In production, this would:
    # 1. Load Sentinel-2 scene for AOI
    # 2. Apply vehicle detection model
    # 3. Filter by parking lot polygons
    # 4. Count vehicles per lot
    
    return DetectionResult(
        detection_type="vehicles_optical",
        date=target_date,
        count=len(detections),
        details=detections,
        metadata={
            "data_source": "Sentinel-2",
            "method": "YOLOv8",
            "note": "Requires cloud-free imagery",
        },
    )


def detect_tank_levels(aoi_path: str, target_date: str, output_base: str = "outputs") -> DetectionResult:
    """
    Detect oil storage tank fill levels from floating roof position.
    
    Used for: oil_storage
    
    Method: 
    1. Identify circular tanks in AOI
    2. Measure shadow length on SAR/optical
    3. Shadow length correlates with roof height
    4. Roof height indicates fill level
    """
    from pipeline.detection_storage import analyze_cushing_storage
    
    # Run the actual storage analysis
    storage_result = analyze_cushing_storage(
        aoi_path=aoi_path,
        target_date=target_date,
        output_base=output_base,
        compare_eia=False,
    )
    
    fill_pct = storage_result.get("fill_pct", 0)
    tanks_detected = storage_result.get("tanks_detected", 0)
    status = storage_result.get("status", "unknown")
    
    details = [{
        "fill_pct": fill_pct,
        "tanks_detected": tanks_detected,
        "estimated_barrels": storage_result.get("estimated_barrels", 0),
        "method": storage_result.get("method", "unknown"),
    }]
    
    return DetectionResult(
        detection_type="tank_levels",
        date=target_date,
        count=tanks_detected,
        details=details,
        metadata={
            "data_source": storage_result.get("data_source", "sentinel"),
            "method": "shadow_analysis",
            "status": status,
            "scene_id": storage_result.get("scene_id"),
            "scene_date": storage_result.get("scene_date"),
        },
    )


def detect_crop_health(aoi_path: str, target_date: str, output_base: str = "outputs") -> DetectionResult:
    """
    Detect crop health using vegetation indices.
    
    Used for: agricultural
    
    Method:
    1. Calculate NDVI from Sentinel-2 red/NIR bands
    2. Compare to historical baseline
    3. Anomaly indicates yield deviation
    """
    from pipeline.detection_agriculture import process_sentinel2_for_ndvi

    result = process_sentinel2_for_ndvi(
        aoi_path=aoi_path,
        target_date=target_date,
        output_base=output_base,
    )

    ndvi_stats = result.get("ndvi_stats", {})
    health_distribution = result.get("health_distribution", {})
    valid_pixels = ndvi_stats.get("valid_pixels", 0)

    metadata = {
        "data_source": "Sentinel-2",
        "method": "NDVI",
        "seasonal": True,
        "status": result.get("status", "unknown"),
    }
    if "cloud_cover" in result:
        metadata["cloud_cover"] = result["cloud_cover"]
    if "scene_id" in result:
        metadata["scene_id"] = result["scene_id"]
    if result.get("status") != "success":
        metadata["message"] = result.get("message") or result.get("error")

    details = [{
        "ndvi_mean": ndvi_stats.get("mean", 0.0),
        "ndvi_median": ndvi_stats.get("median", 0.0),
        "ndvi_std": ndvi_stats.get("std", 0.0),
        "valid_pixels": valid_pixels,
        "healthy_share": health_distribution.get("healthy", 0.0),
        "very_healthy_share": health_distribution.get("very_healthy", 0.0),
        "aoi": result.get("aoi"),
    }]

    return DetectionResult(
        detection_type="crop_health",
        date=target_date,
        count=max(1, int(valid_pixels > 0)),
        details=details,
        metadata=metadata,
    )


def detect_containers_port(aoi_path: str, target_date: str, output_base: str = "outputs") -> DetectionResult:
    """
    Detect shipping containers and cranes at ports.
    
    Used for: port_logistics
    
    Method:
    1. SAR for ship detection (existing)
    2. Optical for container stack height
    3. Crane activity detection
    """
    detections = []
    
    # TODO: Implement port activity detection
    # In production:
    # 1. Count ships in anchorage vs berthed
    # 2. Estimate container stack density
    # 3. Detect crane positions (working vs idle)
    
    return DetectionResult(
        detection_type="port_activity",
        date=target_date,
        count=len(detections),
        details=detections,
        metadata={
            "data_source": "Sentinel-1 + Sentinel-2",
            "method": "multi_sensor",
        },
    )


# Registry of detection functions by monitoring type
DETECTION_REGISTRY = {
    "chokepoint": detect_ships_sar,
    "port_logistics": detect_containers_port,
    "retail_parking": detect_vehicles_optical,
    "auto_inventory": detect_vehicles_optical,
    "oil_storage": detect_tank_levels,
    "agricultural": detect_crop_health,
    "agriculture": detect_crop_health,
}


def run_detection(
    monitoring_type: str,
    aoi_path: str,
    target_date: str,
    output_base: str = "outputs"
) -> DetectionResult:
    """
    Run appropriate detection based on monitoring type.
    
    Args:
        monitoring_type: Type of monitoring (chokepoint, retail_parking, etc.)
        aoi_path: Path to AOI GeoJSON
        target_date: Date to analyze
        output_base: Output directory
    
    Returns:
        DetectionResult with counts and details
    """
    detector = DETECTION_REGISTRY.get(monitoring_type)
    
    if detector is None:
        raise ValueError(f"Unknown monitoring type: {monitoring_type}")
    
    return detector(aoi_path, target_date, output_base)


if __name__ == "__main__":
    # Test detection registry
    print("Detection Registry:")
    for mtype, func in DETECTION_REGISTRY.items():
        print(f"  {mtype}: {func.__name__}")
