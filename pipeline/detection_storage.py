"""
Oil storage tank level detection from satellite imagery.

Detects floating roof tank levels by analyzing:
1. Shadow cast by tank roof
2. Roof position relative to tank rim
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import date
import cv2


class TankLevelDetector:
    """
    Detect oil storage tank fill levels from satellite imagery.
    
    Floating roof tanks have a roof that moves up/down with oil level.
    When tank is full: roof near top, short shadow
    When tank is empty: roof near bottom, long shadow
    
    Method:
    1. Detect circular tanks using Hough transform
    2. Measure shadow length (if optical) or roof position (if SAR)
    3. Convert to fill percentage
    """
    
    def __init__(self, min_tank_radius: int = 10, max_tank_radius: int = 100):
        """
        Initialize tank detector.
        
        Args:
            min_tank_radius: Minimum tank radius in pixels
            max_tank_radius: Maximum tank radius in pixels
        """
        self.min_radius = min_tank_radius
        self.max_radius = max_tank_radius
    
    def detect_tanks_hough(self, image: np.ndarray) -> List[Dict]:
        """
        Detect circular tanks using Hough Circle Transform.
        
        Args:
            image: Grayscale image
        
        Returns:
            List of detected tanks with positions and radii
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Hough Circle Transform
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=gray.shape[0] // 20,
            param1=50,  # Edge detection threshold
            param2=30,  # Accumulator threshold
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )
        
        tanks = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i, (x, y, r) in enumerate(circles[0, :]):
                tanks.append({
                    "id": i,
                    "center": (int(x), int(y)),
                    "radius": int(r),
                    "area_pixels": np.pi * r * r,
                })
        
        return tanks
    
    def measure_shadow_length(
        self,
        image: np.ndarray,
        tank: Dict,
        sun_direction: Tuple[float, float] = (1, 0),
    ) -> float:
        """
        Measure shadow length cast by tank roof.
        
        Args:
            image: Grayscale image
            tank: Tank dictionary from detect_tanks_hough
            sun_direction: Direction of sunlight (azimuth)
        
        Returns:
            Shadow length in pixels
        """
        cx, cy = tank["center"]
        r = tank["radius"]
        
        # Extract region around tank
        margin = int(r * 0.5)
        x1 = max(0, cx - r - margin)
        x2 = min(image.shape[1], cx + r + margin)
        y1 = max(0, cy - r - margin)
        y2 = min(image.shape[0], cy + r + margin)
        
        roi = image[y1:y2, x1:x2]
        
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            roi_gray = roi
        
        # Detect dark regions (shadows) in sun direction
        # Shadows are darker than surrounding area
        threshold = np.percentile(roi_gray, 20)
        shadows = roi_gray < threshold
        
        # Measure shadow extent in sun direction
        # This is simplified - real implementation would use sun azimuth
        shadow_pixels = np.sum(shadows)
        
        # Estimate shadow length
        shadow_length = np.sqrt(shadow_pixels) / 2  # Rough estimate
        
        return float(shadow_length)
    
    def shadow_to_fill_level(
        self,
        shadow_length: float,
        tank_radius: float,
        tank_height: float = 15.0,  # meters, typical
        sun_elevation: float = 45.0,  # degrees
    ) -> float:
        """
        Convert shadow length to fill level percentage.
        
        Args:
            shadow_length: Measured shadow length in pixels
            tank_radius: Tank radius in pixels
            tank_height: Tank height in meters
            sun_elevation: Sun elevation angle in degrees
        
        Returns:
            Fill level percentage (0-100)
        """
        # Simplified geometry:
        # shadow_length = roof_height / tan(sun_elevation)
        # roof_height = (1 - fill_level) * tank_height
        
        # Invert to get fill level
        # fill_level = 1 - (shadow_length * tan(sun_elevation) / tank_height)
        
        # Normalize shadow length to tank radius
        normalized_shadow = shadow_length / tank_radius
        
        # Map to fill level (simplified linear model)
        # Long shadow = empty, short shadow = full
        # This needs calibration with real data
        
        fill_level = max(0, min(100, 100 * (1 - normalized_shadow / 2)))
        
        return fill_level
    
    def analyze_tank_farm(
        self,
        image: np.ndarray,
        sun_azimuth: float = 180.0,
        sun_elevation: float = 45.0,
    ) -> Dict:
        """
        Analyze all tanks in an image.
        
        Args:
            image: Satellite image of tank farm
            sun_azimuth: Sun azimuth angle
            sun_elevation: Sun elevation angle
        
        Returns:
            Dictionary with tank analysis results
        """
        # Detect tanks
        tanks = self.detect_tanks_hough(image)
        
        results = {
            "num_tanks": len(tanks),
            "tanks": [],
            "aggregate": {
                "total_capacity_proxy": 0,
                "total_fill_proxy": 0,
                "average_fill": 0,
            },
        }
        
        total_fill = 0
        
        for tank in tanks:
            # Measure shadow
            shadow = self.measure_shadow_length(image, tank)
            
            # Convert to fill level
            fill_level = self.shadow_to_fill_level(
                shadow, tank["radius"], sun_elevation=sun_elevation
            )
            
            tank_result = {
                "id": tank["id"],
                "center": tank["center"],
                "radius": tank["radius"],
                "shadow_length": shadow,
                "fill_level": fill_level,
            }
            
            results["tanks"].append(tank_result)
            total_fill += fill_level
        
        # Calculate aggregates
        if tanks:
            results["aggregate"]["average_fill"] = total_fill / len(tanks)
            results["aggregate"]["total_capacity_proxy"] = sum(
                t["radius"] ** 2 for t in tanks
            )
            results["aggregate"]["total_fill_proxy"] = sum(
                t["radius"] ** 2 * results["tanks"][i]["fill_level"] / 100
                for i, t in enumerate(tanks)
            )
        
        return results


def analyze_cushing_storage(
    aoi_path: str,
    target_date: str,
    output_base: str = "outputs",
    compare_eia: bool = True,
) -> Dict:
    """
    Analyze Cushing oil storage levels.
    
    Args:
        aoi_path: Path to Cushing AOI GeoJSON
        target_date: Date to analyze
        output_base: Output directory
        compare_eia: Compare results to EIA weekly report
    
    Returns:
        Dictionary with storage analysis
    """
    # TODO: Load Sentinel-1/Landsat imagery
    # TODO: Run tank detection
    # TODO: Calculate aggregate fill level
    # TODO: Compare to EIA data if available
    
    results = {
        "date": target_date,
        "location": "Cushing, OK",
        "status": "pending",
        "tanks_detected": 0,
        "average_fill_level": 0,
        "estimated_barrels": 0,
        "eia_comparison": None,
        "note": "Requires satellite imagery integration",
    }
    
    # Placeholder EIA comparison
    if compare_eia:
        results["eia_comparison"] = {
            "eia_reported_level": 0,
            "satellite_estimated_level": 0,
            "difference": 0,
            "note": "EIA data needs to be fetched from API",
        }
    
    # Save results
    output_path = Path(output_base) / target_date / "storage"
    output_path.mkdir(parents=True, exist_ok=True)
    
    result_file = output_path / "cushing_levels.json"
    result_file.write_text(json.dumps(results, indent=2))
    
    return results


def get_eia_storage_data(api_key: str = None) -> Dict:
    """
    Fetch EIA weekly storage data for comparison.
    
    EIA API: https://api.eia.gov/v2/petroleum/stor/wpy/
    """
    # TODO: Implement EIA API integration
    # Free API key available from eia.gov
    
    return {
        "source": "EIA",
        "status": "not_implemented",
        "note": "Requires EIA API key",
    }


if __name__ == "__main__":
    print("Oil Storage Tank Level Detection")
    print("=" * 40)
    print()
    print("Method:")
    print("  1. Detect circular tanks using Hough transform")
    print("  2. Measure shadow cast by floating roof")
    print("  3. Convert shadow length to fill level")
    print()
    print("Usage:")
    print("  from pipeline.detection_storage import TankLevelDetector")
    print("  detector = TankLevelDetector()")
    print("  results = detector.analyze_tank_farm(image)")
    print()
    print("Data sources:")
    print("  - Sentinel-1 SAR (cloud-independent)")
    print("  - Landsat 8/9 (free, 30m)")
    print("  - Maxar/Planet (high-res, paid)")
