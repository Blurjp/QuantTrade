"""
Oil Tank Level Detection Algorithm
Analyzes floating roof tank shadows to estimate fill levels
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

try:
    import rasterio
    from rasterio.windows import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class OilTankDetector:
    """
    Detects oil storage tank fill levels from satellite imagery
    
    Method:
    1. Identify circular tank structures (Hough Circle detection)
    2. Measure shadow length on floating roof
    3. Estimate fill level based on shadow geometry
    4. Aggregate multiple tanks for total storage estimate
    """
    
    def __init__(self, min_tank_radius: int = 10, max_tank_radius: int = 100):
        """
        Initialize tank detector
        
        Args:
            min_tank_radius: Minimum tank radius in pixels
            max_tank_radius: Maximum tank radius in pixels
        """
        self.min_radius = min_tank_radius
        self.max_radius = max_tank_radius
        
    def detect_tanks_from_image(self, image: np.ndarray) -> Dict:
        """
        Detect circular tanks in an image
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Dict with tank locations and radii
        """
        if not CV2_AVAILABLE:
            return {"error": "OpenCV not installed", "tanks": []}
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        tanks = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                tanks.append({
                    "center_x": int(x),
                    "center_y": int(y),
                    "radius": int(r),
                    "area": np.pi * r * r
                })
        
        return {
            "total_tanks": len(tanks),
            "tanks": tanks
        }
    
    def estimate_tank_level(self, image: np.ndarray, tank: Dict) -> Dict:
        """
        Estimate fill level of a single tank based on shadow analysis
        
        Floating roof tanks have visible shadows when partially filled:
        - Empty: Roof at bottom, long shadow
        - Full: Roof at top, short/no shadow
        - Shadow direction depends on sun angle
        
        Args:
            image: RGB image array
            tank: Tank dict with center_x, center_y, radius
            
        Returns:
            Dict with estimated fill level
        """
        x, y, r = tank["center_x"], tank["center_y"], tank["radius"]
        
        # Extract tank region with padding
        pad = int(r * 0.5)
        x1 = max(0, x - r - pad)
        x2 = min(image.shape[1], x + r + pad)
        y1 = max(0, y - r - pad)
        y2 = min(image.shape[0], y + r + pad)
        
        tank_region = image[y1:y2, x1:x2]
        
        if tank_region.size == 0:
            return {"level_pct": None, "error": "Empty region"}
        
        # Convert to grayscale
        if len(tank_region.shape) == 3:
            gray = cv2.cvtColor(tank_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = tank_region
        
        # Simple shadow detection based on intensity
        # Darker pixels = shadow
        mean_intensity = np.mean(gray)
        dark_threshold = mean_intensity * 0.7
        
        # Count dark pixels (shadow)
        shadow_mask = gray < dark_threshold
        shadow_pct = np.sum(shadow_mask) / shadow_mask.size * 100
        
        # Estimate fill level
        # More shadow = lower fill level
        # This is a simplified model - real implementation would need calibration
        # with ground truth data from EIA
        
        # Invert shadow percentage to get fill level
        # 0% shadow ≈ 100% full
        # 50% shadow ≈ 50% full
        # 100% shadow ≈ 0% full
        
        fill_level = max(0, min(100, 100 - shadow_pct))
        
        return {
            "level_pct": round(fill_level, 1),
            "shadow_pct": round(shadow_pct, 1),
            "mean_intensity": round(mean_intensity, 1)
        }
    
    def analyze_storage_facility(self, image: np.ndarray) -> Dict:
        """
        Analyze all tanks in a storage facility
        
        Args:
            image: RGB image array
            
        Returns:
            Dict with aggregate storage estimates
        """
        # Detect all tanks
        detection = self.detect_tanks_from_image(image)
        
        if detection["total_tanks"] == 0:
            return {
                "total_tanks": 0,
                "avg_fill_level": None,
                "total_storage_estimate": None
            }
        
        # Analyze each tank
        tank_levels = []
        for tank in detection["tanks"]:
            level_info = self.estimate_tank_level(image, tank)
            tank["level_pct"] = level_info.get("level_pct")
            tank["shadow_pct"] = level_info.get("shadow_pct")
            
            if tank["level_pct"] is not None:
                tank_levels.append(tank["level_pct"])
        
        # Calculate aggregate statistics
        avg_level = np.mean(tank_levels) if tank_levels else None
        
        # Estimate total storage (simplified)
        # Real implementation would need tank capacity data
        total_area = sum(t["area"] for t in detection["tanks"])
        
        return {
            "total_tanks": detection["total_tanks"],
            "tanks_analyzed": len(tank_levels),
            "avg_fill_level": round(avg_level, 1) if avg_level else None,
            "total_tank_area_sq_pixels": round(total_area, 1),
            "tanks": detection["tanks"],
            "signal": self._generate_signal(avg_level)
        }
    
    def _generate_signal(self, avg_level: float) -> Dict:
        """
        Generate trading signal based on storage level
        
        Args:
            avg_level: Average fill level percentage
            
        Returns:
            Dict with signal and rationale
        """
        if avg_level is None:
            return {"direction": "neutral", "rationale": "Unable to determine level"}
        
        # Thresholds (these should be calibrated with historical data)
        # High inventory (>80%) = bearish (supply glut)
        # Low inventory (<40%) = bullish (supply tight)
        # Normal (40-80%) = neutral
        
        if avg_level > 80:
            return {
                "direction": "short",
                "rationale": f"High storage level ({avg_level:.1f}%) suggests supply glut"
            }
        elif avg_level < 40:
            return {
                "direction": "long",
                "rationale": f"Low storage level ({avg_level:.1f}%) suggests supply tightness"
            }
        else:
            return {
                "direction": "neutral",
                "rationale": f"Normal storage level ({avg_level:.1f}%)"
            }
    
    def detect_from_geotiff(self, tiff_path: str, bounds: Tuple[float, float, float, float] = None) -> Dict:
        """
        Detect tanks from GeoTIFF file
        
        Args:
            tiff_path: Path to GeoTIFF file
            bounds: Optional bounds to crop (minx, miny, maxx, maxy)
            
        Returns:
            Dict with detection results
        """
        if not RASTERIO_AVAILABLE:
            return {"error": "rasterio not installed"}
        
        with rasterio.open(tiff_path) as src:
            if bounds:
                window = from_bounds(*bounds, src.transform)
                image = src.read(window=window)
            else:
                image = src.read()
            
            # Convert to RGB
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            elif image.shape[0] > 3:
                image = np.transpose(image[:3], (1, 2, 0))
            
            # Normalize
            image = (image / image.max() * 255).astype(np.uint8)
        
        return self.analyze_storage_facility(image)


def analyze_cushing_storage(image_or_path) -> Dict:
    """
    Convenience function to analyze Cushing oil storage
    
    Args:
        image_or_path: Image array or path to GeoTIFF
        
    Returns:
        Dict with storage analysis
    """
    detector = OilTankDetector()
    
    if isinstance(image_or_path, str):
        return detector.detect_from_geotiff(image_or_path)
    else:
        return detector.analyze_storage_facility(image_or_path)


if __name__ == "__main__":
    print("⛽ Oil Tank Detection Module")
    print("="*60)
    print()
    
    if not CV2_AVAILABLE:
        print("❌ OpenCV not installed")
        print("   Install with: pip install opencv-python")
    else:
        print("✅ OpenCV available")
        print()
        print("Features:")
        print("  • Circular tank detection (Hough Circle)")
        print("  • Shadow-based fill level estimation")
        print("  • Aggregate storage analysis")
        print("  • Trading signal generation")
        print()
        print("Usage:")
        print("  from pipeline.tank_detection import OilTankDetector")
        print("  detector = OilTankDetector()")
        print("  result = detector.analyze_storage_facility(image)")
        print()
        print("⚠️  Note: This is a simplified algorithm.")
        print("   For production use, calibrate with EIA data.")
        print()
        print("Calibration steps:")
        print("  1. Get historical Cushing storage data from EIA")
        print("  2. Get corresponding Sentinel-2 images")
        print("  3. Compare detected levels with EIA data")
        print("  4. Adjust shadow thresholds")
        print("  5. Validate on out-of-sample data")
