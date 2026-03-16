"""
Soil Moisture Monitoring Module

Uses satellite data to monitor soil moisture levels for predicting crop yields,
drought conditions, and irrigation needs.

Data Source:
- SMAP (Soil Moisture Active Passive): Surface soil moisture
- Sentinel-1: Radar-based soil moisture
- Available via NASA/Planetary Computer (free)
- Update frequency: Daily
- Latency: 1-3 days
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SoilMoistureMonitor:
    """Monitor soil moisture for commodity trading signals."""
    
    def __init__(
        self,
        output_base: str = "outputs",
        cache_days: int = 30
    ):
        """
        Initialize soil moisture monitor.
        
        Args:
            output_base: Base directory for outputs
            cache_days: Number of days to cache data
        """
        self.output_base = Path(output_base)
        self.cache_days = cache_days
        
        # Key agricultural regions for monitoring
        self.regions = {
            # USA - Major Crop Regions
            "usa_midwest": {
                "name": "US Midwest Corn Belt",
                "bbox": [-100.0, 36.0, -82.0, 48.0],
                "country": "USA",
                "type": "row_crops",
                "instruments": ["CORN", "SOYB", "WEAT"],
                "description": "Iowa, Illinois, Indiana corn/soybeans",
                "baseline_moisture": 0.25,  # m³/m³ volumetric
                "critical_months": [4, 5, 6, 7, 8, 9],
                "crops": ["corn", "soybeans"],
                "soil_type": "loam"
            },
            "usa_great_plains": {
                "name": "US Great Plains Wheat",
                "bbox": [-105.0, 32.0, -95.0, 45.0],
                "country": "USA",
                "type": "row_crops",
                "instruments": ["WEAT", "KWK", "SORGHUM"],
                "description": "Kansas, Nebraska, Dakotas wheat",
                "baseline_moisture": 0.18,
                "critical_months": [3, 4, 5, 6, 7],
                "crops": ["winter_wheat", "spring_wheat", "sorghum"],
                "soil_type": "sandy_loam"
            },
            
            # South America
            "brazil_central": {
                "name": "Brazil Central Soybeans",
                "bbox": [-58.0, -20.0, -45.0, -10.0],
                "country": "Brazil",
                "type": "row_crops",
                "instruments": ["SOYB", "CORN", "COTTON"],
                "description": "Mato Grosso, Goiás soybeans",
                "baseline_moisture": 0.28,
                "critical_months": [10, 11, 12, 1, 2, 3],
                "crops": ["soybeans", "corn", "cotton"],
                "soil_type": "clay_loam"
            },
            "argentina_pampas": {
                "name": "Argentina Pampas",
                "bbox": [-65.0, -40.0, -56.0, -30.0],
                "country": "Argentina",
                "type": "row_crops",
                "instruments": ["SOYB", "CORN", "WEAT"],
                "description": "Buenos Aires, Santa Fe crops",
                "baseline_moisture": 0.22,
                "critical_months": [10, 11, 12, 1, 2, 3],
                "crops": ["soybeans", "corn", "wheat"],
                "soil_type": "loam"
            },
            
            # Europe
            "europe_central": {
                "name": "Central European Plains",
                "bbox": [-5.0, 42.0, 30.0, 55.0],
                "country": "Multiple",
                "type": "row_crops",
                "instruments": ["WEAT", "EXI1", "CORN"],
                "description": "France, Germany, Poland crops",
                "baseline_moisture": 0.24,
                "critical_months": [4, 5, 6, 7, 8],
                "crops": ["wheat", "barley", "corn"],
                "soil_type": "loam"
            },
            
            # Asia
            "india_gangetic": {
                "name": "India Gangetic Plain",
                "bbox": [73.0, 22.0, 92.0, 32.0],
                "country": "India",
                "type": "row_crops",
                "instruments": ["RICE", "WHEAT", "SUGAR"],
                "description": "Punjab, Haryana, UP crops",
                "baseline_moisture": 0.20,
                "critical_months": [6, 7, 8, 9, 10, 11, 12, 1, 2, 3],
                "crops": ["rice", "wheat", "sugarcane"],
                "soil_type": "alluvial"
            },
            "china_north_plain": {
                "name": "China North Plain",
                "bbox": [110.0, 32.0, 122.0, 42.0],
                "country": "China",
                "type": "row_crops",
                "instruments": ["FXI", "WEAT", "CORN"],
                "description": "Henan, Shandong, Hebei wheat/corn",
                "baseline_moisture": 0.18,
                "critical_months": [3, 4, 5, 6, 7, 8, 9, 10],
                "crops": ["wheat", "corn", "cotton"],
                "soil_type": "loam"
            },
            
            # Africa
            "africa_sahel": {
                "name": "Africa Sahel Belt",
                "bbox": [-18.0, 12.0, 30.0, 20.0],
                "country": "Multiple",
                "type": "rainfed",
                "instruments": ["COTTON", "COCOA", "SHEA"],
                "description": "Sahel agricultural zone",
                "baseline_moisture": 0.12,
                "critical_months": [6, 7, 8, 9, 10],
                "crops": ["millet", "sorghum", "cotton"],
                "soil_type": "sandy"
            },
            
            # Australia
            "australia_wheat": {
                "name": "Australia Wheat Belt",
                "bbox": [115.0, -35.0, 152.0, -25.0],
                "country": "Australia",
                "type": "row_crops",
                "instruments": ["WEAT", "AWB", "BARLEY"],
                "description": "Western Australia wheat",
                "baseline_moisture": 0.15,
                "critical_months": [5, 6, 7, 8, 9, 10],
                "crops": ["wheat", "barley", "canola"],
                "soil_type": "sandy_loam"
            },
        }
        
        # Create output directory
        self.output_dir = self.output_base / "soil_moisture"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_soil_moisture_data(self, region_id: str, date: str) -> Optional[Dict]:
        """
        Fetch soil moisture data for a region.
        
        In production, this would use NASA SMAP API.
        For now, returns simulated data based on realistic patterns.
        
        Args:
            region_id: Region identifier
            date: Date string (YYYY-MM-DD)
            
        Returns:
            Dictionary with soil moisture metrics
        """
        region = self.regions.get(region_id)
        if not region:
            logger.error(f"Unknown region: {region_id}")
            return None
        
        logger.info(f"Fetching soil moisture data for {region_id} on {date}")
        
        # Simulate realistic soil moisture data
        # In production: use NASA SMAP API
        np.random.seed(hash(date + region_id) % 2**32)
        
        # Get baseline moisture
        baseline = region["baseline_moisture"]
        
        # Add seasonal variation
        month = datetime.strptime(date, "%Y-%m-%d").month
        day_of_year = datetime.strptime(date, "%Y-%m-%d").timetuple().tm_yday
        
        # Determine if in critical growing season
        is_critical_season = month in region["critical_months"]
        
        # Seasonal factor (higher in rainy season)
        if region_id in ["australia_wheat"]:
            # Southern hemisphere - opposite seasons
            seasonal_factor = 0.3 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
        else:
            # Northern hemisphere
            seasonal_factor = 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Add drought/flood variation
        drought_factor = np.random.uniform(-0.4, 0.3)
        
        # Random daily variation
        daily_noise = np.random.normal(0, 0.02)
        
        # Calculate actual soil moisture
        moisture = baseline * (1 + seasonal_factor + drought_factor) + daily_noise
        moisture = max(0.05, min(0.45, moisture))
        
        # Calculate anomaly
        moisture_anomaly = moisture - baseline
        moisture_anomaly_pct = (moisture - baseline) / baseline * 100
        
        # Determine soil moisture status
        if moisture_anomaly_pct < -40:
            status = "severe_drought"
        elif moisture_anomaly_pct < -25:
            status = "drought"
        elif moisture_anomaly_pct < -15:
            status = "dry"
        elif moisture_anomaly_pct < -5:
            status = "slightly_dry"
        elif moisture_anomaly_pct > 25:
            status = "waterlogged"
        elif moisture_anomaly_pct > 15:
            status = "wet"
        elif moisture_anomaly_pct > 5:
            status = "optimal"
        else:
            status = "normal"
        
        # Calculate plant available water (PAW)
        # PAW = (current - wilting_point) / (field_capacity - wilting_point)
        wilting_point = 0.10  # Typical for loam
        field_capacity = 0.30  # Typical for loam
        paw = (moisture - wilting_point) / (field_capacity - wilting_point)
        paw = max(0, min(1, paw))
        
        # Calculate impact score (0-100)
        # During critical season, deviations have more impact
        if is_critical_season:
            impact_multiplier = 1.5
        else:
            impact_multiplier = 0.7
        
        impact_score = min(100, abs(moisture_anomaly_pct) * impact_multiplier)
        
        # Root zone soil moisture (0-1m depth, typically lower than surface)
        root_zone_moisture = moisture * 0.85 + np.random.normal(0, 0.01)
        root_zone_moisture = max(0.05, min(0.40, root_zone_moisture))
        
        # Calculate irrigation need (0-100)
        if paw < 0.5:
            irrigation_need = (0.5 - paw) * 200
        else:
            irrigation_need = 0
        
        return {
            "region_id": region_id,
            "region_name": region["name"],
            "region_type": region["type"],
            "country": region["country"],
            "date": date,
            "month": month,
            "surface_moisture": round(moisture, 3),  # m³/m³
            "root_zone_moisture": round(root_zone_moisture, 3),
            "baseline_moisture": baseline,
            "moisture_anomaly": round(moisture_anomaly, 3),
            "moisture_anomaly_pct": round(moisture_anomaly_pct, 2),
            "status": status,
            "plant_available_water": round(paw, 3),  # 0-1 fraction
            "irrigation_need": round(irrigation_need, 1),  # 0-100
            "is_critical_season": is_critical_season,
            "impact_score": round(impact_score, 1),
            "soil_type": region["soil_type"],
            "crops": region["crops"],
            "data_source": "SMAP_Sentinel1",
            "satellites": ["SMAP", "Sentinel-1"],
            "quality": "good" if np.random.random() > 0.1 else "degraded"
        }
    
    def calculate_baseline(self, region_id: str, days: int = 90) -> Dict:
        """
        Calculate baseline soil moisture for a region.
        
        Args:
            region_id: Region identifier
            days: Number of days for baseline calculation
            
        Returns:
            Dictionary with baseline metrics
        """
        logger.info(f"Calculating {days}-day baseline for {region_id}")
        
        # Fetch historical data
        end_date = datetime.now()
        historical_moisture = []
        historical_anomaly = []
        
        for i in range(days):
            date = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
            data = self.fetch_soil_moisture_data(region_id, date)
            if data and data["quality"] == "good":
                historical_moisture.append(data["surface_moisture"])
                historical_anomaly.append(data["moisture_anomaly_pct"])
        
        if not historical_moisture:
            return {"error": "No valid historical data"}
        
        # Calculate baseline statistics
        baseline = {
            "region_id": region_id,
            "period_days": len(historical_moisture),
            "moisture": {
                "mean": round(np.mean(historical_moisture), 3),
                "std": round(np.std(historical_moisture), 3),
                "median": round(np.median(historical_moisture), 3),
            },
            "anomaly": {
                "mean": round(np.mean(historical_anomaly), 2),
                "std": round(np.std(historical_anomaly), 2),
                "median": round(np.median(historical_anomaly), 2),
            }
        }
        
        return baseline
    
    def detect_anomaly(
        self,
        current_data: Dict,
        baseline: Dict,
        threshold_std: float = 2.0
    ) -> Dict:
        """
        Detect anomalies in soil moisture.
        
        Args:
            current_data: Current moisture data
            baseline: Baseline statistics
            threshold_std: Number of standard deviations for anomaly
            
        Returns:
            Dictionary with anomaly detection results
        """
        # Calculate z-scores
        moisture_z = (current_data["surface_moisture"] - baseline["moisture"]["mean"]) / \
                     baseline["moisture"]["std"] if baseline["moisture"]["std"] > 0 else 0
        
        # Determine anomaly status
        moisture_anomaly = "significant" if abs(moisture_z) > threshold_std else \
                          "moderate" if abs(moisture_z) > 1.5 else "none"
        
        return {
            "moisture_z_score": round(moisture_z, 2),
            "moisture_anomaly": moisture_anomaly,
            "moisture_deviation_pct": round((current_data["surface_moisture"] - 
                                            baseline["moisture"]["mean"]) / 
                                            baseline["moisture"]["mean"] * 100, 2),
            "combined_z_score": round(abs(moisture_z), 2),
            "overall_anomaly": "significant" if abs(moisture_z) > 2.0 else \
                              "moderate" if abs(moisture_z) > 1.5 else "none"
        }
    
    def generate_signal(
        self,
        region_id: str,
        date: Optional[str] = None,
        baseline_days: int = 90
    ) -> Dict:
        """
        Generate trading signal for a region.
        
        Args:
            region_id: Region identifier
            date: Date for signal (default: today)
            baseline_days: Days for baseline calculation
            
        Returns:
            Dictionary with signal information
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        region = self.regions.get(region_id)
        if not region:
            return {"error": f"Unknown region: {region_id}"}
        
        logger.info(f"Generating signal for {region_id} on {date}")
        
        # Fetch current data
        current_data = self.fetch_soil_moisture_data(region_id, date)
        if not current_data:
            return {"error": "Failed to fetch current data"}
        
        # Calculate baseline
        baseline = self.calculate_baseline(region_id, baseline_days)
        if "error" in baseline:
            return {"error": baseline["error"]}
        
        # Detect anomaly
        anomaly = self.detect_anomaly(current_data, baseline)
        
        # Generate signal based on moisture status
        status = current_data["status"]
        is_critical = current_data["is_critical_season"]
        
        if status in ["severe_drought", "drought"]:
            direction = "short"
            confidence = min(100, 65 + current_data["impact_score"] * 0.5)
            if is_critical:
                confidence = min(100, confidence * 1.2)
            rationale = f"Severe soil moisture deficit in {region['name']}. {current_data['moisture_anomaly_pct']:.1f}% below normal. Crop stress likely."
        
        elif status == "dry":
            if is_critical:
                direction = "short"
                confidence = 62
                rationale = f"Dry soil conditions during critical period. {current_data['moisture_anomaly_pct']:.1f}% below normal. Irrigation needed."
            else:
                direction = "neutral"
                confidence = 50
                rationale = f"Dry soil outside critical period. Limited yield impact expected."
        
        elif status == "slightly_dry":
            direction = "neutral"
            confidence = 50
            rationale = f"Slightly dry soil. {current_data['moisture_anomaly_pct']:.1f}% below normal. Monitor conditions."
        
        elif status == "optimal":
            direction = "long"
            confidence = min(100, 58 + abs(current_data["moisture_anomaly_pct"]) * 0.5)
            rationale = f"Optimal soil moisture in {region['name']}. {current_data['moisture_anomaly_pct']:+.1f}% from normal. Excellent growing conditions."
        
        elif status == "normal":
            direction = "neutral"
            confidence = 50
            rationale = f"Normal soil moisture in {region['name']}. {current_data['moisture_anomaly_pct']:+.1f}% from baseline. Expected yields."
        
        elif status == "wet":
            if is_critical:
                direction = "short"
                confidence = 58
                rationale = f"Excess soil moisture during critical period. {current_data['moisture_anomaly_pct']:+.1f}% above normal. Waterlogging risk."
            else:
                direction = "neutral"
                confidence = 50
                rationale = f"Wet soil conditions. {current_data['moisture_anomaly_pct']:+.1f}% above normal. Adequate moisture reserve."
        
        elif status == "waterlogged":
            direction = "short"
            confidence = min(100, 60 + abs(current_data["moisture_anomaly_pct"]) * 0.3)
            rationale = f"Waterlogged soil in {region['name']}. {current_data['moisture_anomaly_pct']:+.1f}% above normal. Root damage risk."
        
        else:
            direction = "neutral"
            confidence = 50
            rationale = f"Soil moisture status: {status}."
        
        signal = {
            "region_id": region_id,
            "region_name": region["name"],
            "region_type": region["type"],
            "country": region["country"],
            "date": date,
            "signal_type": "soil_moisture",
            "direction": direction,
            "confidence": round(confidence, 1),
            "rationale": rationale,
            "instruments": region["instruments"],
            "surface_moisture": current_data["surface_moisture"],
            "root_zone_moisture": current_data["root_zone_moisture"],
            "moisture_anomaly_pct": current_data["moisture_anomaly_pct"],
            "status": status,
            "plant_available_water": current_data["plant_available_water"],
            "irrigation_need": current_data["irrigation_need"],
            "is_critical_season": is_critical,
            "baseline_moisture": baseline["moisture"]["mean"],
            "moisture_z_score": anomaly["moisture_z_score"],
            "impact_score": current_data["impact_score"],
            "soil_type": current_data["soil_type"],
            "data_quality": current_data["quality"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Save signal
        signal_file = self.output_dir / f"signal_{region_id}_{date}.json"
        signal_file.write_text(json.dumps(signal, indent=2))
        logger.info(f"Signal saved to {signal_file}")
        
        return signal
    
    def generate_all_signals(self, date: Optional[str] = None) -> List[Dict]:
        """
        Generate signals for all monitored regions.
        
        Args:
            date: Date for signals (default: today)
            
        Returns:
            List of signal dictionaries
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        signals = []
        
        for region_id in self.regions.keys():
            try:
                signal = self.generate_signal(region_id, date)
                if "error" not in signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error generating signal for {region_id}: {e}")
        
        # Sort by impact score
        signals.sort(key=lambda x: x["impact_score"], reverse=True)
        
        # Save summary
        summary = {
            "date": date,
            "total_regions": len(self.regions),
            "signals_generated": len(signals),
            "long_signals": sum(1 for s in signals if s["direction"] == "long"),
            "short_signals": sum(1 for s in signals if s["direction"] == "short"),
            "neutral_signals": sum(1 for s in signals if s["direction"] == "neutral"),
            "drought_regions": sum(1 for s in signals if s["status"] in ["drought", "severe_drought"]),
            "optimal_regions": sum(1 for s in signals if s["status"] == "optimal"),
            "waterlogged_regions": sum(1 for s in signals if s["status"] == "waterlogged"),
            "critical_season_regions": sum(1 for s in signals if s["is_critical_season"]),
            "by_region_type": self._group_by_type(signals),
            "signals": signals,
            "timestamp": datetime.now().isoformat()
        }
        
        summary_file = self.output_dir / f"summary_{date}.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        logger.info(f"Summary saved to {summary_file}")
        
        return signals
    
    def _group_by_type(self, signals: List[Dict]) -> Dict:
        """Group signals by region type."""
        groups = {}
        for signal in signals:
            rtype = signal["region_type"]
            if rtype not in groups:
                groups[rtype] = {"count": 0, "long": 0, "short": 0, "neutral": 0}
            groups[rtype]["count"] += 1
            groups[rtype][signal["direction"]] += 1
        return groups
    
    def get_regional_summary(self) -> Dict:
        """
        Get summary of all monitored regions.
        
        Returns:
            Dictionary with regional information
        """
        return {
            "monitoring_type": "soil_moisture",
            "satellites": ["SMAP", "Sentinel-1"],
            "metrics": ["Surface Moisture", "Root Zone Moisture", "PAW", "Irrigation Need"],
            "update_frequency": "Daily",
            "latency": "1-3 days",
            "total_regions": len(self.regions),
            "region_types": list(set(r["type"] for r in self.regions.values())),
            "regions": self.regions,
            "signal_logic": {
                "drought": "SHORT crops (yield risk)",
                "optimal": "LONG crops (excellent conditions)",
                "waterlogged": "SHORT crops (root damage)",
                "normal": "NEUTRAL (expected yields)"
            },
            "trading_instruments": list(set(
                inst for region in self.regions.values() 
                for inst in region["instruments"]
            ))
        }


def main():
    """Test soil moisture monitoring."""
    logging.basicConfig(level=logging.INFO)
    
    monitor = SoilMoistureMonitor()
    
    # Get regional summary
    print("\n💧 Soil Moisture Monitor - Regional Summary")
    print("=" * 60)
    summary = monitor.get_regional_summary()
    print(f"Monitoring {summary['total_regions']} agricultural regions")
    print(f"Satellites: {', '.join(summary['satellites'])}")
    
    # Generate signals for all regions
    print("\n🚀 Generating signals for all regions...")
    signals = monitor.generate_all_signals()
    
    print(f"\n📈 Generated {len(signals)} signals:")
    print("-" * 60)
    
    for signal in signals[:5]:  # Show top 5
        print(f"\n{signal['region_name']} ({signal['country']})")
        print(f"  Direction: {signal['direction'].upper()}")
        print(f"  Confidence: {signal['confidence']}%")
        print(f"  Surface Moisture: {signal['surface_moisture']:.3f} m³/m³ (baseline: {signal['baseline_moisture']:.3f})")
        print(f"  Status: {signal['status'].upper()}")
        print(f"  Anomaly: {signal['moisture_anomaly_pct']:+.1f}%")
        print(f"  PAW: {signal['plant_available_water']:.2f}")
        print(f"  Irrigation Need: {signal['irrigation_need']:.1f}%")
        print(f"  Instruments: {', '.join(signal['instruments'])}")
        print(f"  Rationale: {signal['rationale']}")


if __name__ == "__main__":
    main()
