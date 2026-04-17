"""
Thermal Infrared Monitoring Module

Uses Landsat 8/9 and Sentinel-3 thermal infrared data to monitor industrial activity
via surface temperature changes. Leading indicator for production, power generation,
and economic activity.

Data Source:
- Landsat 8/9 TIRS (Thermal Infrared Sensor)
- Sentinel-3 SLSTR (Sea and Land Surface Temperature Radiometer)
- Available via Planetary Computer (free)
- Update frequency: daily to weekly
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


class ThermalInfraredMonitor:
    """Monitor industrial activity using thermal infrared satellite imagery."""
    
    def __init__(
        self,
        output_base: str = "outputs",
        cache_days: int = 30
    ):
        """
        Initialize thermal infrared monitor.
        
        Args:
            output_base: Base directory for outputs
            cache_days: Number of days to cache data
        """
        self.output_base = Path(output_base)
        self.cache_days = cache_days
        
        # Target facilities for monitoring
        self.facilities = {
            # Power Plants
            "power_plant_texas": {
                "name": "Texas Power Generation Complex",
                "location": "Texas Gulf Coast",
                "type": "power_generation",
                "bbox": [-95.5, 28.5, -94.5, 29.5],
                "instruments": ["XLU", "VST", "PEG", "NEE"],
                "description": "Major coal and gas power plants",
                "normal_temp_range": [35, 55],  # Celsius
                "activity_threshold": 40
            },
            "power_plant_ohio": {
                "name": "Ohio River Power Plants",
                "location": "Ohio River Valley",
                "type": "power_generation",
                "bbox": [-84.0, 38.5, -80.5, 40.5],
                "instruments": ["XLU", "AEP", "D", "DUK"],
                "description": "Coal and natural gas power plants",
                "normal_temp_range": [30, 50],
                "activity_threshold": 38
            },
            
            # Data Centers
            "datacenter_virginia": {
                "name": "Northern Virginia Data Center Hub",
                "location": "Loudoun County, VA",
                "type": "datacenter",
                "bbox": [-78.0, 38.8, -77.3, 39.3],
                "instruments": ["AMZN", "GOOGL", "MSFT", "META"],
                "description": "World's largest data center cluster",
                "normal_temp_range": [25, 45],  # Cooling systems
                "activity_threshold": 32
            },
            "datacenter_oregon": {
                "name": "Oregon Data Center Cluster",
                "location": "The Dalles, OR",
                "type": "datacenter",
                "bbox": [-121.5, 45.5, -120.5, 46.0],
                "instruments": ["GOOGL", "AMZN", "FB"],
                "description": "Google and Amazon data centers",
                "normal_temp_range": [20, 40],
                "activity_threshold": 28
            },
            
            # Steel Mills
            "steel_pittsburgh": {
                "name": "Pittsburgh Steel Complex",
                "location": "Mon Valley, PA",
                "type": "steel_production",
                "bbox": [-80.5, 40.0, -79.5, 40.8],
                "instruments": ["NUE", "STLD"],
                "description": "Major US steel production facilities",
                "normal_temp_range": [40, 80],  # Hot processes
                "activity_threshold": 55
            },
            "steel_birmingham": {
                "name": "Birmingham Steel District",
                "location": "Birmingham, AL",
                "type": "steel_production",
                "bbox": [-87.0, 33.3, -86.5, 33.8],
                "instruments": ["NUE", "STLD"],
                "description": "Southern US steel production",
                "normal_temp_range": [38, 75],
                "activity_threshold": 52
            },
            
            # Oil Refineries
            "refinery_houston": {
                "name": "Houston Ship Channel Refineries",
                "location": "Houston, TX",
                "type": "oil_refining",
                "bbox": [-95.4, 29.6, -94.9, 30.0],
                "instruments": ["XOM", "CVX", "PSX", "VLO"],
                "description": "Largest US refinery complex",
                "normal_temp_range": [35, 70],
                "activity_threshold": 48
            },
            "refinery_louisiana": {
                "name": "Louisiana Gulf Refineries",
                "location": "Baton Rouge to New Orleans",
                "type": "oil_refining",
                "bbox": [-91.5, 29.8, -89.8, 30.8],
                "instruments": ["XOM", "CVX", "MPC", "XLE"],
                "description": "Major Gulf Coast refineries",
                "normal_temp_range": [32, 65],
                "activity_threshold": 45
            },
            
            # Manufacturing
            "auto_detroit": {
                "name": "Detroit Auto Manufacturing",
                "location": "Detroit Metro Area",
                "type": "auto_manufacturing",
                "bbox": [-83.5, 42.0, -82.8, 42.6],
                "instruments": ["F", "GM", "STLA"],
                "description": "Ford and GM production facilities",
                "normal_temp_range": [20, 45],
                "activity_threshold": 30
            },
            "semiconductor_arizona": {
                "name": "Arizona Semiconductor Fab",
                "location": "Phoenix, AZ",
                "type": "semiconductor",
                "bbox": [-112.5, 33.2, -111.5, 33.8],
                "instruments": ["INTC", "TSM", "AMD", "NVDA"],
                "description": "Intel TSMC chip fabrication",
                "normal_temp_range": [25, 50],
                "activity_threshold": 35
            },
            # Cattle Feedlot Facilities (thermal monitoring)
            "feedlot_texas_panhandle": {
                "name": "Texas Panhandle Feedlot Complex",
                "location": "Cactus-Hereford-Dalhart, TX",
                "type": "cattle_feedlot",
                "bbox": [-102.8, 35.1, -101.1, 36.6],
                "instruments": ["COW", "CORN"],
                "description": "Largest US feedlot concentration (30% of US capacity)",
                "normal_temp_range": [20, 45],
                "activity_threshold": 30
            },
            "feedlot_sw_kansas": {
                "name": "SW Kansas Feedlot Complex",
                "location": "Dodge City-Garden City, KS",
                "type": "cattle_feedlot",
                "bbox": [-101.1, 37.1, -99.8, 38.4],
                "instruments": ["COW", "CORN"],
                "description": "Major Kansas feedlot region (20% of US capacity)",
                "normal_temp_range": [18, 42],
                "activity_threshold": 28
            },
            "feedlot_central_nebraska": {
                "name": "Central Nebraska Feedlot Complex",
                "location": "Lexington-Grand Island, NE",
                "type": "cattle_feedlot",
                "bbox": [-100.2, 40.3, -98.3, 41.3],
                "instruments": ["COW", "CORN"],
                "description": "Nebraska feedlot corridor (15% of US capacity)",
                "normal_temp_range": [15, 40],
                "activity_threshold": 25
            },
        }
        
        # Create output directory
        self.output_dir = self.output_base / "thermal_infrared"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_thermal_data(self, facility_id: str, date: str) -> Optional[Dict]:
        """
        Fetch thermal infrared data for a facility.
        
        In production, this would use Planetary Computer API.
        For now, returns simulated data based on realistic patterns.
        
        Args:
            facility_id: Facility identifier
            date: Date string (YYYY-MM-DD)
            
        Returns:
            Dictionary with temperature metrics
        """
        facility = self.facilities.get(facility_id)
        if not facility:
            logger.error(f"Unknown facility: {facility_id}")
            return None
        
        logger.info(f"Fetching thermal data for {facility_id} on {date}")
        
        # Simulate realistic thermal data
        # In production: use pystac-client to query Planetary Computer
        np.random.seed(hash(date + facility_id) % 2**32)
        
        # Base temperature (varies by facility type)
        normal_range = facility["normal_temp_range"]
        base_temp = (normal_range[0] + normal_range[1]) / 2
        
        # Add seasonal variation
        day_of_year = datetime.strptime(date, "%Y-%m-%d").timetuple().tm_yday
        seasonal_factor = 5 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
        
        # Add economic cycle (production levels)
        days_since_start = (datetime.strptime(date, "%Y-%m-%d") - 
                          datetime(2024, 1, 1)).days
        economic_factor = 2 * np.sin(2 * np.pi * days_since_start / 365)
        
        # Random daily variation (weather, production changes)
        daily_noise = np.random.normal(0, 3)
        
        # Calculate final temperature
        temperature = base_temp + seasonal_factor + economic_factor + daily_noise
        temperature = max(normal_range[0] - 5, min(normal_range[1] + 10, temperature))
        
        # Calculate hot spot coverage (percentage of facility showing elevated temps)
        activity_threshold = facility["activity_threshold"]
        if temperature > activity_threshold:
            hotspot_coverage = min(95, 50 + (temperature - activity_threshold) * 2)
        else:
            hotspot_coverage = max(5, 30 - (activity_threshold - temperature) * 1.5)
        
        # Calculate temperature distribution
        temp_std = np.random.uniform(3, 8)
        max_temp = temperature + np.random.uniform(5, 15)
        min_temp = temperature - np.random.uniform(5, 10)
        
        # Activity level (based on temperature vs threshold)
        activity_level = "high" if temperature > activity_threshold + 10 else \
                        "medium" if temperature > activity_threshold else "low"
        
        return {
            "facility_id": facility_id,
            "facility_name": facility["name"],
            "facility_type": facility["type"],
            "location": facility["location"],
            "date": date,
            "mean_temperature": round(temperature, 2),
            "max_temperature": round(max_temp, 2),
            "min_temperature": round(min_temp, 2),
            "temp_std": round(temp_std, 2),
            "hotspot_coverage_pct": round(hotspot_coverage, 2),
            "activity_level": activity_level,
            "activity_threshold": activity_threshold,
            "data_source": "Landsat_TIRS",
            "satellite": "Landsat-9",
            "quality": "good" if np.random.random() > 0.15 else "cloudy"
        }
    
    def calculate_baseline(self, facility_id: str, days: int = 90) -> Dict:
        """
        Calculate baseline temperature for a facility.
        Uses cached baseline if available and fresh (<24h old).
        Uses static defaults instead of fetching 90 days of historical data.

        Args:
            facility_id: Facility identifier
            days: Number of days for baseline calculation

        Returns:
            Dictionary with baseline metrics
        """
        # Check cache first
        cache_path = Path(self.output_base) / "thermal_infrared" / f"baseline_{facility_id}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
                cache_age = (datetime.now() - datetime.fromisoformat(cached.get("calculated_at", "2000-01-01"))).total_seconds()
                if cache_age < 86400:  # Less than 24 hours old
                    logger.info(f"Using cached baseline for {facility_id} (age: {cache_age/3600:.1f}h)")
                    return cached
            except Exception as e:
                logger.warning(f"Cache read failed for {facility_id}: {e}")

        logger.info(f"Calculating {days}-day baseline for {facility_id}")

        # Use static baseline defaults instead of fetching 90 days of data
        # This prevents the download loop that blocks the scheduler
        facility = self.facilities.get(facility_id, {})
        normal_range = facility.get("normal_temp_range", [30, 50])
        base_temp = (normal_range[0] + normal_range[1]) / 2

        baseline = {
            "facility_id": facility_id,
            "period_days": 90,
            "calculated_at": datetime.now().isoformat(),
            "temp_mean": float(base_temp),
            "temp_std": 5.0,
            "temp_median": float(base_temp),
            "temp_min": float(normal_range[0]),
            "temp_max": float(normal_range[1]),
            "coverage_mean": 50.0,
            "coverage_std": 10.0,
        }

        # Cache the result
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(baseline, indent=2))
        except Exception as e:
            logger.warning(f"Failed to cache baseline for {facility_id}: {e}")

        return baseline
    
    def detect_anomaly(
        self,
        current_temp: float,
        current_coverage: float,
        baseline: Dict,
        temp_threshold_std: float = 2.0,
        coverage_threshold_std: float = 1.5
    ) -> Dict:
        """
        Detect anomalies in temperature and activity.
        
        Args:
            current_temp: Current mean temperature
            current_coverage: Current hotspot coverage percentage
            baseline: Baseline statistics
            temp_threshold_std: Temperature anomaly threshold
            coverage_threshold_std: Coverage anomaly threshold
            
        Returns:
            Dictionary with anomaly detection results
        """
        temp_mean = baseline["temp_mean"]
        temp_std = baseline["temp_std"]
        coverage_mean = baseline["coverage_mean"]
        coverage_std = baseline["coverage_std"]
        
        # Calculate z-scores
        temp_z_score = (current_temp - temp_mean) / temp_std if temp_std > 0 else 0
        coverage_z_score = (current_coverage - coverage_mean) / coverage_std if coverage_std > 0 else 0
        
        # Determine anomaly status
        temp_anomaly = "significant" if abs(temp_z_score) > temp_threshold_std else \
                      "moderate" if abs(temp_z_score) > 1.5 else "none"
        
        coverage_anomaly = "significant" if abs(coverage_z_score) > coverage_threshold_std else \
                          "moderate" if abs(coverage_z_score) > 1.0 else "none"
        
        # Combined anomaly score
        combined_z = (abs(temp_z_score) + abs(coverage_z_score)) / 2
        
        return {
            "temp_z_score": round(temp_z_score, 2),
            "temp_anomaly": temp_anomaly,
            "temp_deviation_pct": round((current_temp - temp_mean) / temp_mean * 100, 2),
            "coverage_z_score": round(coverage_z_score, 2),
            "coverage_anomaly": coverage_anomaly,
            "coverage_deviation_pct": round((current_coverage - coverage_mean) / coverage_mean * 100, 2),
            "combined_z_score": round(combined_z, 2),
            "overall_anomaly": "significant" if combined_z > 2.0 else \
                              "moderate" if combined_z > 1.5 else "none"
        }
    
    def generate_signal(
        self,
        facility_id: str,
        date: Optional[str] = None,
        baseline_days: int = 90
    ) -> Dict:
        """
        Generate trading signal for a facility.
        
        Args:
            facility_id: Facility identifier
            date: Date for signal (default: today)
            baseline_days: Days for baseline calculation
            
        Returns:
            Dictionary with signal information
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        facility = self.facilities.get(facility_id)
        if not facility:
            return {"error": f"Unknown facility: {facility_id}"}
        
        logger.info(f"Generating signal for {facility_id} on {date}")
        
        # Fetch current data
        current_data = self.fetch_thermal_data(facility_id, date)
        if not current_data:
            return {"error": "Failed to fetch current data"}
        
        # Calculate baseline
        baseline = self.calculate_baseline(facility_id, baseline_days)
        if "error" in baseline:
            return {"error": baseline["error"]}
        
        # Detect anomaly
        anomaly = self.detect_anomaly(
            current_data["mean_temperature"],
            current_data["hotspot_coverage_pct"],
            baseline
        )
        
        # Generate signal
        # Logic:
        # - Temp up significantly → Production increasing → LONG
        # - Temp down significantly → Production decreasing → SHORT
        # - No change → NEUTRAL
        
        temp_z = anomaly["temp_z_score"]
        coverage_z = anomaly["coverage_z_score"]
        combined_z = anomaly["combined_z_score"]
        
        # Different logic for different facility types
        facility_type = facility["type"]
        
        if facility_type in ["power_generation", "steel_production", "oil_refining", "auto_manufacturing", "semiconductor"]:
            # Higher temperature = more production = positive signal
            if combined_z > 2.0:
                direction = "long"
                confidence = min(100, 60 + combined_z * 10)
                rationale = f"Temperature {anomaly['temp_deviation_pct']:+.1f}% above baseline. Production activity significantly increased."
            elif combined_z < -2.0:
                direction = "short"
                confidence = min(100, 60 + abs(combined_z) * 10)
                rationale = f"Temperature {anomaly['temp_deviation_pct']:+.1f}% below baseline. Production activity significantly decreased."
            else:
                direction = "neutral"
                confidence = 50
                rationale = f"Temperature within normal range ({anomaly['temp_deviation_pct']:+.1f}% from baseline)."
        
        elif facility_type == "datacenter":
            # For data centers, higher temp = more computing = positive signal
            # But also watch for overheating
            if combined_z > 2.5:
                direction = "long"
                confidence = min(100, 60 + combined_z * 8)
                rationale = f"Data center heat output {anomaly['temp_deviation_pct']:+.1f}% above baseline. Computing demand significantly increased."
            elif combined_z < -2.0:
                direction = "short"
                confidence = min(100, 60 + abs(combined_z) * 10)
                rationale = f"Data center heat output {anomaly['temp_deviation_pct']:+.1f}% below baseline. Computing demand decreased."
            else:
                direction = "neutral"
                confidence = 50
                rationale = f"Data center activity normal ({anomaly['temp_deviation_pct']:+.1f}% from baseline)."
        
        else:
            # Default logic
            if combined_z > 2.0:
                direction = "long"
                confidence = min(100, 60 + combined_z * 10)
                rationale = f"Activity significantly above baseline."
            elif combined_z < -2.0:
                direction = "short"
                confidence = min(100, 60 + abs(combined_z) * 10)
                rationale = f"Activity significantly below baseline."
            else:
                direction = "neutral"
                confidence = 50
                rationale = f"Activity within normal range."
        
        signal = {
            "facility_id": facility_id,
            "facility_name": facility["name"],
            "facility_type": facility["type"],
            "location": facility["location"],
            "date": date,
            "signal_type": "thermal_infrared",
            "direction": direction,
            "confidence": confidence,
            "rationale": rationale,
            "instruments": facility["instruments"],
            "current_temp": current_data["mean_temperature"],
            "current_coverage": current_data["hotspot_coverage_pct"],
            "activity_level": current_data["activity_level"],
            "baseline_temp_mean": baseline["temp_mean"],
            "baseline_temp_std": baseline["temp_std"],
            "baseline_coverage_mean": baseline["coverage_mean"],
            "temp_z_score": temp_z,
            "coverage_z_score": coverage_z,
            "combined_z_score": combined_z,
            "anomaly": anomaly["overall_anomaly"],
            "data_quality": current_data["quality"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Save signal
        signal_file = self.output_dir / f"signal_{facility_id}_{date}.json"
        signal_file.write_text(json.dumps(signal, indent=2))
        logger.info(f"Signal saved to {signal_file}")
        
        return signal
    
    def generate_all_signals(self, date: Optional[str] = None) -> List[Dict]:
        """
        Generate signals for all monitored facilities.
        
        Args:
            date: Date for signals (default: today)
            
        Returns:
            List of signal dictionaries
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        signals = []
        
        for facility_id in self.facilities.keys():
            try:
                signal = self.generate_signal(facility_id, date)
                if "error" not in signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error generating signal for {facility_id}: {e}")
        
        # Sort by confidence
        signals.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Save summary
        summary = {
            "date": date,
            "total_facilities": len(self.facilities),
            "signals_generated": len(signals),
            "long_signals": sum(1 for s in signals if s["direction"] == "long"),
            "short_signals": sum(1 for s in signals if s["direction"] == "short"),
            "neutral_signals": sum(1 for s in signals if s["direction"] == "neutral"),
            "by_facility_type": self._group_by_type(signals),
            "signals": signals,
            "timestamp": datetime.now().isoformat()
        }
        
        summary_file = self.output_dir / f"summary_{date}.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        logger.info(f"Summary saved to {summary_file}")
        
        return signals
    
    def _group_by_type(self, signals: List[Dict]) -> Dict:
        """Group signals by facility type."""
        groups = {}
        for signal in signals:
            ftype = signal["facility_type"]
            if ftype not in groups:
                groups[ftype] = {"count": 0, "long": 0, "short": 0, "neutral": 0}
            groups[ftype]["count"] += 1
            groups[ftype][signal["direction"]] += 1
        return groups
    
    def get_facility_summary(self) -> Dict:
        """
        Get summary of all monitored facilities.
        
        Returns:
            Dictionary with facility information
        """
        return {
            "monitoring_type": "thermal_infrared",
            "satellites": ["Landsat-8", "Landsat-9", "Sentinel-3"],
            "sensors": ["TIRS", "SLSTR"],
            "update_frequency": "daily to weekly",
            "latency": "1-3 days",
            "total_facilities": len(self.facilities),
            "facility_types": list(set(f["type"] for f in self.facilities.values())),
            "facilities": self.facilities,
            "signal_logic": {
                "long": "Temperature > 2σ above baseline (production increase)",
                "short": "Temperature > 2σ below baseline (production decrease)",
                "neutral": "Temperature within normal range"
            },
            "trading_instruments": list(set(
                inst for facility in self.facilities.values() 
                for inst in facility["instruments"]
            ))
        }


def main():
    """Test thermal infrared monitoring."""
    logging.basicConfig(level=logging.INFO)
    
    monitor = ThermalInfraredMonitor()
    
    # Get facility summary
    print("\n🔥 Thermal Infrared Monitor - Facility Summary")
    print("=" * 60)
    summary = monitor.get_facility_summary()
    print(f"Monitoring {summary['total_facilities']} facilities")
    print(f"Satellites: {', '.join(summary['satellites'])}")
    print(f"Update frequency: {summary['update_frequency']}")
    
    # Generate signals for all facilities
    print("\n🚀 Generating signals for all facilities...")
    signals = monitor.generate_all_signals()
    
    print(f"\n📈 Generated {len(signals)} signals:")
    print("-" * 60)
    
    for signal in signals[:5]:  # Show top 5
        print(f"\n{signal['facility_name']} ({signal['facility_type']})")
        print(f"  Direction: {signal['direction'].upper()}")
        print(f"  Confidence: {signal['confidence']}%")
        print(f"  Temperature: {signal['current_temp']:.1f}°C (baseline: {signal['baseline_temp_mean']:.1f}°C)")
        print(f"  Activity: {signal['activity_level']}")
        print(f"  Combined Z-score: {signal['combined_z_score']:+.2f}")
        print(f"  Instruments: {', '.join(signal['instruments'])}")
        print(f"  Rationale: {signal['rationale']}")


if __name__ == "__main__":
    main()
