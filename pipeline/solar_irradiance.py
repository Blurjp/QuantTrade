"""
Solar Irradiance Monitoring Module

Uses satellite data to monitor solar irradiance levels for predicting solar power generation.
Leading indicator for solar energy production, electricity demand, and natural gas consumption.

Data Source:
- MODIS (Terra/Aqua): Cloud cover, aerosol optical depth
- Sentinel-3 SLSTR: Surface solar irradiance
- Available via Planetary Computer (free)
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


class SolarIrradianceMonitor:
    """Monitor solar irradiance for energy trading signals."""
    
    def __init__(
        self,
        output_base: str = "outputs",
        cache_days: int = 30
    ):
        """
        Initialize solar irradiance monitor.
        
        Args:
            output_base: Base directory for outputs
            cache_days: Number of days to cache data
        """
        self.output_base = Path(output_base)
        self.cache_days = cache_days
        
        # Solar farm regions for monitoring
        self.regions = {
            # USA - Major Solar Markets
            "usa_california_solar": {
                "name": "California Solar Belt",
                "bbox": [-121.0, 32.5, -114.0, 40.0],
                "country": "USA",
                "type": "solar_farm_cluster",
                "instruments": ["TAN", "FSLR", "SPWR", "SEDG", "ENPH"],
                "description": "Largest US solar installation region",
                "installed_capacity_gw": 35.0,
                "baseline_irradiance": 5.5,  # kWh/m²/day
                "grid_region": "CAISO"
            },
            "usa_texas_solar": {
                "name": "Texas Solar Corridor",
                "bbox": [-106.5, 26.0, -93.5, 34.0],
                "country": "USA",
                "type": "solar_farm_cluster",
                "instruments": ["TAN", "FSLR", "NOVA", "XLU"],
                "description": "Fastest growing US solar market",
                "installed_capacity_gw": 20.0,
                "baseline_irradiance": 5.2,
                "grid_region": "ERCOT"
            },
            "usa_arizona_solar": {
                "name": "Arizona Desert Solar",
                "bbox": [-115.0, 31.0, -109.0, 37.0],
                "country": "USA",
                "type": "solar_farm_cluster",
                "instruments": ["TAN", "FSLR", "SPWR"],
                "description": "High irradiance desert region",
                "installed_capacity_gw": 8.0,
                "baseline_irradiance": 6.2,
                "grid_region": "WECC"
            },
            
            # Europe
            "europe_spain_solar": {
                "name": "Spain Solar Hub",
                "bbox": [-9.5, 36.0, 3.5, 44.0],
                "country": "Spain",
                "type": "solar_farm_cluster",
                "instruments": ["TAN", "ICLN", "PBW"],
                "description": "Europe's largest solar market",
                "installed_capacity_gw": 18.0,
                "baseline_irradiance": 5.0,
                "grid_region": "Spain"
            },
            "europe_germany_solar": {
                "name": "Germany Solar Region",
                "bbox": [6.0, 47.0, 15.0, 55.0],
                "country": "Germany",
                "type": "solar_farm_cluster",
                "instruments": ["TAN", "ICLN", "QCLN"],
                "description": "Europe's second largest market",
                "installed_capacity_gw": 60.0,
                "baseline_irradiance": 3.2,
                "grid_region": "Germany"
            },
            
            # Asia
            "china_solar_west": {
                "name": "Western China Solar Base",
                "bbox": [75.0, 30.0, 110.0, 45.0],
                "country": "China",
                "type": "solar_farm_cluster",
                "instruments": ["FXI", "MCHI", "TAN"],
                "description": "Largest solar installations in China",
                "installed_capacity_gw": 150.0,
                "baseline_irradiance": 5.8,
                "grid_region": "China"
            },
            "india_solar_rajasthan": {
                "name": "Rajasthan Solar Park",
                "bbox": [69.0, 23.0, 78.0, 30.5],
                "country": "India",
                "type": "solar_farm_cluster",
                "instruments": ["INDA", "TAN", "ICLN"],
                "description": "India's largest solar park",
                "installed_capacity_gw": 25.0,
                "baseline_irradiance": 5.9,
                "grid_region": "India"
            },
            
            # Natural Gas Demand Proxies (cloudy regions = more gas power)
            "usa_northeast_cloud": {
                "name": "US Northeast Cloud Cover",
                "bbox": [-80.0, 38.0, -66.0, 48.0],
                "country": "USA",
                "type": "gas_demand_proxy",
                "instruments": ["UNG", "XLU", "D"],
                "description": "Low solar → high gas power demand",
                "installed_capacity_gw": 5.0,
                "baseline_irradiance": 3.8,
                "grid_region": "PJM"
            },
            "europe_uk_cloud": {
                "name": "UK Cloud Cover Region",
                "bbox": [-8.0, 49.5, 2.0, 61.0],
                "country": "UK",
                "type": "gas_demand_proxy",
                "instruments": ["UNG", "XLU"],
                "description": "Low solar → high UK gas demand",
                "installed_capacity_gw": 15.0,
                "baseline_irradiance": 2.5,
                "grid_region": "UK"
            },
        }
        
        # Create output directory
        self.output_dir = self.output_base / "solar_irradiance"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_irradiance_data(self, region_id: str, date: str) -> Optional[Dict]:
        """
        Fetch solar irradiance data for a region.
        
        In production, this would use Planetary Computer API.
        For now, returns simulated data based on realistic patterns.
        
        Args:
            region_id: Region identifier
            date: Date string (YYYY-MM-DD)
            
        Returns:
            Dictionary with irradiance metrics
        """
        region = self.regions.get(region_id)
        if not region:
            logger.error(f"Unknown region: {region_id}")
            return None
        
        logger.info(f"Fetching solar irradiance data for {region_id} on {date}")
        
        # Simulate realistic solar irradiance data
        # In production: use pystac-client to query Planetary Computer
        np.random.seed(hash(date + region_id) % 2**32)
        
        # Get baseline irradiance
        baseline = region["baseline_irradiance"]
        
        # Add seasonal variation (higher in summer)
        day_of_year = datetime.strptime(date, "%Y-%m-%d").timetuple().tm_yday
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Add cloud cover variation (random weather)
        cloud_factor = np.random.uniform(0.4, 1.0)
        
        # Add latitude effect
        lat_factor = 1.0  # Already in baseline
        
        # Calculate actual irradiance
        irradiance = baseline * seasonal_factor * cloud_factor
        irradiance = max(0.5, min(8.0, irradiance))
        
        # Calculate derived metrics
        capacity = region["installed_capacity_gw"]
        capacity_factor = irradiance / (baseline * 1.2)  # Relative to optimal
        capacity_factor = min(1.0, max(0.0, capacity_factor))
        
        # Estimated power generation (GWh/day)
        daily_generation = capacity * capacity_factor * 24 * 0.8  # 80% of nameplate
        
        # Cloud cover percentage
        cloud_cover_pct = (1 - cloud_factor) * 100
        
        # Clear sky vs actual
        clear_sky_irradiance = baseline * seasonal_factor
        
        return {
            "region_id": region_id,
            "region_name": region["name"],
            "region_type": region["type"],
            "country": region["country"],
            "date": date,
            "irradiance_kwh_m2_day": round(irradiance, 2),
            "clear_sky_irradiance": round(clear_sky_irradiance, 2),
            "cloud_cover_pct": round(cloud_cover_pct, 1),
            "capacity_factor": round(capacity_factor, 3),
            "installed_capacity_gw": capacity,
            "estimated_generation_gwh": round(daily_generation, 1),
            "grid_region": region["grid_region"],
            "data_source": "MODIS_Sentinel3",
            "satellites": ["Terra", "Aqua", "Sentinel-3"],
            "quality": "good" if np.random.random() > 0.1 else "partial_cloud"
        }
    
    def calculate_baseline(self, region_id: str, days: int = 90) -> Dict:
        """
        Calculate baseline irradiance for a region.
        Uses cached baseline if available and fresh (<24h old).
        Uses static defaults instead of fetching 90 days of historical data.

        Args:
            region_id: Region identifier
            days: Number of days for baseline calculation

        Returns:
            Dictionary with baseline metrics
        """
        # Check cache first
        cache_path = Path(self.output_base) / "solar_irradiance" / f"baseline_{region_id}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
                cache_age = (datetime.now() - datetime.fromisoformat(cached.get("calculated_at", "2000-01-01"))).total_seconds()
                if cache_age < 86400:  # Less than 24 hours old
                    logger.info(f"Using cached baseline for {region_id} (age: {cache_age/3600:.1f}h)")
                    return cached
            except Exception as e:
                logger.warning(f"Cache read failed for {region_id}: {e}")

        logger.info(f"Calculating {days}-day baseline for {region_id}")

        # Use static baseline defaults instead of fetching 90 days of data
        # This prevents the download loop that blocks the scheduler
        region = self.regions.get(region_id, {})
        baseline_irradiance = region.get("baseline_irradiance", 4.0)
        capacity_gw = region.get("installed_capacity_gw", 10.0)
        # Estimate generation: capacity * avg_capacity_factor * 24h * 0.8 efficiency
        avg_generation = capacity_gw * 0.5 * 24 * 0.8

        baseline = {
            "region_id": region_id,
            "period_days": 90,
            "calculated_at": datetime.now().isoformat(),
            "irradiance": {
                "mean": float(baseline_irradiance),
                "std": float(baseline_irradiance * 0.2),
                "median": float(baseline_irradiance),
            },
            "cloud_cover": {
                "mean": 30.0,
                "std": 15.0,
                "median": 28.0,
            },
            "generation": {
                "mean": round(avg_generation, 1),
                "std": round(avg_generation * 0.2, 1),
                "median": round(avg_generation, 1),
            }
        }

        # Cache the result
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(baseline, indent=2))
        except Exception as e:
            logger.warning(f"Failed to cache baseline for {region_id}: {e}")

        return baseline
    
    def detect_anomaly(
        self,
        current_data: Dict,
        baseline: Dict,
        threshold_std: float = 2.0
    ) -> Dict:
        """
        Detect anomalies in solar irradiance.
        
        Args:
            current_data: Current irradiance data
            baseline: Baseline statistics
            threshold_std: Number of standard deviations for anomaly
            
        Returns:
            Dictionary with anomaly detection results
        """
        # Calculate z-scores
        irr_z = (current_data["irradiance_kwh_m2_day"] - baseline["irradiance"]["mean"]) / \
                baseline["irradiance"]["std"] if baseline["irradiance"]["std"] > 0 else 0
        
        cloud_z = (current_data["cloud_cover_pct"] - baseline["cloud_cover"]["mean"]) / \
                  baseline["cloud_cover"]["std"] if baseline["cloud_cover"]["std"] > 0 else 0
        
        gen_z = (current_data["estimated_generation_gwh"] - baseline["generation"]["mean"]) / \
                baseline["generation"]["std"] if baseline["generation"]["std"] > 0 else 0
        
        # Determine anomaly status
        irr_anomaly = "significant" if abs(irr_z) > threshold_std else \
                     "moderate" if abs(irr_z) > 1.5 else "none"
        
        cloud_anomaly = "significant" if abs(cloud_z) > threshold_std else \
                       "moderate" if abs(cloud_z) > 1.5 else "none"
        
        gen_anomaly = "significant" if abs(gen_z) > threshold_std else \
                     "moderate" if abs(gen_z) > 1.5 else "none"
        
        # Combined score
        combined_z = (abs(irr_z) + abs(cloud_z) + abs(gen_z)) / 3
        
        return {
            "irradiance_z_score": round(irr_z, 2),
            "irradiance_anomaly": irr_anomaly,
            "irradiance_deviation_pct": round((current_data["irradiance_kwh_m2_day"] - 
                                              baseline["irradiance"]["mean"]) / 
                                              baseline["irradiance"]["mean"] * 100, 2),
            "cloud_z_score": round(cloud_z, 2),
            "cloud_anomaly": cloud_anomaly,
            "cloud_deviation_pct": round((current_data["cloud_cover_pct"] - 
                                         baseline["cloud_cover"]["mean"]) / 
                                         baseline["cloud_cover"]["mean"] * 100, 2),
            "generation_z_score": round(gen_z, 2),
            "generation_anomaly": gen_anomaly,
            "generation_deviation_pct": round((current_data["estimated_generation_gwh"] - 
                                              baseline["generation"]["mean"]) / 
                                              baseline["generation"]["mean"] * 100, 2),
            "combined_z_score": round(combined_z, 2),
            "overall_anomaly": "significant" if combined_z > 2.0 else \
                              "moderate" if combined_z > 1.5 else "none"
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
        current_data = self.fetch_irradiance_data(region_id, date)
        if not current_data:
            return {"error": "Failed to fetch current data"}
        
        # Calculate baseline
        baseline = self.calculate_baseline(region_id, baseline_days)
        if "error" in baseline:
            return {"error": baseline["error"]}
        
        # Detect anomaly
        anomaly = self.detect_anomaly(current_data, baseline)
        
        # Generate signal based on region type
        region_type = region["type"]
        combined_z = anomaly["combined_z_score"]
        
        if region_type == "solar_farm_cluster":
            # High irradiance = high solar generation = LONG solar stocks
            if combined_z > 2.0:
                direction = "long"
                confidence = min(100, 60 + combined_z * 10)
                rationale = f"Solar irradiance {anomaly['irradiance_deviation_pct']:+.1f}% above baseline. Solar generation significantly increased."
            elif combined_z < -2.0:
                direction = "short"
                confidence = min(100, 60 + abs(combined_z) * 10)
                rationale = f"Solar irradiance {anomaly['irradiance_deviation_pct']:+.1f}% below baseline. Solar generation significantly decreased."
            else:
                direction = "neutral"
                confidence = 50
                rationale = f"Solar irradiance within normal range ({anomaly['irradiance_deviation_pct']:+.1f}% from baseline)."
        
        elif region_type == "gas_demand_proxy":
            # LOW irradiance = HIGH cloud cover = HIGH gas demand
            if anomaly["cloud_z_score"] > 2.0:
                # Very cloudy → high gas demand → LONG gas
                direction = "long"
                confidence = min(100, 60 + abs(anomaly["cloud_z_score"]) * 10)
                rationale = f"Cloud cover {anomaly['cloud_deviation_pct']:+.1f}% above baseline. Natural gas power demand increased."
            elif anomaly["cloud_z_score"] < -2.0:
                # Very sunny → low gas demand → SHORT gas
                direction = "short"
                confidence = min(100, 60 + abs(anomaly["cloud_z_score"]) * 10)
                rationale = f"Cloud cover {anomaly['cloud_deviation_pct']:+.1f}% below baseline. Natural gas power demand decreased."
            else:
                direction = "neutral"
                confidence = 50
                rationale = f"Cloud cover within normal range ({anomaly['cloud_deviation_pct']:+.1f}% from baseline)."
        
        else:
            # Default logic
            if combined_z > 2.0:
                direction = "long"
                confidence = min(100, 60 + combined_z * 10)
                rationale = f"Irradiance significantly above baseline."
            elif combined_z < -2.0:
                direction = "short"
                confidence = min(100, 60 + abs(combined_z) * 10)
                rationale = f"Irradiance significantly below baseline."
            else:
                direction = "neutral"
                confidence = 50
                rationale = f"Irradiance within normal range."
        
        signal = {
            "region_id": region_id,
            "region_name": region["name"],
            "region_type": region["type"],
            "country": region["country"],
            "date": date,
            "signal_type": "solar_irradiance",
            "direction": direction,
            "confidence": confidence,
            "rationale": rationale,
            "instruments": region["instruments"],
            "current_irradiance": current_data["irradiance_kwh_m2_day"],
            "current_cloud_cover": current_data["cloud_cover_pct"],
            "current_generation": current_data["estimated_generation_gwh"],
            "capacity_factor": current_data["capacity_factor"],
            "baseline_irradiance": baseline["irradiance"]["mean"],
            "baseline_cloud_cover": baseline["cloud_cover"]["mean"],
            "baseline_generation": baseline["generation"]["mean"],
            "irradiance_z_score": anomaly["irradiance_z_score"],
            "cloud_z_score": anomaly["cloud_z_score"],
            "combined_z_score": combined_z,
            "anomaly": anomaly["overall_anomaly"],
            "grid_region": current_data["grid_region"],
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
        
        # Sort by confidence
        signals.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Save summary
        summary = {
            "date": date,
            "total_regions": len(self.regions),
            "signals_generated": len(signals),
            "long_signals": sum(1 for s in signals if s["direction"] == "long"),
            "short_signals": sum(1 for s in signals if s["direction"] == "short"),
            "neutral_signals": sum(1 for s in signals if s["direction"] == "neutral"),
            "total_generation_gwh": sum(s["current_generation"] for s in signals),
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
            "monitoring_type": "solar_irradiance",
            "satellites": ["Terra (MODIS)", "Aqua (MODIS)", "Sentinel-3 (SLSTR)"],
            "metrics": ["Irradiance", "Cloud Cover", "Capacity Factor"],
            "update_frequency": "Daily",
            "latency": "1-3 days",
            "total_regions": len(self.regions),
            "total_installed_capacity_gw": sum(r["installed_capacity_gw"] for r in self.regions.values()),
            "region_types": list(set(r["type"] for r in self.regions.values())),
            "regions": self.regions,
            "signal_logic": {
                "solar_farms": "High irradiance → LONG solar stocks (TAN, FSLR)",
                "gas_demand": "High cloud cover → LONG natural gas (UNG)"
            },
            "trading_instruments": list(set(
                inst for region in self.regions.values() 
                for inst in region["instruments"]
            ))
        }


def main():
    """Test solar irradiance monitoring."""
    logging.basicConfig(level=logging.INFO)
    
    monitor = SolarIrradianceMonitor()
    
    # Get regional summary
    print("\n☀️ Solar Irradiance Monitor - Regional Summary")
    print("=" * 60)
    summary = monitor.get_regional_summary()
    print(f"Monitoring {summary['total_regions']} regions")
    print(f"Total installed capacity: {summary['total_installed_capacity_gw']:.0f} GW")
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
        print(f"  Irradiance: {signal['current_irradiance']:.2f} kWh/m²/day (baseline: {signal['baseline_irradiance']:.2f})")
        print(f"  Cloud Cover: {signal['current_cloud_cover']:.1f}%")
        print(f"  Generation: {signal['current_generation']:.1f} GWh")
        print(f"  Capacity Factor: {signal['capacity_factor']:.1%}")
        print(f"  Instruments: {', '.join(signal['instruments'])}")
        print(f"  Rationale: {signal['rationale']}")


if __name__ == "__main__":
    main()
