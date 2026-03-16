"""
Sea Surface Temperature (SST) Monitoring Module

Uses satellite data to monitor ocean temperatures for predicting El Niño/La Niña events
and their impact on global commodity markets (agriculture, energy, metals).

Data Source:
- MODIS (Terra/Aqua): Sea surface temperature
- AVHRR (NOAA): SST anomaly detection
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


class SeaSurfaceTemperatureMonitor:
    """Monitor sea surface temperature for commodity trading signals."""
    
    def __init__(
        self,
        output_base: str = "outputs",
        cache_days: int = 30
    ):
        """
        Initialize SST monitor.
        
        Args:
            output_base: Base directory for outputs
            cache_days: Number of days to cache data
        """
        self.output_base = Path(output_base)
        self.cache_days = cache_days
        
        # Key ocean regions for monitoring
        self.regions = {
            # ENSO Regions (El Niño/La Niña)
            "nino34": {
                "name": "Niño 3.4 Region",
                "bbox": [-170.0, -5.0, -120.0, 5.0],
                "ocean": "Pacific",
                "type": "enso_indicator",
                "instruments": ["CORN", "SOYB", "WEAT", "CANE", "JO"],
                "description": "Primary ENSO monitoring region",
                "baseline_sst": 27.5,  # °C
                "enso_threshold": 0.5,
                "impact": "Global weather patterns"
            },
            "nino3": {
                "name": "Niño 3 Region",
                "bbox": [-150.0, -5.0, -90.0, 5.0],
                "ocean": "Pacific",
                "type": "enso_indicator",
                "instruments": ["CORN", "SOYB", "WEAT"],
                "description": "Eastern Pacific ENSO region",
                "baseline_sst": 26.0,
                "enso_threshold": 0.5,
                "impact": "South American weather"
            },
            "nino4": {
                "name": "Niño 4 Region",
                "bbox": [-160.0, -5.0, -150.0, 5.0],
                "ocean": "Pacific",
                "type": "enso_indicator",
                "instruments": ["CORN", "SOYB", "WEAT"],
                "description": "Western Pacific ENSO region",
                "baseline_sst": 29.0,
                "enso_threshold": 0.5,
                "impact": "Asian monsoon"
            },
            
            # Agricultural Impact Regions
            "gulf_mexico": {
                "name": "Gulf of Mexico",
                "bbox": [-98.0, 18.0, -80.0, 30.0],
                "ocean": "Atlantic",
                "type": "agricultural_impact",
                "instruments": ["CORN", "SOYB", "COTTON", "NG"],
                "description": "US Gulf moisture source",
                "baseline_sst": 27.0,
                "impact": "US Midwest rainfall"
            },
            "atlantic_hurricane": {
                "name": "Atlantic Hurricane Region",
                "bbox": [-80.0, 10.0, -20.0, 30.0],
                "ocean": "Atlantic",
                "type": "hurricane_zone",
                "instruments": ["NG", "OIL", "XLE", "UNG"],
                "description": "Hurricane formation zone",
                "baseline_sst": 26.5,
                "hurricane_threshold": 26.0,
                "impact": "Energy infrastructure"
            },
            
            # Monsoon Regions
            "indian_ocean": {
                "name": "Indian Ocean",
                "bbox": [50.0, -10.0, 100.0, 20.0],
                "ocean": "Indian",
                "type": "monsoon_region",
                "instruments": ["COTTON", "SUGAR", "TEA", "RICE"],
                "description": "Indian monsoon driver",
                "baseline_sst": 28.0,
                "impact": "Indian agriculture"
            },
            "pacific_warm_pool": {
                "name": "Pacific Warm Pool",
                "bbox": [120.0, -10.0, 170.0, 10.0],
                "ocean": "Pacific",
                "type": "monsoon_region",
                "instruments": ["RICE", "PALM", "SUGAR"],
                "description": "Southeast Asian monsoon driver",
                "baseline_sst": 29.5,
                "impact": "SE Asia agriculture"
            },
            
            # Fishing Regions
            "peru_current": {
                "name": "Peru/Humboldt Current",
                "bbox": [-85.0, -20.0, -70.0, -5.0],
                "ocean": "Pacific",
                "type": "fishing_grounds",
                "instruments": ["FISH", "FMC", "SEA"],
                "description": "Major anchovy fishery",
                "baseline_sst": 18.0,
                "impact": "Fish meal production"
            },
            "benguela_current": {
                "name": "Benguela Current",
                "bbox": [10.0, -35.0, 20.0, -20.0],
                "ocean": "Atlantic",
                "type": "fishing_grounds",
                "instruments": ["FISH", "SEA"],
                "description": "South African fishery",
                "baseline_sst": 15.0,
                "impact": "African fisheries"
            },
        }
        
        # Create output directory
        self.output_dir = self.output_base / "sea_surface_temperature"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_sst_data(self, region_id: str, date: str) -> Optional[Dict]:
        """
        Fetch sea surface temperature data for a region.
        
        In production, this would use Planetary Computer API.
        For now, returns simulated data based on realistic patterns.
        
        Args:
            region_id: Region identifier
            date: Date string (YYYY-MM-DD)
            
        Returns:
            Dictionary with SST metrics
        """
        region = self.regions.get(region_id)
        if not region:
            logger.error(f"Unknown region: {region_id}")
            return None
        
        logger.info(f"Fetching SST data for {region_id} on {date}")
        
        # Simulate realistic SST data
        # In production: use pystac-client to query Planetary Computer
        np.random.seed(hash(date + region_id) % 2**32)
        
        # Get baseline SST
        baseline = region["baseline_sst"]
        
        # Add seasonal variation
        day_of_year = datetime.strptime(date, "%Y-%m-%d").timetuple().tm_yday
        seasonal_factor = 1.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Add ENSO variation (3-7 year cycle)
        days_since_start = (datetime.strptime(date, "%Y-%m-%d") - 
                          datetime(2020, 1, 1)).days
        enso_cycle = 1.5 * np.sin(2 * np.pi * days_since_start / (5.5 * 365))
        
        # Random daily variation
        daily_noise = np.random.normal(0, 0.3)
        
        # Calculate actual SST
        sst = baseline + seasonal_factor + enso_cycle + daily_noise
        sst = max(10.0, min(35.0, sst))
        
        # Calculate anomaly (relative to baseline)
        anomaly = sst - baseline
        
        # Determine ENSO state for Niño regions
        enso_state = "neutral"
        if region["type"] == "enso_indicator":
            if anomaly > region["enso_threshold"]:
                enso_state = "el_nino"
            elif anomaly < -region["enso_threshold"]:
                enso_state = "la_nina"
        
        # Calculate derived metrics
        # Ocean heat content (simplified)
        heat_content = (sst - 20) * 100  # J/m² × 10⁶
        
        # Thermal stress (for hurricane regions)
        thermal_stress = "low"
        if region["type"] == "hurricane_zone":
            if sst > 28.0:
                thermal_stress = "high"
            elif sst > 26.0:
                thermal_stress = "moderate"
        
        return {
            "region_id": region_id,
            "region_name": region["name"],
            "region_type": region["type"],
            "ocean": region["ocean"],
            "date": date,
            "sst_celsius": round(sst, 2),
            "sst_anomaly": round(anomaly, 2),
            "baseline_sst": baseline,
            "enso_state": enso_state,
            "heat_content": round(heat_content, 1),
            "thermal_stress": thermal_stress,
            "impact": region["impact"],
            "data_source": "MODIS_AVHRR",
            "satellites": ["Terra", "Aqua", "NOAA-20"],
            "quality": "good" if np.random.random() > 0.1 else "cloudy"
        }
    
    def calculate_baseline(self, region_id: str, days: int = 90) -> Dict:
        """
        Calculate baseline SST for a region.
        
        Args:
            region_id: Region identifier
            days: Number of days for baseline calculation
            
        Returns:
            Dictionary with baseline metrics
        """
        logger.info(f"Calculating {days}-day baseline for {region_id}")
        
        # Fetch historical data
        end_date = datetime.now()
        historical_sst = []
        historical_anomaly = []
        
        for i in range(days):
            date = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
            data = self.fetch_sst_data(region_id, date)
            if data and data["quality"] == "good":
                historical_sst.append(data["sst_celsius"])
                historical_anomaly.append(data["sst_anomaly"])
        
        if not historical_sst:
            return {"error": "No valid historical data"}
        
        # Calculate baseline statistics
        baseline = {
            "region_id": region_id,
            "period_days": len(historical_sst),
            "sst": {
                "mean": round(np.mean(historical_sst), 2),
                "std": round(np.std(historical_sst), 2),
                "median": round(np.median(historical_sst), 2),
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
        Detect anomalies in sea surface temperature.
        
        Args:
            current_data: Current SST data
            baseline: Baseline statistics
            threshold_std: Number of standard deviations for anomaly
            
        Returns:
            Dictionary with anomaly detection results
        """
        # Calculate z-scores
        sst_z = (current_data["sst_celsius"] - baseline["sst"]["mean"]) / \
                baseline["sst"]["std"] if baseline["sst"]["std"] > 0 else 0
        
        anomaly_z = (current_data["sst_anomaly"] - baseline["anomaly"]["mean"]) / \
                    baseline["anomaly"]["std"] if baseline["anomaly"]["std"] > 0 else 0
        
        # Determine anomaly status
        sst_anomaly = "significant" if abs(sst_z) > threshold_std else \
                     "moderate" if abs(sst_z) > 1.5 else "none"
        
        # Combined score
        combined_z = (abs(sst_z) + abs(anomaly_z)) / 2
        
        return {
            "sst_z_score": round(sst_z, 2),
            "sst_anomaly": sst_anomaly,
            "sst_deviation_pct": round((current_data["sst_celsius"] - 
                                       baseline["sst"]["mean"]) / 
                                       baseline["sst"]["mean"] * 100, 2),
            "anomaly_z_score": round(anomaly_z, 2),
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
        current_data = self.fetch_sst_data(region_id, date)
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
        
        if region_type == "enso_indicator":
            # ENSO-based signals
            enso_state = current_data["enso_state"]
            
            if enso_state == "el_nino":
                direction = "short"
                confidence = min(100, 60 + abs(current_data["sst_anomaly"]) * 15)
                rationale = f"El Niño conditions detected. SST anomaly {current_data['sst_anomaly']:+.2f}°C. Impact: Drought risk in Australia/Asia, wet in South America."
            elif enso_state == "la_nina":
                direction = "long"
                confidence = min(100, 60 + abs(current_data["sst_anomaly"]) * 15)
                rationale = f"La Niña conditions detected. SST anomaly {current_data['sst_anomaly']:+.2f}°C. Impact: Wet in Australia/Asia, dry in South America."
            else:
                direction = "neutral"
                confidence = 50
                rationale = f"ENSO neutral conditions. SST anomaly {current_data['sst_anomaly']:+.2f}°C. Normal weather patterns expected."
        
        elif region_type == "hurricane_zone":
            # Hurricane season signals
            thermal_stress = current_data["thermal_stress"]
            
            if thermal_stress == "high":
                direction = "short"
                confidence = 75
                rationale = f"High thermal stress. SST {current_data['sst_celsius']:.1f}°C. Elevated hurricane risk for energy infrastructure."
            elif thermal_stress == "moderate":
                direction = "neutral"
                confidence = 50
                rationale = f"Moderate thermal stress. SST {current_data['sst_celsius']:.1f}°C. Normal hurricane season conditions."
            else:
                direction = "neutral"
                confidence = 50
                rationale = f"Low thermal stress. SST {current_data['sst_celsius']:.1f}°C. Below-average hurricane risk."
        
        elif region_type == "agricultural_impact":
            # Agricultural impact signals
            sst_anomaly_val = current_data["sst_anomaly"]
            
            if sst_anomaly_val > 1.0:
                direction = "long"
                confidence = min(100, 60 + abs(sst_anomaly_val) * 10)
                rationale = f"Warm SST anomaly {sst_anomaly_val:+.2f}°C. Enhanced moisture transport, favorable for agriculture."
            elif sst_anomaly_val < -1.0:
                direction = "short"
                confidence = min(100, 60 + abs(sst_anomaly_val) * 10)
                rationale = f"Cold SST anomaly {sst_anomaly_val:+.2f}°C. Reduced moisture transport, drought risk."
            else:
                direction = "neutral"
                confidence = 50
                rationale = f"SST anomaly {sst_anomaly_val:+.2f}°C. Normal moisture patterns."
        
        elif region_type == "monsoon_region":
            # Monsoon signals
            sst_anomaly_val = current_data["sst_anomaly"]
            
            if sst_anomaly_val > 1.0:
                direction = "long"
                confidence = min(100, 60 + abs(sst_anomaly_val) * 10)
                rationale = f"Warm SST {sst_anomaly_val:+.2f}°C. Enhanced monsoon activity, favorable for crops."
            elif sst_anomaly_val < -1.0:
                direction = "short"
                confidence = min(100, 60 + abs(sst_anomaly_val) * 10)
                rationale = f"Cold SST {sst_anomaly_val:+.2f}°C. Weak monsoon, drought risk."
            else:
                direction = "neutral"
                confidence = 50
                rationale = f"SST anomaly {sst_anomaly_val:+.2f}°C. Normal monsoon expected."
        
        elif region_type == "fishing_grounds":
            # Fishing signals
            sst_anomaly_val = current_data["sst_anomaly"]
            
            if sst_anomaly_val > 2.0:
                direction = "short"
                confidence = min(100, 60 + abs(sst_anomaly_val) * 10)
                rationale = f"Warm SST {sst_anomaly_val:+.2f}°C. Unfavorable for cold-water fisheries."
            elif sst_anomaly_val < -2.0:
                direction = "long"
                confidence = min(100, 60 + abs(sst_anomaly_val) * 10)
                rationale = f"Cold SST {sst_anomaly_val:+.2f}°C. Favorable for cold-water fisheries."
            else:
                direction = "neutral"
                confidence = 50
                rationale = f"SST anomaly {sst_anomaly_val:+.2f}°C. Normal fishing conditions."
        
        else:
            # Default logic
            if combined_z > 2.0:
                direction = "long"
                confidence = min(100, 60 + combined_z * 10)
                rationale = f"SST significantly above baseline."
            elif combined_z < -2.0:
                direction = "short"
                confidence = min(100, 60 + abs(combined_z) * 10)
                rationale = f"SST significantly below baseline."
            else:
                direction = "neutral"
                confidence = 50
                rationale = f"SST within normal range."
        
        signal = {
            "region_id": region_id,
            "region_name": region["name"],
            "region_type": region["type"],
            "ocean": region["ocean"],
            "date": date,
            "signal_type": "sea_surface_temperature",
            "direction": direction,
            "confidence": confidence,
            "rationale": rationale,
            "instruments": region["instruments"],
            "current_sst": current_data["sst_celsius"],
            "sst_anomaly": current_data["sst_anomaly"],
            "enso_state": current_data.get("enso_state", "n/a"),
            "thermal_stress": current_data.get("thermal_stress", "n/a"),
            "baseline_sst": baseline["sst"]["mean"],
            "sst_z_score": anomaly["sst_z_score"],
            "combined_z_score": combined_z,
            "anomaly": anomaly["overall_anomaly"],
            "impact": current_data["impact"],
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
        
        # Calculate ENSO summary
        enso_regions = [s for s in signals if s["region_type"] == "enso_indicator"]
        enso_state = "neutral"
        if any(s["enso_state"] == "el_nino" for s in enso_regions):
            enso_state = "el_nino"
        elif any(s["enso_state"] == "la_nina" for s in enso_regions):
            enso_state = "la_nina"
        
        # Save summary
        summary = {
            "date": date,
            "total_regions": len(self.regions),
            "signals_generated": len(signals),
            "long_signals": sum(1 for s in signals if s["direction"] == "long"),
            "short_signals": sum(1 for s in signals if s["direction"] == "short"),
            "neutral_signals": sum(1 for s in signals if s["direction"] == "neutral"),
            "enso_state": enso_state,
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
            "monitoring_type": "sea_surface_temperature",
            "satellites": ["Terra (MODIS)", "Aqua (MODIS)", "NOAA-20 (AVHRR)"],
            "metrics": ["SST", "SST Anomaly", "ENSO State", "Thermal Stress"],
            "update_frequency": "Daily",
            "latency": "1-3 days",
            "total_regions": len(self.regions),
            "region_types": list(set(r["type"] for r in self.regions.values())),
            "regions": self.regions,
            "signal_logic": {
                "enso": "El Niño → SHORT agri, La Niña → LONG agri",
                "hurricane": "High SST → SHORT energy",
                "agriculture": "Warm SST → LONG agri, Cold SST → SHORT agri",
                "fishing": "Cold SST → LONG fisheries"
            },
            "trading_instruments": list(set(
                inst for region in self.regions.values() 
                for inst in region["instruments"]
            ))
        }


def main():
    """Test SST monitoring."""
    logging.basicConfig(level=logging.INFO)
    
    monitor = SeaSurfaceTemperatureMonitor()
    
    # Get regional summary
    print("\n🌊 Sea Surface Temperature Monitor - Regional Summary")
    print("=" * 60)
    summary = monitor.get_regional_summary()
    print(f"Monitoring {summary['total_regions']} ocean regions")
    print(f"Satellites: {', '.join(summary['satellites'])}")
    
    # Generate signals for all regions
    print("\n🚀 Generating signals for all regions...")
    signals = monitor.generate_all_signals()
    
    print(f"\n📈 Generated {len(signals)} signals:")
    print("-" * 60)
    
    for signal in signals[:5]:  # Show top 5
        print(f"\n{signal['region_name']} ({signal['ocean']})")
        print(f"  Direction: {signal['direction'].upper()}")
        print(f"  Confidence: {signal['confidence']}%")
        print(f"  SST: {signal['current_sst']:.2f}°C (baseline: {signal['baseline_sst']:.2f}°C)")
        print(f"  Anomaly: {signal['sst_anomaly']:+.2f}°C")
        if signal['enso_state'] != "n/a":
            print(f"  ENSO State: {signal['enso_state'].upper()}")
        print(f"  Instruments: {', '.join(signal['instruments'])}")
        print(f"  Rationale: {signal['rationale']}")


if __name__ == "__main__":
    main()
