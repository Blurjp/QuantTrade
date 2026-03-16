"""
Nighttime Lights Monitoring Module

Uses VIIRS (Suomi NPP) satellite data to monitor economic activity via nighttime lights.
Leading indicator for GDP, industrial production, and economic health.

Data Source:
- VIIRS Day/Night Band (DNB)
- Available via Planetary Computer (free)
- Update frequency: daily
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


class NighttimeLightsMonitor:
    """Monitor economic activity using nighttime satellite imagery."""
    
    def __init__(
        self,
        output_base: str = "outputs",
        cache_days: int = 30
    ):
        """
        Initialize nighttime lights monitor.
        
        Args:
            output_base: Base directory for outputs
            cache_days: Number of days to cache data
        """
        self.output_base = Path(output_base)
        self.cache_days = cache_days
        
        # Target regions for monitoring
        self.regions = {
            # China - Industrial hubs
            "china_shanghai": {
                "name": "Shanghai Industrial Zone",
                "bbox": [120.8, 30.6, 122.2, 31.9],
                "country": "China",
                "type": "industrial",
                "instruments": ["FXI", "MCHI", "ASHR"],
                "description": "Yangtze River Delta industrial zone"
            },
            "china_guangdong": {
                "name": "Guangdong Manufacturing Hub",
                "bbox": [113.0, 22.0, 115.0, 24.0],
                "country": "China",
                "type": "manufacturing",
                "instruments": ["FXI", "MCHI", "KWEB"],
                "description": "Pearl River Delta manufacturing zone"
            },
            "china_beijing": {
                "name": "Beijing-Tianjin Industrial Area",
                "bbox": [115.5, 38.5, 118.0, 40.5],
                "country": "China",
                "type": "industrial",
                "instruments": ["FXI", "MCHI"],
                "description": "Northern China industrial region"
            },
            
            # USA - Key regions
            "usa_texas": {
                "name": "Texas Oil & Industrial Belt",
                "bbox": [-106.5, 25.8, -93.5, 36.5],
                "country": "USA",
                "type": "energy_industrial",
                "instruments": ["XLE", "XOM", "CVX", "OIH"],
                "description": "Permian Basin and Houston industrial area"
            },
            "usa_california": {
                "name": "California Tech & Ports",
                "bbox": [-124.5, 32.5, -114.0, 42.0],
                "country": "USA",
                "type": "tech_logistics",
                "instruments": ["QQQ", "XLK", "TECL"],
                "description": "Silicon Valley and LA ports"
            },
            "usa_midwest": {
                "name": "Midwest Manufacturing Belt",
                "bbox": [-97.0, 36.0, -80.0, 49.0],
                "country": "USA",
                "type": "manufacturing",
                "instruments": ["XLI", "CAT", "DE"],
                "description": "Rust Belt manufacturing region"
            },
            
            # Europe
            "europe_germany": {
                "name": "German Industrial Heartland",
                "bbox": [6.0, 47.0, 15.0, 55.0],
                "country": "Germany",
                "type": "industrial",
                "instruments": ["EWG", "FXD"],
                "description": "Rhine-Ruhr industrial region"
            },
            
            # India
            "india_mumbai": {
                "name": "Mumbai Industrial Corridor",
                "bbox": [72.0, 18.0, 73.5, 19.5],
                "country": "India",
                "type": "industrial",
                "instruments": ["INDA", "EPI"],
                "description": "Western India industrial zone"
            },
        }
        
        # Create output directory
        self.output_dir = self.output_base / "nighttime_lights"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_viirs_data(self, region_id: str, date: str) -> Optional[Dict]:
        """
        Fetch VIIRS nighttime lights data for a region.
        
        In production, this would use Planetary Computer API.
        For now, returns simulated data based on realistic patterns.
        
        Args:
            region_id: Region identifier
            date: Date string (YYYY-MM-DD)
            
        Returns:
            Dictionary with light intensity metrics
        """
        region = self.regions.get(region_id)
        if not region:
            logger.error(f"Unknown region: {region_id}")
            return None
        
        logger.info(f"Fetching VIIRS data for {region_id} on {date}")
        
        # Simulate realistic nighttime lights data
        # In production: use pystac-client to query Planetary Computer
        np.random.seed(hash(date + region_id) % 2**32)
        
        # Base light intensity (varies by region type)
        base_intensity = {
            "industrial": 85,
            "manufacturing": 80,
            "energy_industrial": 75,
            "tech_logistics": 90,
        }.get(region["type"], 70)
        
        # Add seasonal variation
        day_of_year = datetime.strptime(date, "%Y-%m-%d").timetuple().tm_yday
        seasonal_factor = 1 + 0.05 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Add economic cycle (slight upward trend + random noise)
        days_since_start = (datetime.strptime(date, "%Y-%m-%d") - 
                          datetime(2024, 1, 1)).days
        economic_trend = 1 + 0.0001 * days_since_start
        
        # Random daily variation (weather, holidays, etc.)
        daily_noise = np.random.normal(0, 3)
        
        # Calculate final intensity
        intensity = base_intensity * seasonal_factor * economic_trend + daily_noise
        intensity = max(0, min(100, intensity))  # Clamp to 0-100
        
        # Calculate area coverage (percentage of lit pixels)
        coverage = np.random.uniform(70, 95)
        
        # Calculate brightness distribution
        brightness_mean = intensity
        brightness_std = np.random.uniform(10, 20)
        
        return {
            "region_id": region_id,
            "region_name": region["name"],
            "country": region["country"],
            "date": date,
            "intensity": round(intensity, 2),
            "coverage_pct": round(coverage, 2),
            "brightness_mean": round(brightness_mean, 2),
            "brightness_std": round(brightness_std, 2),
            "data_source": "VIIRS_DNB",
            "satellite": "Suomi-NPP",
            "quality": "good" if np.random.random() > 0.1 else "cloudy"
        }
    
    def calculate_baseline(self, region_id: str, days: int = 90) -> Dict:
        """
        Calculate baseline light intensity for a region.
        
        Args:
            region_id: Region identifier
            days: Number of days for baseline calculation
            
        Returns:
            Dictionary with baseline metrics
        """
        logger.info(f"Calculating {days}-day baseline for {region_id}")
        
        # Fetch historical data
        end_date = datetime.now()
        historical_data = []
        
        for i in range(days):
            date = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
            data = self.fetch_viirs_data(region_id, date)
            if data and data["quality"] == "good":
                historical_data.append(data["intensity"])
        
        if not historical_data:
            return {"error": "No valid historical data"}
        
        # Calculate baseline statistics
        baseline = {
            "region_id": region_id,
            "period_days": len(historical_data),
            "mean": round(np.mean(historical_data), 2),
            "std": round(np.std(historical_data), 2),
            "median": round(np.median(historical_data), 2),
            "min": round(np.min(historical_data), 2),
            "max": round(np.max(historical_data), 2),
            "percentile_25": round(np.percentile(historical_data, 25), 2),
            "percentile_75": round(np.percentile(historical_data, 75), 2),
        }
        
        return baseline
    
    def detect_anomaly(
        self,
        current_intensity: float,
        baseline: Dict,
        threshold_std: float = 2.0
    ) -> Dict:
        """
        Detect anomalies in light intensity.
        
        Args:
            current_intensity: Current light intensity
            baseline: Baseline statistics
            threshold_std: Number of standard deviations for anomaly
            
        Returns:
            Dictionary with anomaly detection results
        """
        mean = baseline["mean"]
        std = baseline["std"]
        
        # Calculate z-score
        z_score = (current_intensity - mean) / std if std > 0 else 0
        
        # Determine anomaly status
        if abs(z_score) > threshold_std:
            anomaly = "significant" if abs(z_score) > 2.5 else "moderate"
        else:
            anomaly = "none"
        
        # Calculate percentile rank
        percentile_rank = min(100, max(0, 
            50 + z_score * 16  # Approximate percentile from z-score
        ))
        
        return {
            "z_score": round(z_score, 2),
            "anomaly": anomaly,
            "percentile_rank": round(percentile_rank, 1),
            "deviation_pct": round((current_intensity - mean) / mean * 100, 2),
            "status": "above_baseline" if z_score > 0 else "below_baseline"
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
        current_data = self.fetch_viirs_data(region_id, date)
        if not current_data:
            return {"error": "Failed to fetch current data"}
        
        # Calculate baseline
        baseline = self.calculate_baseline(region_id, baseline_days)
        if "error" in baseline:
            return {"error": baseline["error"]}
        
        # Detect anomaly
        anomaly = self.detect_anomaly(
            current_data["intensity"],
            baseline
        )
        
        # Generate signal
        # Logic: 
        # - Lights up significantly → Economic activity increasing → LONG
        # - Lights down significantly → Economic activity decreasing → SHORT
        # - No change → NEUTRAL
        
        z_score = anomaly["z_score"]
        
        if z_score > 2.0:
            direction = "long"
            confidence = min(100, 60 + abs(z_score) * 10)
            rationale = f"Light intensity {anomaly['deviation_pct']:+.1f}% above baseline. Strong economic activity increase detected."
        elif z_score < -2.0:
            direction = "short"
            confidence = min(100, 60 + abs(z_score) * 10)
            rationale = f"Light intensity {anomaly['deviation_pct']:+.1f}% below baseline. Economic activity slowdown detected."
        else:
            direction = "neutral"
            confidence = 50
            rationale = f"Light intensity within normal range ({anomaly['deviation_pct']:+.1f}% from baseline)."
        
        signal = {
            "region_id": region_id,
            "region_name": region["name"],
            "country": region["country"],
            "date": date,
            "signal_type": "nighttime_lights",
            "direction": direction,
            "confidence": confidence,
            "rationale": rationale,
            "instruments": region["instruments"],
            "current_intensity": current_data["intensity"],
            "baseline_mean": baseline["mean"],
            "baseline_std": baseline["std"],
            "z_score": z_score,
            "anomaly": anomaly["anomaly"],
            "percentile_rank": anomaly["percentile_rank"],
            "deviation_pct": anomaly["deviation_pct"],
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
            "signals": signals,
            "timestamp": datetime.now().isoformat()
        }
        
        summary_file = self.output_dir / f"summary_{date}.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        logger.info(f"Summary saved to {summary_file}")
        
        return signals
    
    def get_regional_summary(self) -> Dict:
        """
        Get summary of all monitored regions.
        
        Returns:
            Dictionary with regional information
        """
        return {
            "monitoring_type": "nighttime_lights",
            "satellite": "Suomi-NPP (VIIRS)",
            "update_frequency": "daily",
            "latency": "1-3 days",
            "total_regions": len(self.regions),
            "regions": self.regions,
            "signal_logic": {
                "long": "Light intensity > 2σ above baseline (economic expansion)",
                "short": "Light intensity > 2σ below baseline (economic contraction)",
                "neutral": "Light intensity within normal range"
            },
            "trading_instruments": list(set(
                inst for region in self.regions.values() 
                for inst in region["instruments"]
            ))
        }


def main():
    """Test nighttime lights monitoring."""
    logging.basicConfig(level=logging.INFO)
    
    monitor = NighttimeLightsMonitor()
    
    # Get regional summary
    print("\n📊 Nighttime Lights Monitor - Regional Summary")
    print("=" * 60)
    summary = monitor.get_regional_summary()
    print(f"Monitoring {summary['total_regions']} regions")
    print(f"Satellite: {summary['satellite']}")
    print(f"Update frequency: {summary['update_frequency']}")
    
    # Generate signals for all regions
    print("\n🚀 Generating signals for all regions...")
    signals = monitor.generate_all_signals()
    
    print(f"\n📈 Generated {len(signals)} signals:")
    print("-" * 60)
    
    for signal in signals[:5]:  # Show top 5
        print(f"\n{signal['region_name']} ({signal['country']})")
        print(f"  Direction: {signal['direction'].upper()}")
        print(f"  Confidence: {signal['confidence']}%")
        print(f"  Z-score: {signal['z_score']:+.2f}")
        print(f"  Intensity: {signal['current_intensity']:.1f} (baseline: {signal['baseline_mean']:.1f})")
        print(f"  Instruments: {', '.join(signal['instruments'])}")
        print(f"  Rationale: {signal['rationale']}")


if __name__ == "__main__":
    main()
