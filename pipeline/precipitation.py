"""
Precipitation Monitoring Module

Uses satellite data to monitor global rainfall patterns for predicting crop yields,
hydroelectric generation, and water scarcity impacts on commodities.

Data Source:
- GPM (Global Precipitation Measurement): Rainfall rates
- IMERG (Integrated Multi-satellitE Retrievals): Precipitation estimates
- Available via NASA/GES DISC (free)
- Update frequency: Daily
- Latency: 1-3 days

Supports both real satellite data (via NASA GES DISC API) and simulated data
for testing and development purposes.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PrecipitationMonitor:
    """Monitor precipitation for commodity trading signals."""
    
    def __init__(
        self,
        output_base: str = "outputs",
        cache_days: int = 30
    ):
        """
        Initialize precipitation monitor.

        Args:
            output_base: Base directory for outputs
            cache_days: Number of days to cache data
        """
        self.output_base = Path(output_base)
        self.cache_days = cache_days
        
        # Key agricultural regions for monitoring
        self.regions = {
            # USA - Major Crop Regions
            "usa_corn_belt": {
                "name": "US Corn Belt",
                "bbox": [-100.0, 36.0, -82.0, 48.0],
                "country": "USA",
                "type": "row_crops",
                "instruments": ["CORN", "SOYB", "WEAT"],
                "description": "Iowa, Illinois, Nebraska corn/soybeans",
                "baseline_precip_mm": 85,  # mm/month
                "critical_months": [4, 5, 6, 7, 8],  # Growing season
                "crops": ["corn", "soybeans"]
            },
            "usa_winter_wheat": {
                "name": "US Winter Wheat Belt",
                "bbox": [-105.0, 32.0, -95.0, 42.0],
                "country": "USA",
                "type": "row_crops",
                "instruments": ["WEAT", "KWK"],
                "description": "Kansas, Oklahoma, Texas wheat",
                "baseline_precip_mm": 65,
                "critical_months": [3, 4, 5, 6],
                "crops": ["winter_wheat"]
            },
            "usa_cotton_belt": {
                "name": "US Cotton Belt",
                "bbox": [-105.0, 28.0, -82.0, 38.0],
                "country": "USA",
                "type": "row_crops",
                "instruments": ["COTTON", "BAL"],
                "description": "Texas, Georgia, North Carolina cotton",
                "baseline_precip_mm": 95,
                "critical_months": [4, 5, 6, 7, 8, 9],
                "crops": ["cotton"]
            },
            
            # South America
            "brazil_soybeans": {
                "name": "Brazil Soybean Region",
                "bbox": [-60.0, -30.0, -45.0, -15.0],
                "country": "Brazil",
                "type": "row_crops",
                "instruments": ["SOYB", "CORN"],
                "description": "Mato Grosso, Paraná soybeans",
                "baseline_precip_mm": 180,
                "critical_months": [10, 11, 12, 1, 2, 3],
                "crops": ["soybeans", "corn"]
            },
            "argentina_pampas": {
                "name": "Argentina Pampas",
                "bbox": [-65.0, -40.0, -56.0, -30.0],
                "country": "Argentina",
                "type": "row_crops",
                "instruments": ["SOYB", "CORN", "WEAT"],
                "description": "Buenos Aires, Santa Fe crops",
                "baseline_precip_mm": 95,
                "critical_months": [10, 11, 12, 1, 2, 3],
                "crops": ["soybeans", "corn", "wheat"]
            },
            
            # Asia
            "india_monsoon": {
                "name": "India Monsoon Region",
                "bbox": [68.0, 8.0, 92.0, 30.0],
                "country": "India",
                "type": "monsoon_agriculture",
                "instruments": ["COTTON", "SUGAR", "RICE", "TEA"],
                "description": "Major Indian agricultural zones",
                "baseline_precip_mm": 250,
                "critical_months": [6, 7, 8, 9, 10],
                "crops": ["cotton", "sugar", "rice", "tea"]
            },
            "china_wheat": {
                "name": "China Wheat Region",
                "bbox": [105.0, 30.0, 122.0, 42.0],
                "country": "China",
                "type": "row_crops",
                "instruments": ["FXI", "WEAT"],
                "description": "North China Plain wheat/corn",
                "baseline_precip_mm": 70,
                "critical_months": [3, 4, 5, 6, 9, 10],
                "crops": ["wheat", "corn"]
            },
            "australia_wheat": {
                "name": "Australia Wheat Belt",
                "bbox": [115.0, -35.0, 152.0, -25.0],
                "country": "Australia",
                "type": "row_crops",
                "instruments": ["WEAT", "AWB"],
                "description": "Western Australia wheat",
                "baseline_precip_mm": 50,
                "critical_months": [5, 6, 7, 8, 9, 10],
                "crops": ["wheat", "barley"]
            },
            
            # Africa
            "west_africa_cocoa": {
                "name": "West Africa Cocoa Belt",
                "bbox": [-8.0, 4.0, 0.0, 10.0],
                "country": "Multiple",
                "type": "tree_crops",
                "instruments": ["NIB", "CHOC"],
                "description": "Ivory Coast, Ghana cocoa",
                "baseline_precip_mm": 150,
                "critical_months": [3, 4, 5, 6, 7, 8, 9, 10],
                "crops": ["cocoa"]
            },
        }
        
        # Create output directory
        self.output_dir = self.output_base / "precipitation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_precipitation_data(self, region_id: str, date: str) -> Optional[Dict]:
        """
        Fetch precipitation data for a region.

        Tries real satellite data first if available, falls back to simulated data.

        Args:
            region_id: Region identifier
            date: Date string (YYYY-MM-DD)

        Returns:
            Dictionary with precipitation metrics
        """
        region = self.regions.get(region_id)
        if not region:
            logger.error(f"Unknown region: {region_id}")
            return None

        logger.info(f"Fetching precipitation data for {region_id} on {date}")

        # Try real data first, fallback to simulated
        real_data = self._fetch_real_precipitation(region_id, region, date)
        if real_data:
            logger.info(f"Using real NASA GPM precipitation data for {region_id}")
            return real_data

        # Fallback to simulated data
        logger.info(f"NASA GPM data unavailable, using simulated precipitation data for {region_id}")
        return self._fetch_simulated_precipitation(region_id, region, date)

    def _fetch_real_precipitation(self, region_id: str, region: Dict, date: str) -> Optional[Dict]:
        """
        Fetch real precipitation data from NASA GES DISC (GPM/IMERG).

        Args:
            region_id: Region identifier
            region: Region configuration dictionary
            date: Date string (YYYY-MM-DD)

        Returns:
            Dictionary with precipitation metrics or None if fetch failed
        """
        # DISABLED: NASA GPM data fetch is causing hangs due to 404 errors
        # TODO: Re-enable once we fix the fetch logic or NASA data is available
        logger.debug(f"NASA GPM data fetch disabled - using simulated data")
        return None
        
        try:
            from pipeline.satellite_data import NASAGESDISCFetcher, DataCache, is_real_data_available

            # Check if real data is available
            if not is_real_data_available():
                logger.debug("Real satellite data not available, using simulated data")
                return None

            cache = DataCache()
            fetcher = NASAGESDISCFetcher(cache)

            # Get bounding box for region
            bbox = region["bbox"]

            # Fetch precipitation data from NASA GES DISC
            result = fetcher.fetch_precipitation(
                bbox=bbox,
                date=date,
                days_range=7
            )

            if not result:
                return None

            # Extract values from real data
            daily_precip = result.get("daily_precip_mm", 0)
            monthly_precip = daily_precip * 30  # Extrapolate to monthly
            baseline = region["baseline_precip_mm"]
            month = datetime.strptime(date, "%Y-%m-%d").month

            # Calculate anomaly
            precip_anomaly = monthly_precip - baseline
            precip_anomaly_pct = (monthly_precip - baseline) / baseline * 100 if baseline > 0 else 0

            # Determine if in critical growing season
            is_critical_season = month in region["critical_months"]

            # Determine drought/flood status
            if precip_anomaly_pct < -40:
                status = "severe_drought"
            elif precip_anomaly_pct < -20:
                status = "drought"
            elif precip_anomaly_pct < -10:
                status = "dry"
            elif precip_anomaly_pct > 40:
                status = "flood"
            elif precip_anomaly_pct > 20:
                status = "wet"
            elif precip_anomaly_pct > 10:
                status = "slightly_wet"
            else:
                status = "normal"

            # Calculate impact score (0-100)
            if is_critical_season:
                impact_multiplier = 1.5
            else:
                impact_multiplier = 0.7

            impact_score = min(100, abs(precip_anomaly_pct) * impact_multiplier)

            return {
                "region_id": region_id,
                "region_name": region["name"],
                "region_type": region["type"],
                "country": region["country"],
                "date": date,
                "month": month,
                "daily_precip_mm": round(daily_precip, 2),
                "monthly_precip_estimate_mm": round(monthly_precip, 1),
                "baseline_precip_mm": baseline,
                "precip_anomaly_mm": round(precip_anomaly, 1),
                "precip_anomaly_pct": round(precip_anomaly_pct, 2),
                "status": status,
                "is_critical_season": is_critical_season,
                "impact_score": round(impact_score, 1),
                "crops": region["crops"],
                "data_source": "GPM_IMERG_REAL",
                "satellites": ["GPM", "IMERG"],
                "quality": "good",
                "days_averaged": result.get("days_averaged", 7)
            }

        except ImportError as e:
            logger.warning(f"satellite_data module not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch real precipitation data: {e}")
            return None

    def _fetch_simulated_precipitation(self, region_id: str, region: Dict, date: str) -> Optional[Dict]:
        """
        Fetch simulated precipitation data for testing and development.

        Args:
            region_id: Region identifier
            region: Region configuration dictionary
            date: Date string (YYYY-MM-DD)

        Returns:
            Dictionary with precipitation metrics
        """
        # Simulate realistic precipitation data
        # In production: use NASA GES DISC API for GPM/IMERG data
        np.random.seed(hash(date + region_id) % 2**32)

        # Get baseline precipitation
        baseline = region["baseline_precip_mm"]

        # Add seasonal variation
        month = datetime.strptime(date, "%Y-%m-%d").month
        day_of_year = datetime.strptime(date, "%Y-%m-%d").timetuple().tm_yday

        # Determine if in critical growing season
        is_critical_season = month in region["critical_months"]

        # Seasonal factor (varies by region and hemisphere)
        if region_id in ["australia_wheat"]:
            # Southern hemisphere - opposite seasons
            seasonal_factor = 1.5 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
        else:
            # Northern hemisphere
            seasonal_factor = 1.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

        # Add weather system variation
        weather_factor = np.random.uniform(0.3, 2.5)

        # Random daily variation
        daily_noise = np.random.normal(0, baseline * 0.15)

        # Calculate actual precipitation (daily, then monthly estimate)
        daily_precip = (baseline / 30) * (1 + seasonal_factor * 0.3) * weather_factor + daily_noise
        daily_precip = max(0, daily_precip)

        # Monthly estimate (extrapolate from daily)
        monthly_precip = daily_precip * 30

        # Calculate anomaly
        precip_anomaly = monthly_precip - baseline
        precip_anomaly_pct = (monthly_precip - baseline) / baseline * 100

        # Determine drought/flood status
        if precip_anomaly_pct < -40:
            status = "severe_drought"
        elif precip_anomaly_pct < -20:
            status = "drought"
        elif precip_anomaly_pct < -10:
            status = "dry"
        elif precip_anomaly_pct > 40:
            status = "flood"
        elif precip_anomaly_pct > 20:
            status = "wet"
        elif precip_anomaly_pct > 10:
            status = "slightly_wet"
        else:
            status = "normal"

        # Calculate impact score (0-100)
        # During critical season, deviations have more impact
        if is_critical_season:
            impact_multiplier = 1.5
        else:
            impact_multiplier = 0.7

        impact_score = min(100, abs(precip_anomaly_pct) * impact_multiplier)

        return {
            "region_id": region_id,
            "region_name": region["name"],
            "region_type": region["type"],
            "country": region["country"],
            "date": date,
            "month": month,
            "daily_precip_mm": round(daily_precip, 2),
            "monthly_precip_estimate_mm": round(monthly_precip, 1),
            "baseline_precip_mm": baseline,
            "precip_anomaly_mm": round(precip_anomaly, 1),
            "precip_anomaly_pct": round(precip_anomaly_pct, 2),
            "status": status,
            "is_critical_season": is_critical_season,
            "impact_score": round(impact_score, 1),
            "crops": region["crops"],
            "data_source": "GPM_IMERG_SIMULATED",
            "satellites": ["GPM", "IMERG"],
            "quality": "good" if np.random.random() > 0.1 else "partial"
        }
    
    def calculate_baseline(self, region_id: str, days: int = 30) -> Dict:
        """
        Calculate baseline precipitation for a region.
        
        Args:
            region_id: Region identifier
            days: Number of days for baseline calculation (limited to 30)
            
        Returns:
            Dictionary with baseline metrics
        """
        # Limit days to prevent hanging on historical data fetch
        days = min(days, 30)
        
        logger.info(f"Calculating {days}-day baseline for {region_id}")
        
        # Fetch historical data, skipping recent 5 days (data latency)
        end_date = datetime.now() - timedelta(days=5)  # Skip recent days due to data latency
        historical_precip = []
        historical_anomaly = []
        
        for i in range(days):
            date = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
            data = self.fetch_precipitation_data(region_id, date)
            if data and data["quality"] == "good":
                historical_precip.append(data["monthly_precip_estimate_mm"])
                historical_anomaly.append(data["precip_anomaly_pct"])
        
        if not historical_precip:
            # Return default baseline if no data available
            region = self.regions.get(region_id, {})
            return {
                "region_id": region_id,
                "period_days": 0,
                "precipitation": {
                    "mean": region.get("baseline_precip_mm", 85.0),
                    "std": 20.0,
                    "median": region.get("baseline_precip_mm", 85.0),
                },
                "anomaly": {
                    "mean": 0.0,
                    "std": 10.0,
                    "median": 0.0,
                }
            }
        
        # Calculate baseline statistics
        baseline = {
            "region_id": region_id,
            "period_days": len(historical_precip),
            "precipitation": {
                "mean": round(np.mean(historical_precip), 1),
                "std": round(np.std(historical_precip), 1),
                "median": round(np.median(historical_precip), 1),
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
        Detect anomalies in precipitation.
        
        Args:
            current_data: Current precipitation data
            baseline: Baseline statistics
            threshold_std: Number of standard deviations for anomaly
            
        Returns:
            Dictionary with anomaly detection results
        """
        # Calculate z-scores
        precip_z = (current_data["monthly_precip_estimate_mm"] - 
                   baseline["precipitation"]["mean"]) / \
                   baseline["precipitation"]["std"] if baseline["precipitation"]["std"] > 0 else 0
        
        # Determine anomaly status
        precip_anomaly = "significant" if abs(precip_z) > threshold_std else \
                        "moderate" if abs(precip_z) > 1.5 else "none"
        
        return {
            "precip_z_score": round(precip_z, 2),
            "precip_anomaly": precip_anomaly,
            "precip_deviation_pct": round((current_data["monthly_precip_estimate_mm"] - 
                                          baseline["precipitation"]["mean"]) / 
                                          baseline["precipitation"]["mean"] * 100, 2),
            "combined_z_score": round(abs(precip_z), 2),
            "overall_anomaly": "significant" if abs(precip_z) > 2.0 else \
                              "moderate" if abs(precip_z) > 1.5 else "none"
        }
    
    def generate_signal(
        self,
        region_id: str,
        date: Optional[str] = None,
        baseline_days: int = 30  # Reduced from 90 to prevent hanging
    ) -> Dict:
        """
        Generate trading signal for a region.
        
        Args:
            region_id: Region identifier
            date: Date for signal (default: today)
            baseline_days: Days for baseline calculation (default 30, max 30)
            
        Returns:
            Dictionary with signal information
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Limit baseline_days to prevent hanging on historical data fetch
        baseline_days = min(baseline_days, 30)
        
        region = self.regions.get(region_id)
        if not region:
            return {"error": f"Unknown region: {region_id}"}
        
        logger.info(f"Generating signal for {region_id} on {date}")
        
        # Fetch current data
        current_data = self.fetch_precipitation_data(region_id, date)
        if not current_data:
            return {"error": "Failed to fetch current data"}
        
        # Calculate baseline
        baseline = self.calculate_baseline(region_id, baseline_days)
        if "error" in baseline:
            return {"error": baseline["error"]}
        
        # Detect anomaly
        anomaly = self.detect_anomaly(current_data, baseline)
        
        # Generate signal based on precipitation status
        status = current_data["status"]
        is_critical = current_data["is_critical_season"]
        
        if status in ["severe_drought", "drought"]:
            direction = "short"
            confidence = min(100, 60 + current_data["impact_score"] * 0.5)
            if is_critical:
                confidence = min(100, confidence * 1.2)
            rationale = f"Drought conditions in {region['name']}. Precipitation {current_data['precip_anomaly_pct']:.1f}% below normal. Crop yield at risk."
        
        elif status == "dry":
            if is_critical:
                direction = "short"
                confidence = 60
                rationale = f"Dry conditions during critical growing season. Precipitation {current_data['precip_anomaly_pct']:.1f}% below normal."
            else:
                direction = "neutral"
                confidence = 50
                rationale = f"Dry conditions outside critical season. Limited crop impact."
        
        elif status in ["flood", "wet"]:
            direction = "short"
            confidence = min(100, 55 + current_data["impact_score"] * 0.4)
            rationale = f"Excessive rainfall in {region['name']}. Precipitation +{abs(current_data['precip_anomaly_pct']):.1f}% above normal. Flood damage risk."
        
        elif status == "slightly_wet":
            if is_critical:
                direction = "long"
                confidence = 55
                rationale = f"Adequate rainfall during growing season. Favorable for crop development."
            else:
                direction = "neutral"
                confidence = 50
                rationale = f"Slightly wet conditions. Normal crop development expected."
        
        else:  # normal
            direction = "long"
            confidence = 55
            rationale = f"Normal precipitation levels in {region['name']}. {current_data['precip_anomaly_pct']:+.1f}% from baseline. Good growing conditions."
        
        signal = {
            "region_id": region_id,
            "region_name": region["name"],
            "region_type": region["type"],
            "country": region["country"],
            "date": date,
            "signal_type": "precipitation",
            "direction": direction,
            "confidence": round(confidence, 1),
            "rationale": rationale,
            "instruments": region["instruments"],
            "current_precip_mm": current_data["monthly_precip_estimate_mm"],
            "precip_anomaly_pct": current_data["precip_anomaly_pct"],
            "status": status,
            "is_critical_season": is_critical,
            "baseline_precip_mm": baseline["precipitation"]["mean"],
            "precip_z_score": anomaly["precip_z_score"],
            "impact_score": current_data["impact_score"],
            "crops": current_data["crops"],
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
            "flood_regions": sum(1 for s in signals if s["status"] in ["flood", "wet"]),
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
            "monitoring_type": "precipitation",
            "satellites": ["GPM", "IMERG"],
            "metrics": ["Precipitation", "Anomaly", "Drought Status"],
            "update_frequency": "Daily",
            "latency": "1-3 days",
            "total_regions": len(self.regions),
            "region_types": list(set(r["type"] for r in self.regions.values())),
            "regions": self.regions,
            "signal_logic": {
                "drought": "SHORT crops (yield risk)",
                "flood": "SHORT crops (damage risk)",
                "normal": "LONG crops (good conditions)",
                "critical_season": "Higher impact during growing season"
            },
            "trading_instruments": list(set(
                inst for region in self.regions.values() 
                for inst in region["instruments"]
            ))
        }


def main():
    """Test precipitation monitoring."""
    logging.basicConfig(level=logging.INFO)
    
    monitor = PrecipitationMonitor()
    
    # Get regional summary
    print("\n🌧️ Precipitation Monitor - Regional Summary")
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
        print(f"  Precipitation: {signal['current_precip_mm']:.1f} mm/month (baseline: {signal['baseline_precip_mm']:.1f})")
        print(f"  Status: {signal['status'].upper()}")
        print(f"  Anomaly: {signal['precip_anomaly_pct']:+.1f}%")
        print(f"  Critical Season: {'YES' if signal['is_critical_season'] else 'NO'}")
        print(f"  Crops: {', '.join(signal['crops'])}")
        print(f"  Instruments: {', '.join(signal['instruments'])}")
        print(f"  Rationale: {signal['rationale']}")


if __name__ == "__main__":
    main()
