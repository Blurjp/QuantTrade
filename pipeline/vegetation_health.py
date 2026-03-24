"""
Vegetation Health Monitoring Module

Uses satellite NDVI/EVI data to monitor vegetation health for predicting crop yields,
forest conditions, and agricultural productivity.

Data Source:
- MODIS (Terra/Aqua): NDVI, EVI
- Sentinel-2: High-resolution vegetation indices
- Available via Planetary Computer (free)
- Update frequency: Daily
- Latency: 1-3 days

Supports real satellite data via pipeline.satellite_data module.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _confidence_label(score: float) -> str:
    if score >= 75:
        return "High"
    if score >= 60:
        return "Medium"
    return "Low"


class VegetationHealthMonitor:
    """Monitor vegetation health for commodity trading signals."""
    
    def __init__(
        self,
        output_base: str = "outputs",
        cache_days: int = 30
    ):
        """
        Initialize vegetation health monitor.

        Args:
            output_base: Base directory for outputs
            cache_days: Number of days to cache data
        """
        self.output_base = Path(output_base)
        self.cache_days = cache_days
        
        # Key agricultural and forestry regions
        self.regions = {
            # USA - Major Crop Regions
            "usa_corn_soybeans": {
                "name": "US Corn & Soybeans Belt",
                "bbox": [-100.0, 36.0, -82.0, 48.0],
                "country": "USA",
                "type": "row_crops",
                "instruments": ["CORN", "SOYB"],
                "description": "Iowa, Illinois, Indiana corn/soybeans",
                "baseline_ndvi": 0.65,
                "critical_months": [6, 7, 8, 9],
                "crops": ["corn", "soybeans"]
            },
            "usa_wheat_plains": {
                "name": "US Wheat Plains",
                "bbox": [-105.0, 32.0, -95.0, 45.0],
                "country": "USA",
                "type": "row_crops",
                "instruments": ["WEAT", "KWK"],
                "description": "Kansas, North Dakota wheat",
                "baseline_ndvi": 0.45,
                "critical_months": [4, 5, 6, 7],
                "crops": ["winter_wheat", "spring_wheat"]
            },
            
            # South America
            "brazil_cerrado": {
                "name": "Brazil Cerrado Soybeans",
                "bbox": [-58.0, -20.0, -45.0, -10.0],
                "country": "Brazil",
                "type": "row_crops",
                "instruments": ["SOYB", "CORN"],
                "description": "Mato Grosso, Goiás soybeans",
                "baseline_ndvi": 0.60,
                "critical_months": [11, 12, 1, 2],
                "crops": ["soybeans", "corn"]
            },
            "argentina_pampas": {
                "name": "Argentina Pampas",
                "bbox": [-65.0, -40.0, -56.0, -30.0],
                "country": "Argentina",
                "type": "row_crops",
                "instruments": ["SOYB", "CORN", "WEAT"],
                "description": "Buenos Aires crops",
                "baseline_ndvi": 0.55,
                "critical_months": [11, 12, 1, 2, 3],
                "crops": ["soybeans", "corn", "wheat"]
            },
            
            # Europe
            "europe_wheat_belt": {
                "name": "European Wheat Belt",
                "bbox": [-5.0, 42.0, 30.0, 55.0],
                "country": "Multiple",
                "type": "row_crops",
                "instruments": ["WEAT", "EXI1"],
                "description": "France, Germany, Poland wheat",
                "baseline_ndvi": 0.58,
                "critical_months": [4, 5, 6, 7],
                "crops": ["wheat", "barley"]
            },
            "ukraine_grain": {
                "name": "Ukraine Grain Region",
                "bbox": [22.0, 45.0, 40.0, 52.0],
                "country": "Ukraine",
                "type": "row_crops",
                "instruments": ["WEAT", "CORN"],
                "description": "Ukraine wheat and corn",
                "baseline_ndvi": 0.52,
                "critical_months": [4, 5, 6, 7],
                "crops": ["wheat", "corn", "sunflower"]
            },
            
            # Asia
            "india_punjab": {
                "name": "India Punjab Wheat",
                "bbox": [73.0, 28.0, 77.0, 33.0],
                "country": "India",
                "type": "row_crops",
                "instruments": ["RICE", "WHEAT"],
                "description": "Punjab wheat and rice",
                "baseline_ndvi": 0.50,
                "critical_months": [11, 12, 1, 2, 3, 10],
                "crops": ["wheat", "rice"]
            },
            "china_northeast": {
                "name": "China Northeast Corn Belt",
                "bbox": [120.0, 40.0, 135.0, 53.0],
                "country": "China",
                "type": "row_crops",
                "instruments": ["FXI", "CORN"],
                "description": "Heilongjiang, Jilin corn/soybeans",
                "baseline_ndvi": 0.62,
                "critical_months": [5, 6, 7, 8, 9],
                "crops": ["corn", "soybeans", "rice"]
            },
            
            # Forestry
            "brazil_amazon": {
                "name": "Amazon Rainforest",
                "bbox": [-75.0, -15.0, -45.0, 0.0],
                "country": "Brazil",
                "type": "forest",
                "instruments": ["WOOD", "PAPER"],
                "description": "Amazon timber and pulp",
                "baseline_ndvi": 0.85,
                "critical_months": list(range(1, 13)),  # Year-round
                "products": ["timber", "pulp", "carbon_credits"]
            },
            "indonesia_palm": {
                "name": "Indonesia Palm Oil",
                "bbox": [95.0, -10.0, 141.0, 6.0],
                "country": "Indonesia",
                "type": "plantation",
                "instruments": ["PALM", "CPO"],
                "description": "Sumatra, Kalimantan palm oil",
                "baseline_ndvi": 0.75,
                "critical_months": list(range(1, 13)),
                "products": ["palm_oil", "palm_kernel"]
            },
        }
        
        # Create output directory
        self.output_dir = self.output_base / "vegetation_health"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_ndvi_data(self, region_id: str, date: str) -> Optional[Dict]:
        """
        Fetch NDVI data for a region.

        Tries real satellite data first (auto-detected), falls back to simulated.

        Args:
            region_id: Region identifier
            date: Date string (YYYY-MM-DD)

        Returns:
            Dictionary with NDVI metrics
        """
        region = self.regions.get(region_id)
        if not region:
            logger.error(f"Unknown region: {region_id}")
            return None

        logger.info(f"Fetching NDVI data for {region_id} on {date}")

        fallback_reason = None

        # Try real data first (auto-detection)
        real_data = self._fetch_real_ndvi(region_id, date)
        if real_data:
            return real_data

        # Fallback to simulated data
        fallback_reason = "real_data_unavailable"
        logger.info(f"Real data unavailable for {region_id}, falling back to simulated")
        return self._fetch_simulated_ndvi(region_id, date, fallback_reason=fallback_reason)

    def _fetch_real_ndvi(self, region_id: str, date: str) -> Optional[Dict]:
        """
        Fetch real NDVI data from Planetary Computer.

        Uses auto-detection to determine if real data is available.

        Args:
            region_id: Region identifier
            date: Date string (YYYY-MM-DD)

        Returns:
            Dictionary with NDVI metrics or None if fetch failed
        """
        try:
            from pipeline.satellite_data import PlanetaryComputerFetcher, is_real_data_available

            # Check if real data is available via auto-detection
            if not is_real_data_available():
                logger.debug("Real satellite data not available (auto-detected)")
                return None

            region = self.regions[region_id]
            bbox = region["bbox"]

            fetcher = PlanetaryComputerFetcher()

            # Search for Sentinel-2 items with low cloud cover
            items = fetcher.search_items(
                collection="sentinel2",
                bbox=bbox,
                date=date,
                days_range=7,
                query={"eo:cloud_cover": {"lt": 30}}
            )

            if not items:
                logger.info(f"No Sentinel-2 items found for {region_id}")
                return None

            # Load data and compute NDVI
            ds = fetcher.load_data(items, ["B04", "B08"], bbox=bbox)
            if ds is None:
                return None

            stats = fetcher.compute_band_statistics(ds, "", compute_ndvi=True)

            if "ndvi_mean" not in stats:
                return None

            ndvi = stats["ndvi_mean"]
            baseline = region["baseline_ndvi"]
            ndvi_anomaly = ndvi - baseline
            ndvi_anomaly_pct = (ndvi - baseline) / baseline * 100 if baseline > 0 else 0

            # Determine status
            if ndvi_anomaly_pct < -20:
                status = "severe_stress"
            elif ndvi_anomaly_pct < -10:
                status = "stress"
            elif ndvi_anomaly_pct < -5:
                status = "slight_stress"
            elif ndvi_anomaly_pct > 10:
                status = "excellent"
            elif ndvi_anomaly_pct > 5:
                status = "good"
            else:
                status = "normal"

            month = datetime.strptime(date, "%Y-%m-%d").month
            is_critical_season = month in region["critical_months"]

            return {
                "region_id": region_id,
                "region_name": region["name"],
                "region_type": region["type"],
                "country": region["country"],
                "date": date,
                "month": month,
                "ndvi": round(ndvi, 3),
                "evi": round(stats.get("ndvi_mean", ndvi) * 0.85, 3),
                "baseline_ndvi": baseline,
                "ndvi_anomaly": round(ndvi_anomaly, 3),
                "ndvi_anomaly_pct": round(ndvi_anomaly_pct, 2),
                "status": status,
                "is_critical_season": is_critical_season,
                "impact_score": round(min(100, abs(ndvi_anomaly_pct) * (1.5 if is_critical_season else 0.7)), 1),
                "lai_estimate": round(max(0, 6.0 * ndvi), 2),
                "chlorophyll_content": round(max(0, min(100, ndvi * 100)), 1),
                "data_source": "Sentinel-2 (Real)",
                "satellites": ["Sentinel-2A", "Sentinel-2B"],
                "quality": "good",
                "is_real_data": True,
                "fallback_reason": None,
            }

        except ImportError:
            logger.warning("satellite_data module not available")
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch real NDVI data: {e}")
            return None

    def _fetch_simulated_ndvi(self, region_id: str, date: str, fallback_reason: str = "simulated_fallback") -> Optional[Dict]:
        """
        Fetch simulated NDVI data (fallback).

        Args:
            region_id: Region identifier
            date: Date string (YYYY-MM-DD)

        Returns:
            Dictionary with NDVI metrics
        """
        region = self.regions[region_id]

        # Set random seed for reproducibility
        np.random.seed(hash(date + region_id) % 2**32)

        # Get baseline NDVI
        baseline = region["baseline_ndvi"]
        
        # Add seasonal variation
        month = datetime.strptime(date, "%Y-%m-%d").month
        day_of_year = datetime.strptime(date, "%Y-%m-%d").timetuple().tm_yday
        
        # Determine if in critical growing season
        is_critical_season = month in region["critical_months"]
        
        # Seasonal factor (higher NDVI in growing season)
        if region["type"] == "forest":
            # Forests have less seasonal variation
            seasonal_factor = 0.05 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        elif region["type"] == "plantation":
            # Plantations have moderate seasonal variation
            seasonal_factor = 0.10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        else:
            # Row crops have strong seasonal variation
            seasonal_factor = 0.20 * np.sin(2 * np.pi * (day_of_year - 120) / 365)
        
        # Add stress factors (drought, disease, etc.)
        stress_factor = np.random.uniform(-0.15, 0.10)
        
        # Random daily variation
        daily_noise = np.random.normal(0, 0.03)
        
        # Calculate actual NDVI
        ndvi = baseline * (1 + seasonal_factor + stress_factor) + daily_noise
        ndvi = max(0.0, min(1.0, ndvi))
        
        # Calculate EVI (Enhanced Vegetation Index) - typically lower than NDVI
        evi = ndvi * 0.85 + np.random.normal(0, 0.02)
        evi = max(0.0, min(1.0, evi))
        
        # Calculate anomaly
        ndvi_anomaly = ndvi - baseline
        ndvi_anomaly_pct = (ndvi - baseline) / baseline * 100
        
        # Determine vegetation health status
        if ndvi_anomaly_pct < -20:
            status = "severe_stress"
        elif ndvi_anomaly_pct < -10:
            status = "stress"
        elif ndvi_anomaly_pct < -5:
            status = "slight_stress"
        elif ndvi_anomaly_pct > 10:
            status = "excellent"
        elif ndvi_anomaly_pct > 5:
            status = "good"
        else:
            status = "normal"
        
        # Calculate impact score (0-100)
        # During critical season, deviations have more impact
        if is_critical_season:
            impact_multiplier = 1.5
        else:
            impact_multiplier = 0.7
        
        impact_score = min(100, abs(ndvi_anomaly_pct) * impact_multiplier)
        
        # Calculate leaf area index (LAI) estimate
        lai = max(0, 6.0 * ndvi + np.random.normal(0, 0.5))
        
        # Calculate chlorophyll content (relative)
        chlorophyll = ndvi * 100 + np.random.normal(0, 5)
        chlorophyll = max(0, min(100, chlorophyll))
        
        return {
            "region_id": region_id,
            "region_name": region["name"],
            "region_type": region["type"],
            "country": region["country"],
            "date": date,
            "month": month,
            "ndvi": round(ndvi, 3),
            "evi": round(evi, 3),
            "baseline_ndvi": baseline,
            "ndvi_anomaly": round(ndvi_anomaly, 3),
            "ndvi_anomaly_pct": round(ndvi_anomaly_pct, 2),
            "status": status,
            "is_critical_season": is_critical_season,
            "impact_score": round(impact_score, 1),
            "lai_estimate": round(lai, 2),  # Leaf Area Index
            "chlorophyll_content": round(chlorophyll, 1),  # Relative
            "data_source": "MODIS_Sentinel2",
            "satellites": ["Terra", "Aqua", "Sentinel-2"],
            "quality": "good" if np.random.random() > 0.1 else "cloudy",
            "is_real_data": False,
            "fallback_reason": fallback_reason,
        }
    
    def calculate_baseline(self, region_id: str, days: int = 90) -> Dict:
        """
        Calculate baseline NDVI for a region.
        
        Args:
            region_id: Region identifier
            days: Number of days for baseline calculation
            
        Returns:
            Dictionary with baseline metrics
        """
        logger.info(f"Calculating {days}-day baseline for {region_id}")
        
        # Fetch historical data
        end_date = datetime.now()
        historical_ndvi = []
        historical_evi = []
        historical_anomaly = []
        
        for i in range(days):
            date = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
            data = self.fetch_ndvi_data(region_id, date)
            if data and data["quality"] == "good":
                historical_ndvi.append(data["ndvi"])
                historical_evi.append(data["evi"])
                historical_anomaly.append(data["ndvi_anomaly_pct"])
        
        if not historical_ndvi:
            return {"error": "No valid historical data"}
        
        # Calculate baseline statistics
        baseline = {
            "region_id": region_id,
            "period_days": len(historical_ndvi),
            "ndvi": {
                "mean": round(np.mean(historical_ndvi), 3),
                "std": round(np.std(historical_ndvi), 3),
                "median": round(np.median(historical_ndvi), 3),
            },
            "evi": {
                "mean": round(np.mean(historical_evi), 3),
                "std": round(np.std(historical_evi), 3),
                "median": round(np.median(historical_evi), 3),
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
        Detect anomalies in vegetation health.
        
        Args:
            current_data: Current NDVI data
            baseline: Baseline statistics
            threshold_std: Number of standard deviations for anomaly
            
        Returns:
            Dictionary with anomaly detection results
        """
        # Calculate z-scores
        ndvi_z = (current_data["ndvi"] - baseline["ndvi"]["mean"]) / \
                 baseline["ndvi"]["std"] if baseline["ndvi"]["std"] > 0 else 0
        
        evi_z = (current_data["evi"] - baseline["evi"]["mean"]) / \
                baseline["evi"]["std"] if baseline["evi"]["std"] > 0 else 0
        
        # Determine anomaly status
        ndvi_anomaly = "significant" if abs(ndvi_z) > threshold_std else \
                      "moderate" if abs(ndvi_z) > 1.5 else "none"
        
        # Combined score
        combined_z = (abs(ndvi_z) + abs(evi_z)) / 2
        
        return {
            "ndvi_z_score": round(ndvi_z, 2),
            "ndvi_anomaly": ndvi_anomaly,
            "ndvi_deviation_pct": round((current_data["ndvi"] - 
                                        baseline["ndvi"]["mean"]) / 
                                        baseline["ndvi"]["mean"] * 100, 2),
            "evi_z_score": round(evi_z, 2),
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
        current_data = self.fetch_ndvi_data(region_id, date)
        if not current_data:
            return {"error": "Failed to fetch current data"}
        
        # Calculate baseline
        baseline = self.calculate_baseline(region_id, baseline_days)
        if "error" in baseline:
            return {"error": baseline["error"]}
        
        # Detect anomaly
        anomaly = self.detect_anomaly(current_data, baseline)
        
        # Generate signal based on vegetation status
        status = current_data["status"]
        is_critical = current_data["is_critical_season"]
        region_type = region["type"]
        
        if region_type in ["row_crops", "plantation"]:
            # Crop signals
            # Stress = supply shortage = bullish prices = LONG
            if status in ["severe_stress", "stress"]:
                direction = "long"
                confidence = min(100, 60 + current_data["impact_score"] * 0.5)
                if is_critical:
                    confidence = min(100, confidence * 1.2)
                rationale = f"Vegetation stress in {region['name']}. NDVI {current_data['ndvi_anomaly_pct']:.1f}% below normal. Supply at risk, bullish for prices."
            
            elif status == "slight_stress":
                if is_critical:
                    direction = "long"
                    confidence = 60
                    rationale = f"Slight vegetation stress during critical period. NDVI {current_data['ndvi_anomaly_pct']:.1f}% below normal. Mild bullish signal."
                else:
                    direction = "neutral"
                    confidence = 50
                    rationale = f"Slight vegetation stress outside critical period. Limited supply impact."
            
            # Excellent = good supply = bearish prices = SHORT
            elif status == "excellent":
                direction = "short"
                confidence = min(100, 60 + abs(current_data["ndvi_anomaly_pct"]) * 0.8)
                rationale = f"Excellent vegetation health in {region['name']}. NDVI +{abs(current_data['ndvi_anomaly_pct']):.1f}% above normal. Strong supply, bearish for prices."
            
            elif status == "good":
                if is_critical:
                    direction = "short"
                    confidence = 58
                    rationale = f"Good vegetation conditions during critical period. Favorable for yields, mildly bearish."
                else:
                    direction = "neutral"
                    confidence = 52
                    rationale = f"Good vegetation conditions. Normal supply expectations."
            
            else:  # normal
                direction = "neutral"
                confidence = 50
                rationale = f"Normal vegetation health in {region['name']}. {current_data['ndvi_anomaly_pct']:+.1f}% from baseline. Expected supply."
        
        elif region_type == "forest":
            # Forestry signals (longer-term, less sensitive)
            # Stress = supply shortage = bullish = LONG
            if status in ["severe_stress", "stress"]:
                direction = "long"
                confidence = min(100, 55 + current_data["impact_score"] * 0.3)
                rationale = f"Forest stress detected in {region['name']}. NDVI {current_data['ndvi_anomaly_pct']:.1f}% below normal. Timber/pulp supply concern, bullish for prices."
            
            # Excellent = good supply = bearish = SHORT
            elif status == "excellent":
                direction = "short"
                confidence = 58
                rationale = f"Excellent forest health. NDVI +{abs(current_data['ndvi_anomaly_pct']):.1f}% above normal. Strong supply, bearish for prices."
            
            else:
                direction = "neutral"
                confidence = 50
                rationale = f"Normal forest conditions. Stable timber/pulp supply."
        
        else:
            # Default logic
            if anomaly["combined_z_score"] > 2.0:
                direction = "long"
                confidence = min(100, 60 + anomaly["combined_z_score"] * 10)
                rationale = f"Vegetation health significantly above baseline."
            elif anomaly["combined_z_score"] < -2.0:
                direction = "short"
                confidence = min(100, 60 + abs(anomaly["combined_z_score"]) * 10)
                rationale = f"Vegetation health significantly below baseline."
            else:
                direction = "neutral"
                confidence = 50
                rationale = f"Vegetation health within normal range."
        
        confidence = round(confidence, 1)
        is_real_data = bool(current_data.get("is_real_data", False))
        confidence_penalty = 0
        if not is_real_data:
            confidence = round(max(35.0, confidence * 0.7), 1)
            confidence_penalty = 30

        signal = {
            "region_id": region_id,
            "region_name": region["name"],
            "region_type": region["type"],
            "country": region["country"],
            "date": date,
            "signal_type": "vegetation_health",
            "direction": direction,
            "confidence": confidence,
            "confidence_label": _confidence_label(confidence),
            "rationale": rationale,
            "instruments": region["instruments"],
            "current_ndvi": current_data["ndvi"],
            "current_evi": current_data["evi"],
            "ndvi_anomaly_pct": current_data["ndvi_anomaly_pct"],
            "status": status,
            "is_critical_season": is_critical,
            "baseline_ndvi": baseline["ndvi"]["mean"],
            "ndvi_z_score": anomaly["ndvi_z_score"],
            "impact_score": current_data["impact_score"],
            "lai_estimate": current_data["lai_estimate"],
            "chlorophyll_content": current_data["chlorophyll_content"],
            "data_quality": current_data["quality"],
            "data_source": current_data.get("data_source", "unknown"),
            "satellites": current_data.get("satellites", []),
            "is_real_data": is_real_data,
            "fallback_reason": current_data.get("fallback_reason"),
            "confidence_penalty_pct": confidence_penalty,
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
            "stress_regions": sum(1 for s in signals if s["status"] in ["stress", "severe_stress"]),
            "excellent_regions": sum(1 for s in signals if s["status"] == "excellent"),
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
            "monitoring_type": "vegetation_health",
            "satellites": ["Terra (MODIS)", "Aqua (MODIS)", "Sentinel-2"],
            "metrics": ["NDVI", "EVI", "LAI", "Chlorophyll"],
            "update_frequency": "Daily",
            "latency": "1-3 days",
            "total_regions": len(self.regions),
            "region_types": list(set(r["type"] for r in self.regions.values())),
            "regions": self.regions,
            "signal_logic": {
                "stress": "LONG crops (supply shortage = bullish prices)",
                "excellent": "SHORT crops (good supply = bearish prices)",
                "normal": "NEUTRAL (expected supply)",
                "critical_season": "Higher impact during growing season",
                "real_data_penalty": "Simulated fallback reduces confidence and blocks strong actionability"
            },
            "trading_instruments": list(set(
                inst for region in self.regions.values() 
                for inst in region["instruments"]
            ))
        }


def main():
    """Test vegetation health monitoring."""
    logging.basicConfig(level=logging.INFO)
    
    monitor = VegetationHealthMonitor()
    
    # Get regional summary
    print("\n🌿 Vegetation Health Monitor - Regional Summary")
    print("=" * 60)
    summary = monitor.get_regional_summary()
    print(f"Monitoring {summary['total_regions']} agricultural/forestry regions")
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
        print(f"  NDVI: {signal['current_ndvi']:.3f} (baseline: {signal['baseline_ndvi']:.3f})")
        print(f"  Status: {signal['status'].upper()}")
        print(f"  Anomaly: {signal['ndvi_anomaly_pct']:+.1f}%")
        print(f"  Critical Season: {'YES' if signal['is_critical_season'] else 'NO'}")
        print(f"  LAI: {signal['lai_estimate']:.2f}")
        print(f"  Instruments: {', '.join(signal['instruments'])}")
        print(f"  Rationale: {signal['rationale']}")


if __name__ == "__main__":
    main()
