"""
Cattle Feedlot Satellite Monitor

Uses thermal infrared and vegetation data to monitor US cattle feedlot activity
and predict beef supply / price movements.

Strategy:
1. Thermal anomalies over feedlot regions indicate herd density changes
2. NDVI on surrounding pasture indicates feed supply (corn/grass)
3. Combined signal predicts beef supply tightness → price direction

Data Sources:
- Sentinel-2 (NDVI for pasture health)
- Sentinel-3 SLSTR (thermal infrared for feedlot activity)
- Landsat 8/9 TIRS (thermal backup)

Tradeable Instruments:
- COW: iPath Bloomberg Livestock Subindex ETN
- LE=F: CME Live Cattle Futures
- HE=F: CME Lean Hog Futures
- CORN: Teucrium Corn Fund (feed cost proxy)

Key Feedlot Regions:
- Texas Panhandle (Cactus, Hereford, Dalhart) — largest concentration
- Southwest Kansas (Dodge City, Garden City, Liberal)
- Central Nebraska (Lexington, Grand Island, North Platte)
- Northeast Colorado (Greeley, Sterling, Fort Morgan)
- Central Iowa (Des Moines area)
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Major US feedlot regions with coordinates (lat, lon, radius_km)
FEEDLOT_REGIONS = {
    "texas_panhandle": {
        "name": "Texas Panhandle Feedlots",
        "description": "Largest US feedlot concentration (Cactus, Hereford, Dalhart)",
        "lat": 35.83,
        "lon": -101.95,
        "radius_km": 80,
        "capacity_share": 0.30,  # ~30% of US feedlot capacity
        "state": "TX",
    },
    "sw_kansas": {
        "name": "Southwest Kansas Feedlots",
        "description": "Major feedlot region (Dodge City, Garden City, Liberal)",
        "lat": 37.75,
        "lon": -100.44,
        "radius_km": 70,
        "capacity_share": 0.20,
        "state": "KS",
    },
    "central_nebraska": {
        "name": "Central Nebraska Feedlots",
        "description": "Key cattle region (Lexington, Grand Island)",
        "lat": 40.78,
        "lon": -99.23,
        "radius_km": 60,
        "capacity_share": 0.15,
        "state": "NE",
    },
    "ne_colorado": {
        "name": "Northeast Colorado Feedlots",
        "description": "Greeley-Sterling feedlot corridor",
        "lat": 40.42,
        "lon": -104.31,
        "radius_km": 50,
        "capacity_share": 0.10,
        "state": "CO",
    },
    "central_iowa": {
        "name": "Central Iowa Feedlots",
        "description": "Iowa cattle finishing operations",
        "lat": 41.59,
        "lon": -93.62,
        "radius_km": 50,
        "capacity_share": 0.08,
        "state": "IA",
    },
    "oklahoma_panhandle": {
        "name": "Oklahoma Panhandle Feedlots",
        "description": "Guymon-area feedlots",
        "lat": 36.68,
        "lon": -101.48,
        "radius_km": 40,
        "capacity_share": 0.07,
        "state": "OK",
    },
    "idaho_snake_river": {
        "name": "Idaho Snake River Basin",
        "description": "Pacific Northwest feedlot operations",
        "lat": 43.46,
        "lon": -112.05,
        "radius_km": 50,
        "capacity_share": 0.05,
        "state": "ID",
    },
    "central_california": {
        "name": "Central Valley California",
        "description": "California feedlot and dairy operations",
        "lat": 36.24,
        "lon": -119.81,
        "radius_km": 60,
        "capacity_share": 0.05,
        "state": "CA",
    },
}

# Pasture/grazing regions (for feed supply)
PASTURE_REGIONS = {
    "flint_hills": {
        "name": "Flint Hills Kansas Grazing",
        "lat": 38.35,
        "lon": -96.55,
        "radius_km": 80,
    },
    "sandhills_ne": {
        "name": "Nebraska Sandhills",
        "lat": 42.05,
        "lon": -101.75,
        "radius_km": 100,
    },
    "tx_hill_country": {
        "name": "Texas Hill Country Pasture",
        "lat": 30.30,
        "lon": -98.87,
        "radius_km": 60,
    },
}

# Instruments for trading signals
INSTRUMENTS = {
    "beef_bullish": ["COW", "LE=F"],   # Long cattle on supply tightness
    "beef_bearish": ["COW", "LE=F"],    # Short cattle on oversupply
    "feed_proxy": ["CORN", "SOYB"],      # Corn/soy as feed cost
}


class CattleFeedlotMonitor:
    """Monitor US cattle feedlots via satellite thermal and vegetation data."""

    def __init__(self, output_base: str = "outputs"):
        self.output_base = Path(output_base)
        self.regions = FEEDLOT_REGIONS
        self.pasture_regions = PASTURE_REGIONS
        self.logger = logging.getLogger(f"{__name__}.CattleFeedlotMonitor")

    def analyze_feedlot_thermal(self, region_id: str) -> Dict:
        """
        Analyze thermal infrared data over a feedlot region.
        Higher thermal anomaly = more cattle activity / density.
        """
        region = self.regions.get(region_id)
        if not region:
            return {}

        # Map cattle region_id to thermal facility IDs
        thermal_map = {
            "texas_panhandle": "feedlot_texas_panhandle",
            "sw_kansas": "feedlot_sw_kansas",
            "central_nebraska": "feedlot_central_nebraska",
        }

        thermal_id = thermal_map.get(region_id)
        if thermal_id:
            thermal_file = self.output_base / "thermal_infrared" / f"signal_{thermal_id}_*.json"
            import glob
            files = sorted(glob.glob(str(thermal_file)))
            if files:
                with open(files[-1]) as f:
                    data = json.load(f)
                    return {
                        "region_id": region_id,
                        "thermal_anomaly": data.get("anomaly_pct", data.get("temperature_anomaly", 0)),
                        "date": data.get("date", datetime.now().isoformat()),
                        "source": "thermal_infrared",
                    }

        return {
            "region_id": region_id,
            "thermal_anomaly": 0,
            "date": datetime.now().isoformat(),
            "source": "no_data",
        }

    def analyze_pasture_health(self) -> Dict:
        """
        Analyze pasture/grazing land health via NDVI from vegetation pipeline.
        """
        # Map pasture region IDs to vegetation pipeline region IDs
        pasture_to_veg = {
            "flint_hills": "usa_flint_hills_pasture",
            "sandhills_ne": "usa_sandhills_pasture",
            "tx_hill_country": "usa_texas_panhandle_feedlot",
        }

        results = {}
        for region_id, region in self.pasture_regions.items():
            veg_id = pasture_to_veg.get(region_id)
            if veg_id:
                # Try loading latest vegetation signal
                import glob
                veg_files = sorted(glob.glob(str(self.output_base / "vegetation" / f"*{veg_id}*")))
                if veg_files:
                    with open(veg_files[-1]) as f:
                        data = json.load(f)
                        results[region_id] = {
                            "ndvi": data.get("current_ndvi", 0),
                            "anomaly_pct": data.get("ndvi_anomaly_pct", 0),
                            "status": data.get("status", "unknown"),
                        }
                        continue

            results[region_id] = {"ndvi": 0, "anomaly_pct": 0, "status": "no_data"}

        return results

    def generate_signal(self) -> List[Dict]:
        """
        Generate trading signals based on feedlot and pasture analysis.

        Logic:
        - High thermal anomaly + poor pasture → supply tightness → LONG cattle
        - Low thermal anomaly + good pasture → oversupply → SHORT cattle
        """
        signals = []

        # Analyze each feedlot region
        thermal_results = {}
        for region_id in self.regions:
            thermal_results[region_id] = self.analyze_feedlot_thermal(region_id)

        # Analyze pasture health
        pasture_health = self.analyze_pasture_health()

        # Weight by capacity share
        weighted_thermal = 0
        total_weight = 0
        for region_id, result in thermal_results.items():
            weight = self.regions[region_id]["capacity_share"]
            anomaly = abs(result.get("thermal_anomaly", 0))
            weighted_thermal += anomaly * weight
            total_weight += weight

        avg_thermal = weighted_thermal / total_weight if total_weight > 0 else 0

        # Average pasture health
        pasture_ndvi_values = [v["ndvi"] for v in pasture_health.values() if v["ndvi"] > 0]
        avg_pasture_ndvi = np.mean(pasture_ndvi_values) if pasture_ndvi_values else 0.5

        # Determine signal
        # High thermal + low pasture NDVI = bullish (supply tight)
        # Low thermal + high pasture NDVI = bearish (abundant supply)

        supply_score = 0
        if avg_thermal > 50:
            supply_score -= 2  # High activity → cattle being moved/sold
        elif avg_thermal > 25:
            supply_score -= 1

        if avg_pasture_ndvi < 0.3:
            supply_score -= 2  # Poor pasture → supply pressure
        elif avg_pasture_ndvi < 0.45:
            supply_score -= 1
        elif avg_pasture_ndvi > 0.65:
            supply_score += 1  # Good pasture → healthy supply

        # Generate confidence
        confidence = min(100, max(10, int(abs(supply_score) * 25 + 10)))

        direction = "neutral"
        instruments = INSTRUMENTS["beef_bullish"]
        rationale = ""

        if supply_score <= -2:
            direction = "long"
            instruments = INSTRUMENTS["beef_bullish"]
            rationale = (
                f"Cattle supply tightness detected. Feedlot thermal anomaly: {avg_thermal:.1f}%, "
                f"Pasture NDVI: {avg_pasture_ndvi:.3f}. "
                f"High feedlot activity with poor grazing conditions suggest reduced future supply."
            )
        elif supply_score >= 1:
            direction = "short"
            instruments = INSTRUMENTS["beef_bearish"]
            rationale = (
                f"Cattle oversupply conditions. Feedlot thermal anomaly: {avg_thermal:.1f}%, "
                f"Pasture NDVI: {avg_pasture_ndvi:.3f}. "
                f"Good pasture conditions support healthy cattle supply pipeline."
            )
        else:
            direction = "neutral"
            instruments = INSTRUMENTS["feed_proxy"]
            rationale = (
                f"Cattle market neutral. Feedlot thermal: {avg_thermal:.1f}%, "
                f"Pasture NDVI: {avg_pasture_ndvi:.3f}. No strong directional signal."
            )
            confidence = max(10, confidence // 2)

        # Per-region signals
        for region_id, result in thermal_results.items():
            region = self.regions[region_id]
            anomaly = result.get("thermal_anomaly", 0)
            source = result.get("source", "unknown")

            region_confidence = min(100, max(10, int(abs(anomaly) * 1.5 + 10)))
            region_direction = "long" if anomaly > 30 else "short" if anomaly < -30 else "neutral"

            if region_direction == "neutral":
                region_confidence = max(10, region_confidence // 2)

            signal = {
                "region_id": f"usa_{region_id}_cattle",
                "region_name": f"{region['name']} (Cattle)",
                "region_type": "livestock",
                "country": "USA",
                "state": region["state"],
                "date": datetime.now().strftime("%Y-%m-%d"),
                "signal_type": "cattle_feedlot",
                "direction": region_direction,
                "confidence": region_confidence,
                "confidence_label": (
                    "High" if region_confidence >= 75
                    else "Medium" if region_confidence >= 50
                    else "Low"
                ),
                "rationale": (
                    f"Feedlot thermal anomaly in {region['name']}: {anomaly:.1f}% "
                    f"(capacity share: {region['capacity_share']*100:.0f}%). "
                    f"Source: {source}."
                ),
                "instruments": instruments,
                "thermal_anomaly": anomaly,
                "pasture_ndvi": avg_pasture_ndvi,
                "capacity_share": region["capacity_share"],
                "source": source,
            }
            signals.append(signal)

        # Aggregate US-wide signal
        signals.append({
            "region_id": "usa_cattle_aggregate",
            "region_name": "US Cattle Aggregate Signal",
            "region_type": "livestock",
            "country": "USA",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "signal_type": "cattle_feedlot",
            "direction": direction,
            "confidence": confidence,
            "confidence_label": (
                "High" if confidence >= 75
                else "Medium" if confidence >= 50
                else "Low"
            ),
            "rationale": rationale,
            "instruments": instruments,
            "weighted_thermal": round(avg_thermal, 1),
            "avg_pasture_ndvi": round(avg_pasture_ndvi, 3),
            "supply_score": supply_score,
            "monitored_regions": len(self.regions),
        })

        return signals


def run(output_base: str = "outputs") -> List[Dict]:
    """Entry point for pipeline integration."""
    monitor = CattleFeedlotMonitor(output_base=output_base)
    signals = monitor.generate_signal()

    # Save signals
    out_dir = Path(output_base) / "cattle_feedlot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"signals_{datetime.now().strftime('%Y%m%d')}.json"
    with open(out_file, "w") as f:
        json.dump(signals, f, indent=2, default=str)

    logger.info(f"Cattle feedlot: {len(signals)} signals generated")
    for s in signals:
        logger.info(
            f"  {s['region_id']}: {s['direction']} conf={s['confidence']}% "
            f"({s['confidence_label']}) [{', '.join(s['instruments'])}]"
        )

    return signals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    signals = run()
    print(f"\nGenerated {len(signals)} cattle feedlot signals")
    for s in signals:
        print(f"  {s['region_id']}: {s['direction']} {s['confidence']}%")
