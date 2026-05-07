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

from pipeline.regions import get_regions_by_type

logger = logging.getLogger(__name__)


INSTRUMENTS = {
    "beef_bullish": ["LE=F", "CORN", "SOYB"],
    "beef_bearish": ["LE=F", "CORN", "SOYB"],
    "feed_proxy": ["CORN", "SOYB"],
}

THERMAL_HIGH_THRESHOLD = 50
THERMAL_MODERATE_THRESHOLD = 25
NDVI_POOR_THRESHOLD = 0.3
NDVI_FAIR_THRESHOLD = 0.45
NDVI_GOOD_THRESHOLD = 0.65
REGION_ANOMALY_DIRECTION_THRESHOLD = 30
CONFIDENCE_MULTIPLIER = 25
CONFIDENCE_BASE = 10
MAX_CONFIDENCE = 100
MIN_CONFIDENCE = 10


def _load_feedlot_regions() -> Dict:
    regions = get_regions_by_type("cattle_feedlot")
    out = {}
    for region_id, cfg in regions.items():
        out[region_id] = {
            "name": cfg.get("name", region_id),
            "description": cfg.get("description", ""),
            "lat": cfg.get("center", [0, 0])[1],
            "lon": cfg.get("center", [0, 0])[0],
            "radius_km": cfg.get("radius_km", 50),
            "capacity_share": cfg.get("capacity_share", 0),
            "state": cfg.get("state", ""),
            "thermal_id": cfg.get("thermal_id"),
            "veg_id": cfg.get("veg_id"),
        }
    return out


def _load_pasture_regions() -> Dict:
    regions = get_regions_by_type("pasture")
    out = {}
    for region_id, cfg in regions.items():
        out[region_id] = {
            "name": cfg.get("name", region_id),
            "lat": cfg.get("center", [0, 0])[1],
            "lon": cfg.get("center", [0, 0])[0],
            "radius_km": cfg.get("radius_km", 50),
            "veg_id": cfg.get("veg_id"),
        }
    return out


class CattleFeedlotMonitor:
    """Monitor US cattle feedlots via satellite thermal and vegetation data."""

    def __init__(self, output_base: str = "outputs"):
        self.output_base = Path(output_base)
        self.regions = _load_feedlot_regions()
        self.pasture_regions = _load_pasture_regions()
        self.logger = logging.getLogger(f"{__name__}.CattleFeedlotMonitor")

    def analyze_feedlot_thermal(self, region_id: str) -> Dict:
        """
        Analyze thermal infrared data over a feedlot region.
        Higher thermal anomaly = more cattle activity / density.
        """
        region = self.regions.get(region_id)
        if not region:
            return {}

        thermal_id = region.get("thermal_id")
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
        results = {}
        for region_id, region in self.pasture_regions.items():
            veg_id = region.get("veg_id")
            if veg_id:
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
        if avg_thermal > THERMAL_HIGH_THRESHOLD:
            supply_score -= 2
        elif avg_thermal > THERMAL_MODERATE_THRESHOLD:
            supply_score -= 1

        if avg_pasture_ndvi < NDVI_POOR_THRESHOLD:
            supply_score -= 2
        elif avg_pasture_ndvi < NDVI_FAIR_THRESHOLD:
            supply_score -= 1
        elif avg_pasture_ndvi > NDVI_GOOD_THRESHOLD:
            supply_score += 1

        confidence = min(MAX_CONFIDENCE, max(MIN_CONFIDENCE, int(abs(supply_score) * CONFIDENCE_MULTIPLIER + CONFIDENCE_BASE)))

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
            confidence = max(MIN_CONFIDENCE, confidence // 2)

        # Per-region signals
        for region_id, result in thermal_results.items():
            region = self.regions[region_id]
            anomaly = result.get("thermal_anomaly", 0)
            source = result.get("source", "unknown")

            region_confidence = min(MAX_CONFIDENCE, max(MIN_CONFIDENCE, int(abs(anomaly) * 1.5 + CONFIDENCE_BASE)))
            region_direction = "long" if anomaly > REGION_ANOMALY_DIRECTION_THRESHOLD else "short" if anomaly < -REGION_ANOMALY_DIRECTION_THRESHOLD else "neutral"

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
