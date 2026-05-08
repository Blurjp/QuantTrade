"""Combined agriculture signal generation for validated crop trading hypotheses."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from pipeline.precipitation import PrecipitationMonitor
from pipeline.vegetation_health import VegetationHealthMonitor


def _confidence_label(score: float) -> str:
    # Convert to 0-1 scale if needed (assuming input might be 0-100)
    if score > 1.0:
        score = score / 100.0
    
    if pd.isna(score):
        return "Low"  # Default to Low for consistency
    if score >= 0.75:
        return "High"
    if score >= 0.55:
        return "Medium"
    return "Low"


def _direction_to_vote(direction: str) -> int:
    normalized = (direction or "").upper()
    if normalized == "LONG":
        return 1
    if normalized == "SHORT":
        return -1
    return 0


AGRICULTURE_SETUPS = {
    "agriculture_us_corn_soy": {
        "label": "US Corn and Soybeans Combined",
        "vegetation_region": "usa_corn_soybeans",
        "precip_region": "usa_corn_belt",
        "instruments": ["CORN", "SOYB"],
        "confirmations_required": 2,
    },
    "agriculture_us_wheat": {
        "label": "US Wheat Combined",
        "vegetation_region": "usa_wheat_plains",
        "precip_region": "usa_winter_wheat",
        "instruments": ["WEAT", "XOP"],
        "confirmations_required": 2,
    },
}


def _safe_signal(monitor, region_id: str, target_date: str) -> Dict:
    signal = monitor.generate_signal(region_id, target_date)
    if not signal or "error" in signal:
        return {"error": signal.get("error", "unknown_error") if signal else "missing_signal"}
    return signal


def _component_score(signal: Dict, source: str) -> tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    if source == "vegetation":
        anomaly = float(signal.get("ndvi_anomaly_pct", 0.0) or 0.0)
        status = signal.get("status")
        if status in {"severe_stress", "stress"} or anomaly <= -10:
            score += 2
            reasons.append(f"NDVI stress {anomaly:.1f}%")
        elif status == "slight_stress" or anomaly <= -5:
            score += 1
            reasons.append(f"NDVI mild stress {anomaly:.1f}%")
        elif status == "excellent" or anomaly >= 10:
            score -= 2
            reasons.append(f"NDVI excellent {anomaly:.1f}%")
        elif status == "good" or anomaly >= 5:
            score -= 1
            reasons.append(f"NDVI favorable {anomaly:.1f}%")
    else:
        anomaly = float(signal.get("precip_anomaly_pct", 0.0) or 0.0)
        status = signal.get("status")
        if status in {"severe_drought", "drought", "flood", "wet"}:
            score += 2
            reasons.append(f"Precip disruption {anomaly:.1f}%")
        elif status == "dry":
            score += 1
            reasons.append(f"Precip dry {anomaly:.1f}%")
        elif status == "slightly_wet":
            score -= 1
            reasons.append(f"Precip favorable {anomaly:.1f}%")
        elif status == "normal":
            score -= 1
            reasons.append(f"Precip normal {anomaly:.1f}%")
    return score, reasons


def build_agriculture_signals(target_date: str, output_base: str = "outputs") -> Dict[str, Dict]:
    vegetation = VegetationHealthMonitor(output_base=output_base)
    precipitation = PrecipitationMonitor(output_base=output_base)
    combined: Dict[str, Dict] = {}

    for signal_id, setup in AGRICULTURE_SETUPS.items():
        veg_signal = _safe_signal(vegetation, setup["vegetation_region"], target_date)
        precip_signal = _safe_signal(precipitation, setup["precip_region"], target_date)

        if "error" in veg_signal or "error" in precip_signal:
            combined[signal_id] = {
                "signal": f"{setup['label']} unavailable",
                "confidence": "Low",
                "actionability": "Ignore",
                "trading_action": "FLAT",
                "type": "agriculture_combined",
                "instruments": setup["instruments"],
                "error": {
                    "vegetation": veg_signal.get("error"),
                    "precipitation": precip_signal.get("error"),
                },
                "portfolio_trade": False,
            }
            continue

        is_critical = bool(veg_signal.get("is_critical_season") and precip_signal.get("is_critical_season"))
        veg_score, veg_reasons = _component_score(veg_signal, "vegetation")
        precip_score, precip_reasons = _component_score(precip_signal, "precipitation")
        veg_vote = _direction_to_vote(veg_signal.get("direction"))
        precip_vote = _direction_to_vote(precip_signal.get("direction"))
        consensus_direction = 0
        if veg_vote != 0 and veg_vote == precip_vote:
            consensus_direction = veg_vote

        score = veg_score + precip_score
        real_data_ratio = sum(1 for s in (veg_signal, precip_signal) if s.get("is_real_data")) / 2.0
        avg_confidence = (float(veg_signal.get("confidence", 50.0)) + float(precip_signal.get("confidence", 50.0))) / 2.0

        if not is_critical or consensus_direction == 0 or abs(score) < 3:
            score = 0

        if score >= 3 and consensus_direction > 0:
            trading_action = "LONG"
            signal_text = f"{setup['label']} confirmed supply stress"
        elif score <= -3 and consensus_direction < 0:
            trading_action = "SHORT"
            signal_text = f"{setup['label']} confirmed strong supply"
        else:
            trading_action = "FLAT"
            signal_text = f"{setup['label']} no component consensus"

        if real_data_ratio == 0:
            avg_confidence = min(avg_confidence, 45.0)
        elif real_data_ratio < 1.0:
            avg_confidence = min(avg_confidence, 65.0)

        confidence = _confidence_label(avg_confidence)
        actionability = "Actionable" if trading_action != "FLAT" and confidence in {"High", "Medium"} and real_data_ratio >= 0.5 else "Ignore"

        combined[signal_id] = {
            "signal": signal_text,
            "confidence": confidence,
            "actionability": actionability,
            "trading_action": trading_action,
            "type": "agriculture_combined",
            "portfolio_trade": actionability == "Actionable",
            "instruments": setup["instruments"],
            "region_name": setup["label"],
            "meta_group": "agriculture_real_alpha",
            "combined_score": score,
            "consensus_direction": consensus_direction,
            "critical_season": is_critical,
            "real_data_ratio": real_data_ratio,
            "data_quality_mode": "real" if real_data_ratio == 1.0 else "mixed" if real_data_ratio > 0 else "simulated",
            "numeric_confidence": round(avg_confidence, 1),
            "components": {
                "vegetation": veg_signal,
                "precipitation": precip_signal,
            },
            "bias": "Bullish crop prices" if trading_action == "LONG" else "Bearish crop prices" if trading_action == "SHORT" else "Mixed crop outlook",
            "rationale": "; ".join(veg_reasons + precip_reasons) or "No strong agriculture signal",
            "confirmations_required": setup["confirmations_required"],
            "timestamp": datetime.now().isoformat(),
        }

    output_dir = Path(output_base) / target_date
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "agriculture_signals.json"
    output_file.write_text(json.dumps(combined, indent=2, default=str))
    return combined
