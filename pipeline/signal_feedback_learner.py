"""
Signal Feedback Learner — closed-loop learning from trade outcomes.

Every time a position is closed, we record whether the signal was correct.
Over time, this adjusts:
1. Per-region confidence weights (amplify accurate regions, dampen inaccurate)
2. Dynamic confidence thresholds (lower for high-accuracy regions, higher for poor ones)
3. Per-signal-type accuracy scores

This is the feedback loop:  Signal → Trade → Outcome → Learn → Adjust Signal
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

FEEDBACK_FILE = "signal_feedback.json"

# Minimum evaluations before adjusting weights (avoid overfitting on small samples)
MIN_EVALUATIONS = 3

# Default confidence threshold
DEFAULT_THRESHOLD = 75.0

# Weight adjustment bounds
MIN_WEIGHT = 0.3
MAX_WEIGHT = 2.0

# Default weight (no adjustment)
BASE_WEIGHT = 1.0


def _load_feedback(output_base: str = "outputs") -> Dict:
    path = Path(output_base) / FEEDBACK_FILE
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {
        "trades": [],           # Closed trade records
        "region_accuracy": {},   # region_id -> {correct, total, weight, threshold}
        "signal_type_accuracy": {},  # signal_type -> {correct, total, weight}
        "last_updated": None,
    }


def _save_feedback(feedback: Dict, output_base: str = "outputs") -> None:
    path = Path(output_base) / FEEDBACK_FILE
    feedback["last_updated"] = datetime.now().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(feedback, indent=2, default=str))


def record_trade_outcome(
    region_id: str,
    signal_type: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    pnl: float,
    rationale: str = "",
    output_base: str = "outputs",
) -> Dict:
    """
    Record a closed trade outcome for learning.
    
    Call this after every position close (stop-loss, take-profit, or manual).
    """
    feedback = _load_feedback(output_base)
    
    # Determine if signal was correct
    if direction == "long":
        was_correct = exit_price > entry_price
    elif direction == "short":
        was_correct = exit_price < entry_price
    else:
        was_correct = None
    
    trade_record = {
        "region_id": region_id,
        "signal_type": signal_type,
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": pnl,
        "pnl_pct": ((exit_price - entry_price) / entry_price * 100) if entry_price else 0,
        "was_correct": was_correct,
        "rationale": rationale,
        "closed_at": datetime.now().isoformat(),
    }
    
    feedback["trades"].append(trade_record)
    
    # Update region accuracy
    if region_id not in feedback["region_accuracy"]:
        feedback["region_accuracy"][region_id] = {
            "correct": 0, "total": 0,
            "weight": BASE_WEIGHT, "threshold": DEFAULT_THRESHOLD,
        }
    
    region = feedback["region_accuracy"][region_id]
    region["total"] += 1
    if was_correct:
        region["correct"] += 1
    
    # Update signal type accuracy
    if signal_type not in feedback["signal_type_accuracy"]:
        feedback["signal_type_accuracy"][signal_type] = {
            "correct": 0, "total": 0, "weight": BASE_WEIGHT,
        }
    
    st = feedback["signal_type_accuracy"][signal_type]
    st["total"] += 1
    if was_correct:
        st["correct"] += 1
    
    # Recalculate weights and thresholds
    _recalculate_weights(feedback)
    
    _save_feedback(feedback, output_base)
    
    win_rate = region["correct"] / region["total"] * 100
    logger.info(
        f"Trade outcome recorded: {direction} {region_id} "
        f"{'✅' if was_correct else '❌'} P&L=${pnl:+.2f} "
        f"Region win rate: {win_rate:.0f}% ({region['correct']}/{region['total']}) "
        f"Weight: {region['weight']:.2f} Threshold: {region['threshold']:.0f}%"
    )
    
    return trade_record


def _recalculate_weights(feedback: Dict) -> None:
    """
    Adjust weights and thresholds based on accumulated accuracy data.
    
    Weight formula:
    - win_rate > 65% → weight = min(MAX_WEIGHT, 1.0 + (win_rate - 50) / 50)
    - win_rate 40-65% → weight = 1.0 (neutral)
    - win_rate < 40% → weight = max(MIN_WEIGHT, win_rate / 50)
    
    Threshold formula:
    - win_rate > 70% → threshold = max(60, 75 - (win_rate - 70) * 0.5)
    - win_rate 40-70% → threshold = 75 (default)
    - win_rate < 40% → threshold = min(90, 75 + (40 - win_rate) * 0.5)
    """
    for region_id, stats in feedback["region_accuracy"].items():
        if stats["total"] < MIN_EVALUATIONS:
            continue  # Not enough data to adjust
        
        win_rate = stats["correct"] / stats["total"] * 100
        
        # Weight adjustment
        if win_rate > 65:
            stats["weight"] = round(min(MAX_WEIGHT, 1.0 + (win_rate - 50) / 50), 2)
        elif win_rate < 40:
            stats["weight"] = round(max(MIN_WEIGHT, win_rate / 50), 2)
        else:
            stats["weight"] = BASE_WEIGHT
        
        # Threshold adjustment
        if win_rate > 70:
            stats["threshold"] = round(max(60, DEFAULT_THRESHOLD - (win_rate - 70) * 0.5), 1)
        elif win_rate < 40:
            stats["threshold"] = round(min(90, DEFAULT_THRESHOLD + (40 - win_rate) * 0.5), 1)
        else:
            stats["threshold"] = DEFAULT_THRESHOLD
    
    # Same for signal types
    for sig_type, stats in feedback["signal_type_accuracy"].items():
        if stats["total"] < MIN_EVALUATIONS:
            continue
        win_rate = stats["correct"] / stats["total"] * 100
        if win_rate > 65:
            stats["weight"] = round(min(MAX_WEIGHT, 1.0 + (win_rate - 50) / 50), 2)
        elif win_rate < 40:
            stats["weight"] = round(max(MIN_WEIGHT, win_rate / 50), 2)
        else:
            stats["weight"] = BASE_WEIGHT


def get_region_weight(region_id: str, output_base: str = "outputs") -> float:
    """Get the learned weight for a region (default 1.0)."""
    feedback = _load_feedback(output_base)
    region = feedback.get("region_accuracy", {}).get(region_id)
    if region and region["total"] >= MIN_EVALUATIONS:
        return region["weight"]
    return BASE_WEIGHT


def get_signal_type_weight(signal_type: str, output_base: str = "outputs") -> float:
    """Get the learned weight for a signal type (default 1.0)."""
    feedback = _load_feedback(output_base)
    st = feedback.get("signal_type_accuracy", {}).get(signal_type)
    if st and st["total"] >= MIN_EVALUATIONS:
        return st["weight"]
    return BASE_WEIGHT


def get_effective_confidence(
    confidence: float,
    region_id: str,
    signal_type: str,
    output_base: str = "outputs",
) -> float:
    """
    Calculate effective confidence after applying learned weights.
    
    effective_confidence = raw_confidence * region_weight * signal_type_weight
    """
    region_w = get_region_weight(region_id, output_base)
    type_w = get_signal_type_weight(signal_type, output_base)
    return round(confidence * region_w * type_w, 1)


def get_dynamic_threshold(region_id: str, output_base: str = "outputs") -> float:
    """Get the dynamic confidence threshold for a region."""
    feedback = _load_feedback(output_base)
    region = feedback.get("region_accuracy", {}).get(region_id)
    if region and region["total"] >= MIN_EVALUATIONS:
        return region["threshold"]
    return DEFAULT_THRESHOLD


def should_trade(
    confidence: float,
    region_id: str,
    signal_type: str,
    output_base: str = "outputs",
) -> Tuple[bool, float, str]:
    """
    Determine if a signal should trigger a trade based on learned thresholds.
    
    Returns: (should_trade, effective_confidence, reason)
    """
    effective = get_effective_confidence(confidence, region_id, signal_type, output_base)
    threshold = get_dynamic_threshold(region_id, output_base)
    
    if effective >= threshold:
        return True, effective, f"Effective conf {effective:.1f}% >= threshold {threshold:.0f}%"
    else:
        return False, effective, f"Effective conf {effective:.1f}% < threshold {threshold:.0f}%"


def get_summary(output_base: str = "outputs") -> Dict:
    """Get a human-readable summary of learning state."""
    feedback = _load_feedback(output_base)
    
    total_trades = len(feedback.get("trades", []))
    total_correct = sum(1 for t in feedback["trades"] if t.get("was_correct"))
    total_pnl = sum(t.get("pnl", 0) for t in feedback["trades"])
    
    region_summary = {}
    for rid, stats in feedback.get("region_accuracy", {}).items():
        if stats["total"] == 0:
            continue
        wr = stats["correct"] / stats["total"] * 100
        region_summary[rid] = {
            "win_rate": f"{wr:.0f}%",
            "trades": stats["total"],
            "weight": stats["weight"],
            "threshold": stats["threshold"],
            "status": "🟢 good" if wr > 65 else "🟡 ok" if wr > 40 else "🔴 poor",
        }
    
    type_summary = {}
    for stype, stats in feedback.get("signal_type_accuracy", {}).items():
        if stats["total"] == 0:
            continue
        wr = stats["correct"] / stats["total"] * 100
        type_summary[stype] = {
            "win_rate": f"{wr:.0f}%",
            "trades": stats["total"],
            "weight": stats["weight"],
        }
    
    return {
        "total_closed_trades": total_trades,
        "overall_win_rate": f"{total_correct / total_trades * 100:.0f}%" if total_trades else "N/A",
        "total_realized_pnl": f"${total_pnl:,.2f}",
        "regions_learned": len([r for r in feedback.get("region_accuracy", {}).values() if r["total"] >= MIN_EVALUATIONS]),
        "region_details": region_summary,
        "signal_type_details": type_summary,
    }
