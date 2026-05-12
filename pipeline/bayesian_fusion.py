"""
Bayesian Signal Fusion Engine — v2 Learning System

Replaces simple confidence-threshold with probabilistic signal fusion:
- Each signal source has a learned prior (historical accuracy)
- Multiple signals on the same instrument are fused via Bayesian update
- Conflicting signals are resolved by weighted probability, not highest confidence
- Time decay: recent trades count more than old ones

Plus Kelly Criterion for position sizing (v3):
- Bet size = f(win_rate, payoff_ratio)
- Higher conviction + better odds = larger position
"""

import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FEEDBACK_FILE = "signal_feedback.json"

# ─── Kelly Criterion ───────────────────────────────────────────────

def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Calculate optimal bet fraction using Kelly Criterion.
    
    f* = (p * b - q) / b
    where p = win_rate, q = 1-p, b = avg_win / avg_loss
    
    Returns fraction of capital to risk (0.0 to 0.25, capped for safety).
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.05  # Default 5% if insufficient data
    
    b = avg_win / avg_loss  # payoff ratio
    q = 1 - win_rate
    f = (win_rate * b - q) / b
    
    # Half-Kelly for safety (reduces volatility)
    f = f * 0.5
    
    # Cap between 2% and 20%
    return max(0.02, min(0.20, f))


# ─── Time-Decayed Statistics ───────────────────────────────────────

def _time_weight(trade_date: str, half_life_days: float = 30.0) -> float:
    """
    Exponential decay weight for a trade based on age.
    half_life_days: trades this old have 50% weight.
    """
    try:
        trade_dt = datetime.fromisoformat(trade_date.replace("Z", "+00:00").replace("+00:00", ""))
    except (ValueError, AttributeError):
        return 0.5  # Unknown date → medium weight
    
    age_days = (datetime.now() - trade_dt).total_seconds() / 86400
    if age_days < 0:
        age_days = 0
    return math.exp(-math.log(2) * age_days / half_life_days)


# ─── Bayesian Engine ───────────────────────────────────────────────

class BayesianSignalFusion:
    """
    Fuses multiple signals on the same instrument using Bayesian updating.
    
    For each signal source (signal_type), we maintain:
    - Prior P(correct) = historical win rate
    - Likelihood from signal confidence
    
    Posterior = combined belief after seeing all signals.
    """
    
    def __init__(self, output_base: str = "outputs"):
        self.output_base = output_base
        self._stats_cache = None
        self._cache_time = None
    
    def _load_feedback(self) -> Dict:
        path = Path(self.output_base) / FEEDBACK_FILE
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass
        return {"trades": [], "region_accuracy": {}, "signal_type_accuracy": {}}
    
    def _get_signal_type_stats(self) -> Dict:
        """Get time-decoded win rates per signal type."""
        feedback = self._load_feedback()
        
        stats = {}
        for stype, data in feedback.get("signal_type_accuracy", {}).items():
            total = data.get("total", 0)
            correct = data.get("correct", 0)
            
            if total < 3:
                stats[stype] = {"win_rate": 0.5, "total": total, "weight": 1.0}
                continue
            
            # Calculate time-decoded win rate
            type_trades = [t for t in feedback["trades"] 
                          if t.get("signal_type") == stype and t.get("was_correct") is not None]
            
            if type_trades:
                weighted_correct = sum(
                    _time_weight(t.get("closed_at", "")) 
                    for t in type_trades if t["was_correct"]
                )
                weighted_total = sum(
                    _time_weight(t.get("closed_at", "")) 
                    for t in type_trades
                )
                win_rate = weighted_correct / weighted_total if weighted_total > 0 else 0.5
            else:
                win_rate = correct / total
            
            stats[stype] = {
                "win_rate": win_rate,
                "total": total,
                "weight": data.get("weight", 1.0),
            }
        
        return stats
    
    def _get_region_stats(self) -> Dict:
        """Get time-decoded win rates per region."""
        feedback = self._load_feedback()
        
        stats = {}
        for rid, data in feedback.get("region_accuracy", {}).items():
            total = data.get("total", 0)
            correct = data.get("correct", 0)
            
            if total < 3:
                stats[rid] = {"win_rate": 0.5, "total": total}
                continue
            
            # Time-decoded
            region_trades = [t for t in feedback["trades"] 
                           if t.get("region_id") == rid and t.get("was_correct") is not None]
            
            if region_trades:
                weighted_correct = sum(
                    _time_weight(t.get("closed_at", ""))
                    for t in region_trades if t["was_correct"]
                )
                weighted_total = sum(
                    _time_weight(t.get("closed_at", ""))
                    for t in region_trades
                )
                win_rate = weighted_correct / weighted_total if weighted_total > 0 else 0.5
            else:
                win_rate = correct / total
            
            stats[rid] = {"win_rate": win_rate, "total": total}
        
        return stats
    
    def _get_payoff_stats(self) -> Dict:
        """Get average win/loss amounts per signal type for Kelly."""
        feedback = self._load_feedback()
        
        payoff = {}
        for t in feedback.get("trades", []):
            if t.get("was_correct") is None:
                continue
            stype = t.get("signal_type", "unknown")
            pnl_pct = abs(t.get("pnl_pct", 0))
            
            if stype not in payoff:
                payoff[stype] = {"wins": [], "losses": []}
            
            if t["was_correct"]:
                payoff[stype]["wins"].append(pnl_pct)
            else:
                payoff[stype]["losses"].append(pnl_pct)
        
        result = {}
        for stype, data in payoff.items():
            avg_win = np.mean(data["wins"]) if data["wins"] else 5.0
            avg_loss = np.mean(data["losses"]) if data["losses"] else 5.0
            result[stype] = {"avg_win_pct": avg_win, "avg_loss_pct": avg_loss}
        
        return result
    
    def fuse_signals(
        self, 
        signals: List[Dict],
        ticker: str,
    ) -> Dict:
        """
        Fuse multiple signals targeting the same ticker.
        
        Uses Bayesian updating:
        - Start with prior P(direction correct) = 0.5
        - For each signal, update: P(correct | signal) using learned accuracy
        - Final: fused probability and direction
        
        Returns: {
            "direction": "long"/"short"/"neutral",
            "fused_confidence": float,  # 0-100
            "fused_probability": float,  # 0-1
            "kelly_fraction": float,     # position size as % of capital
            "sources": [...],            # which signals contributed
            "conflict": bool,            # were signals conflicting?
        }
        """
        # Filter signals relevant to this ticker
        relevant = []
        for sig in signals:
            instruments = sig.get("instruments", [])
            if isinstance(instruments, str):
                instruments = [instruments]
            if ticker in instruments or not instruments:
                relevant.append(sig)
        
        if not relevant:
            return {
                "direction": "neutral",
                "fused_confidence": 0,
                "fused_probability": 0.5,
                "kelly_fraction": 0.05,
                "sources": [],
                "conflict": False,
            }
        
        signal_type_stats = self._get_signal_type_stats()
        region_stats = self._get_region_stats()
        payoff_stats = self._get_payoff_stats()
        
        # Bayesian fusion
        # Prior: P(correct) = 0.5 (we don't know)
        log_prior_long = 0.0  # log-odds, 0 = 50/50
        log_prior_short = 0.0
        
        long_sources = []
        short_sources = []
        
        for sig in relevant:
            direction = sig.get("trade_direction", sig.get("direction", "neutral"))
            if direction == "neutral":
                continue
            
            confidence = sig.get("confidence", 50) / 100.0  # Normalize to 0-1
            stype = sig.get("signal_type", "unknown")
            rid = sig.get("region_id", "")
            
            # Get learned reliability for this signal source
            st_stats = signal_type_stats.get(stype, {"win_rate": 0.5})
            r_stats = region_stats.get(rid, {"win_rate": 0.5})
            
            # Combined reliability: signal_type reliability × region reliability
            # Use geometric mean to be conservative
            # Floor at 0.55 so unlearned sources still contribute evidence
            raw_reliability = math.sqrt(st_stats["win_rate"] * r_stats["win_rate"])
            reliability = max(raw_reliability, 0.6)
            
            # Bayesian update: log-odds += log(reliability / (1 - reliability)) * confidence
            # This means: if reliability > 0.5 (better than random), positive evidence
            if reliability > 0 and reliability < 1:
                evidence_strength = math.log(reliability / (1 - reliability))
                weighted_evidence = evidence_strength * confidence
                
                if direction == "long":
                    log_prior_long += weighted_evidence
                    long_sources.append({
                        "type": stype,
                        "region": rid,
                        "confidence": confidence * 100,
                        "reliability": reliability,
                    })
                elif direction == "short":
                    log_prior_short += weighted_evidence
                    short_sources.append({
                        "type": stype,
                        "region": rid,
                        "confidence": confidence * 100,
                        "reliability": reliability,
                    })
        
        # Convert log-odds to probability
        # P(long) = sigmoid(log_prior_long) / (sigmoid(log_prior_long) + sigmoid(log_prior_short))
        p_long = 1 / (1 + math.exp(-log_prior_long)) if log_prior_long != 0 else 0.5
        p_short = 1 / (1 + math.exp(-log_prior_short)) if log_prior_short != 0 else 0.5
        
        total = p_long + p_short
        if total > 0:
            p_long /= total
            p_short /= total
        
        # Determine direction
        conflict = bool(long_sources and short_sources)
        
        if p_long > 0.55:
            direction = "long"
            fused_prob = p_long
            sources = long_sources
        elif p_short > 0.55:
            direction = "short"
            fused_prob = p_short
            sources = short_sources
        else:
            direction = "neutral"
            fused_prob = max(p_long, p_short)
            sources = long_sources + short_sources
        
        # Fused confidence: convert probability to 0-100 scale
        fused_confidence = fused_prob * 100
        
        # Kelly Criterion for position sizing
        # Use the dominant signal type's payoff stats
        dominant_type = sources[0]["type"] if sources else "unknown"
        pay = payoff_stats.get(dominant_type, {"avg_win_pct": 5.0, "avg_loss_pct": 5.0})
        kelly = kelly_fraction(fused_prob, pay["avg_win_pct"], pay["avg_loss_pct"])
        
        # Scale Kelly by number of agreeing sources
        source_boost = min(1.5, 1.0 + len(sources) * 0.1)
        kelly *= source_boost
        kelly = min(0.20, kelly)  # Never more than 20%
        
        return {
            "direction": direction,
            "fused_confidence": round(fused_confidence, 1),
            "fused_probability": round(fused_prob, 3),
            "kelly_fraction": round(kelly, 3),
            "sources": sources,
            "conflict": conflict,
            "long_probability": round(p_long, 3),
            "short_probability": round(p_short, 3),
        }
    
    def get_trading_decisions(
        self, 
        all_signals: List[Dict],
        held_tickers: List[str],
        cash: float,
    ) -> List[Dict]:
        """
        Generate trading decisions for all candidate instruments.
        
        Returns list of decisions sorted by conviction:
        [{
            "ticker": str,
            "direction": str,
            "fused_confidence": float,
            "kelly_fraction": float,
            "position_value": float,
            "sources": list,
            "conflict": bool,
        }]
        """
        # Collect all unique tickers from signals
        tickers = set()
        for sig in all_signals:
            instruments = sig.get("instruments", [])
            if isinstance(instruments, str):
                instruments = [instruments]
            for inst in instruments:
                if isinstance(inst, str) and inst not in held_tickers:
                    tickers.add(inst)
        
        decisions = []
        for ticker in sorted(tickers):
            fused = self.fuse_signals(all_signals, ticker)
            
            if fused["direction"] == "neutral":
                continue
            
            if fused["fused_confidence"] < 55:
                continue  # Too uncertain
            
            position_value = cash * fused["kelly_fraction"]
            if position_value < 500:
                continue  # Too small
            
            decisions.append({
                "ticker": ticker,
                **fused,
                "position_value": round(position_value, 2),
            })
        
        # Sort by fused confidence (highest first)
        decisions.sort(key=lambda x: -x["fused_confidence"])
        return decisions


# ─── Convenience Functions ──────────────────────────────────────────

def get_fusion_engine(output_base: str = "outputs") -> BayesianSignalFusion:
    """Get a fusion engine instance."""
    return BayesianSignalFusion(output_base=output_base)
