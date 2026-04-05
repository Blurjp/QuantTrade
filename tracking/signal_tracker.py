"""
Signal History Tracking

Tracks the full signal lifecycle: detected -> opened -> closed -> final P&L.
Stores history in outputs/signal_history.json and calculates accuracy metrics.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SignalTracker:
    """Tracks signal lifecycle from detection through position close."""

    def __init__(self, output_base: str = "outputs"):
        self.output_base = Path(output_base)
        self.history_file = self.output_base / "signal_history.json"
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.signals: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        """Load signal history from file."""
        if self.history_file.exists():
            try:
                return json.loads(self.history_file.read_text())
            except Exception as e:
                logger.warning(f"Failed to load signal history: {e}")
        return []

    def _save(self):
        """Save signal history to file."""
        try:
            self.history_file.write_text(json.dumps(self.signals, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save signal history: {e}")

    def _signal_id(self, signal: Dict) -> str:
        """Generate unique signal ID."""
        return (
            f"{signal.get('signal_type', 'unknown')}"
            f"_{signal.get('region_id', 'unknown')}"
            f"_{signal.get('date', '')}"
        )

    def record_signal_detected(self, signal: Dict):
        """Record a newly detected signal."""
        sig_id = self._signal_id(signal)

        # Skip if already recorded
        for existing in self.signals:
            if existing.get("signal_id") == sig_id:
                return

        entry = {
            "signal_id": sig_id,
            "signal_type": signal.get("signal_type", ""),
            "region_id": signal.get("region_id", ""),
            "region": signal.get("region", signal.get("region_name", "")),
            "date_detected": signal.get("date", datetime.now().strftime("%Y-%m-%d")),
            "direction": signal.get("direction", ""),
            "confidence": signal.get("confidence", 0),
            "instruments": signal.get("instruments", []),
            "rationale": signal.get("rationale", ""),
            "status": "detected",
            "ticker": None,
            "entry_price": None,
            "entry_date": None,
            "exit_price": None,
            "exit_date": None,
            "pnl": None,
            "pnl_pct": None,
        }
        self.signals.append(entry)
        self._save()
        logger.info(f"Signal tracked: {sig_id} ({signal.get('direction')})")

    def record_position_opened(
        self, signal: Dict, ticker: str, entry_price: float
    ):
        """Record that a position was opened for a signal."""
        sig_id = self._signal_id(signal)
        today = datetime.now().strftime("%Y-%m-%d")

        for entry in self.signals:
            if entry.get("signal_id") == sig_id and entry["status"] == "detected":
                entry["status"] = "opened"
                entry["ticker"] = ticker
                entry["entry_price"] = entry_price
                entry["entry_date"] = today
                self._save()
                logger.info(f"Signal opened: {sig_id} -> {ticker} @ ${entry_price:.2f}")
                return

        # If signal wasn't tracked yet, create entry
        entry = {
            "signal_id": sig_id,
            "signal_type": signal.get("signal_type", ""),
            "region_id": signal.get("region_id", ""),
            "region": signal.get("region", signal.get("region_name", "")),
            "date_detected": signal.get("date", today),
            "direction": signal.get("direction", ""),
            "confidence": signal.get("confidence", 0),
            "instruments": signal.get("instruments", []),
            "rationale": signal.get("rationale", ""),
            "status": "opened",
            "ticker": ticker,
            "entry_price": entry_price,
            "entry_date": today,
            "exit_price": None,
            "exit_date": None,
            "pnl": None,
            "pnl_pct": None,
        }
        self.signals.append(entry)
        self._save()
        logger.info(f"Signal tracked+opened: {sig_id} -> {ticker} @ ${entry_price:.2f}")

    def record_position_closed(
        self, ticker: str, exit_price: float, pnl: float
    ):
        """Record that a position was closed."""
        today = datetime.now().strftime("%Y-%m-%d")

        for entry in reversed(self.signals):
            if entry.get("ticker") == ticker and entry["status"] == "opened":
                entry["status"] = "closed"
                entry["exit_price"] = exit_price
                entry["exit_date"] = today
                entry["pnl"] = round(pnl, 2)
                if entry.get("entry_price") and entry["entry_price"] > 0:
                    if entry.get("direction") == "long":
                        entry["pnl_pct"] = round(
                            (exit_price - entry["entry_price"]) / entry["entry_price"] * 100, 2
                        )
                    else:
                        entry["pnl_pct"] = round(
                            (entry["entry_price"] - exit_price) / entry["entry_price"] * 100, 2
                        )
                self._save()
                logger.info(
                    f"Signal closed: {entry['signal_id']} -> {ticker} "
                    f"@ ${exit_price:.2f} P&L=${pnl:.2f}"
                )
                return

        logger.warning(f"No open signal found for ticker {ticker} to close")

    def get_stats(self) -> Dict:
        """Calculate signal accuracy and average return."""
        closed = [s for s in self.signals if s["status"] == "closed"]
        total = len(self.signals)

        if not closed:
            return {
                "total_signals": total,
                "detected": len([s for s in self.signals if s["status"] == "detected"]),
                "opened": len([s for s in self.signals if s["status"] == "opened"]),
                "closed": 0,
                "win_rate": 0,
                "avg_return_pct": 0,
                "total_pnl": 0,
            }

        winners = [s for s in closed if (s.get("pnl") or 0) > 0]
        pnl_pcts = [s["pnl_pct"] for s in closed if s.get("pnl_pct") is not None]
        total_pnl = sum(s.get("pnl", 0) for s in closed)

        return {
            "total_signals": total,
            "detected": len([s for s in self.signals if s["status"] == "detected"]),
            "opened": len([s for s in self.signals if s["status"] == "opened"]),
            "closed": len(closed),
            "win_rate": round(len(winners) / len(closed) * 100, 1) if closed else 0,
            "avg_return_pct": round(sum(pnl_pcts) / len(pnl_pcts), 2) if pnl_pcts else 0,
            "total_pnl": round(total_pnl, 2),
        }

    def get_history(self) -> List[Dict]:
        """Return full signal history."""
        return self.signals
