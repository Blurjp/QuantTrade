"""
Data Quality Tracker

Tracks which data sources are real vs simulated and provides quality reports.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Data source types."""
    REAL = "real"
    SIMULATED = "simulated"
    PLACEHOLDER = "placeholder"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class DataQualityRecord:
    """Record for a single data quality check."""
    region: str
    data_type: str
    date: str
    source: DataSource
    source_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DataQualityTracker:
    """Track data quality across all detection types."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.records: List[DataQualityRecord] = []
        self._load_existing_state()

    def _load_existing_state(self):
        """Load existing state from file."""
        state_file = self.output_dir / "data_quality_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                    for record_data in data.get("records", []):
                        record = DataQualityRecord(**record_data)
                        self.records.append(record)
                logger.debug(f"Loaded {len(self.records)} existing quality records")
            except Exception as e:
                logger.warning(f"Failed to load quality state: {e}")

    def record_detection(self, result: Dict[str, Any]) -> DataQualityRecord:
        """Record a detection result and return the quality record."""
        data_type = result.get("type", "unknown")
        region = result.get("region", "unknown")
        date = result.get("date", datetime.now().strftime("%Y-%m-%d"))

        metadata = result.get("metadata", {})
        source_name = metadata.get("data_source", "unknown").lower()

        # Determine data source type
        if result.get("status") == "error":
            source = DataSource.ERROR
        elif any(keyword in source_name for keyword in ["simulated", "placeholder", "fake"]):
            source = DataSource.SIMULATED
        elif metadata.get("is_real_data") is True:
            source = DataSource.REAL
        elif source_name == "unknown":
            source = DataSource.UNKNOWN
        else:
            # Try to infer from source name
            if any(keyword in source_name for keyword in ["sentinel", "landsat", "modis", "viirs", "gpm", "nasa", "noaa"]):
                source = DataSource.REAL
            else:
                source = DataSource.PLACEHOLDER

        record = DataQualityRecord(
            region=region,
            data_type=data_type,
            date=date,
            source=source,
            source_name=metadata.get("data_source", "unknown"),
            metadata=metadata,
        )

        self.records.append(record)
        return record

    def get_quality_report(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Generate a quality report for a specific date or overall."""
        if date:
            records = [r for r in self.records if r.date == date]
        else:
            records = self.records

        if not records:
            return {
                "date": date or "all",
                "total_records": 0,
                "real_data_sources": [],
                "simulated_data_sources": [],
                "coverage": "0%",
                "quality_score": 0.0,
            }

        real_regions = set()
        simulated_regions = set()
        error_regions = set()

        source_counts = {
            DataSource.REAL: 0,
            DataSource.SIMULATED: 0,
            DataSource.PLACEHOLDER: 0,
            DataSource.ERROR: 0,
            DataSource.UNKNOWN: 0,
        }

        source_breakdown = {}

        for record in records:
            source_counts[record.source] += 1

            key = f"{record.data_type}:{record.region}"
            source_breakdown[key] = {
                "source": record.source.value,
                "source_name": record.source_name,
            }

            if record.source == DataSource.REAL:
                real_regions.add(key)
            elif record.source == DataSource.SIMULATED:
                simulated_regions.add(key)
            elif record.source == DataSource.ERROR:
                error_regions.add(key)

        total = len(records)
        quality_score = (source_counts[DataSource.REAL] / total * 100) if total > 0 else 0

        return {
            "date": date or "all",
            "total_records": total,
            "real_data_sources": sorted(list(real_regions)),
            "simulated_data_sources": sorted(list(simulated_regions)),
            "error_data_sources": sorted(list(error_regions)),
            "source_counts": {k.value: v for k, v in source_counts.items()},
            "source_breakdown": source_breakdown,
            "coverage": f"{quality_score:.1f}%",
            "quality_score": round(quality_score, 1),
        }

    def save_state(self):
        """Save current state to file."""
        state_file = self.output_dir / "data_quality_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "last_updated": datetime.now().isoformat(),
            "total_records": len(self.records),
            "records": [
                {
                    "region": r.region,
                    "data_type": r.data_type,
                    "date": r.date,
                    "source": r.source.value,
                    "source_name": r.source_name,
                    "metadata": r.metadata,
                    "timestamp": r.timestamp,
                }
                for r in self.records
            ],
        }

        with open(state_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved {len(self.records)} quality records to {state_file}")

    def get_status_emoji(self) -> str:
        """Get an emoji representing overall data quality."""
        report = self.get_quality_report()
        score = report.get("quality_score", 0)

        if score >= 80:
            return "🟢"
        elif score >= 50:
            return "🟡"
        elif score >= 20:
            return "🟠"
        else:
            return "🔴"

    def get_status_message(self) -> str:
        """Get a human-readable status message."""
        report = self.get_quality_report()
        score = report.get("quality_score", 0)
        real = len(report.get("real_data_sources", []))
        simulated = len(report.get("simulated_data_sources", []))
        errors = len(report.get("error_data_sources", []))

        emoji = self.get_status_emoji()

        if score >= 80:
            status = "Good - Most data is from real sources"
        elif score >= 50:
            status = "Fair - Mixed real and simulated data"
        elif score >= 20:
            status = "Poor - Mostly simulated data"
        else:
            status = "Critical - Almost no real data"

        return (
            f"{emoji} Data Quality: {status} "
            f"({score:.0f}% real, {simulated} simulated, {errors} errors)"
        )


# Global instance
_tracker: Optional[DataQualityTracker] = None


def get_tracker(output_dir: str = "outputs") -> DataQualityTracker:
    """Get the global data quality tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = DataQualityTracker(output_dir)
    return _tracker


def track_detection_result(result: Dict[str, Any]) -> DataQualityRecord:
    """Track a detection result and return the quality record."""
    tracker = get_tracker(result.get("output_base", "outputs"))
    return tracker.record_detection(result)


def get_daily_quality_summary(date: Optional[str] = None) -> Dict[str, Any]:
    """Get a daily quality summary."""
    tracker = get_tracker()
    return tracker.get_quality_report(date)
