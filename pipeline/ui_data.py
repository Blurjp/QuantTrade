"""
Helpers for browsing one-day pipeline artifacts in a local UI.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from pipeline.regions import resolve_region_output_base


def list_available_days(output_base: str = "outputs", region_id: str = "hormuz") -> List[str]:
    """List day directories that look like pipeline runs."""
    base = Path(resolve_region_output_base(output_base, region_id))
    if not base.exists():
        return []

    days = []
    for child in base.iterdir():
        if child.is_dir() and child.name[:4].isdigit() and (child / "metrics" / "daily.parquet").exists():
            days.append(child.name)
    return sorted(days)


def build_day_artifact_index(day_dir: str) -> Dict[str, Optional[Path]]:
    """Resolve canonical artifact paths for a day directory."""
    root = Path(day_dir)
    previews = sorted((root / "qa").glob("*.png")) if (root / "qa").exists() else []

    return {
        "root": root,
        "manifest": root / "manifests" / "manifest.parquet",
        "load_log": root / "logs" / "scene_load_log.parquet",
        "detections_parquet": root / "detections" / "daily_detections.parquet",
        "detections_geojson": root / "detections" / "daily_detections.geojson",
        "metrics": root / "metrics" / "daily.parquet",
        "qa_html": root / "qa" / "index.html",
        "run_report": root / "qa" / "run_report.json",
        "previews": previews,
    }


def load_day_bundle(day_dir: str) -> Dict[str, object]:
    """Load the canonical day artifacts into memory."""
    paths = build_day_artifact_index(day_dir)

    def _load_parquet(path: Path) -> pd.DataFrame:
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    report = {}
    if paths["run_report"].exists():
        report = json.loads(paths["run_report"].read_text())

    return {
        "paths": paths,
        "manifest": _load_parquet(paths["manifest"]),
        "load_log": _load_parquet(paths["load_log"]),
        "detections": _load_parquet(paths["detections_parquet"]),
        "metrics": _load_parquet(paths["metrics"]),
        "report": report,
    }
