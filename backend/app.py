"""
Minimal API service for QuantTrade output artifacts.
"""

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from pipeline.regions import list_regions, resolve_region_output_base
from pipeline.ui_data import list_available_days, load_day_bundle


def _frame_records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []

    normalized = df.copy()
    normalized = normalized.where(pd.notnull(normalized), None)
    return normalized.to_dict(orient="records")


def create_app() -> FastAPI:
    app = FastAPI(title="QuantTrade API", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/regions")
    def regions() -> dict:
        return {"regions": list_regions()}

    @app.get("/days")
    def days(output_base: str = "outputs", region: str = "hormuz") -> dict:
        return {"days": list_available_days(output_base, region)}

    @app.get("/days/{day}")
    def day_bundle(day: str, output_base: str = "outputs", region: str = "hormuz") -> dict:
        day_dir = Path(resolve_region_output_base(output_base, region)) / day
        if not day_dir.exists():
            raise HTTPException(status_code=404, detail=f"Day not found: {day}")

        bundle = load_day_bundle(str(day_dir))
        paths = bundle["paths"]

        return {
            "day": day,
            "region": region,
            "report": bundle["report"],
            "metrics": _frame_records(bundle["metrics"]),
            "detections": _frame_records(bundle["detections"]),
            "load_log": _frame_records(bundle["load_log"]),
            "manifest": _frame_records(bundle["manifest"]),
            "paths": {
                "root": str(paths["root"]),
                "manifest": str(paths["manifest"]),
                "load_log": str(paths["load_log"]),
                "detections_parquet": str(paths["detections_parquet"]),
                "detections_geojson": str(paths["detections_geojson"]),
                "metrics": str(paths["metrics"]),
                "qa_html": str(paths["qa_html"]),
                "run_report": str(paths["run_report"]),
                "previews": [str(path) for path in paths["previews"]],
            },
        }

    @app.get("/calibration")
    def calibration(output_base: str = "outputs", region: str = "hormuz") -> dict:
        calibration_dir = Path(resolve_region_output_base(output_base, region)) / "calibration"
        report_path = calibration_dir / "calibration_report.json"
        metrics_path = calibration_dir / "corrected_metrics.parquet"

        report = {}
        if report_path.exists():
            report = json.loads(report_path.read_text())

        metrics = pd.read_parquet(metrics_path) if metrics_path.exists() else pd.DataFrame()
        return {"report": report, "metrics": _frame_records(metrics)}

    return app


app = create_app()
