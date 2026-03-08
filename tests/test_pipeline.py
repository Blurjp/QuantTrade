"""
Unit and integration-style tests for QuantTrade pipeline.
"""

import json
import sys
import types
from datetime import date, datetime, timezone

import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import LineString, Point

from pipeline.crossings import compute_side_of_gate
from pipeline.detection import DETECTION_COLUMNS, detections_to_geojson
from pipeline.manifest import load_aoi
from pipeline.metrics import aggregate_daily_metrics, summarize_observation_day
from pipeline.qa import write_daily_html
from pipeline.run import run_single_day
from pipeline.ui_data import build_day_artifact_index, list_available_days, load_day_bundle


def test_load_aoi():
    aoi = load_aoi("configs/aoi_hormuz.geojson")
    assert aoi is not None
    assert "features" in aoi
    assert len(aoi["features"]) > 0


def test_compute_side_of_gate():
    gate_line = LineString([[56.5, 26.9], [56.5, 27.1]])
    point_left = Point(56.4, 27.0)
    point_right = Point(56.6, 27.0)
    assert compute_side_of_gate(point_left, gate_line) != compute_side_of_gate(point_right, gate_line)


def test_distance_computation():
    from pipeline.tracking import compute_distance

    assert compute_distance(26.9, 56.5, 26.9, 56.5) == 0.0
    assert 10 < compute_distance(26.9, 56.5, 27.0, 56.5) < 12


def test_bias_correction():
    from pipeline.calibration import apply_bias_correction

    metrics_df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "gc_total": [10, 20],
        "gc_in": [5, 10],
        "gc_out": [5, 10],
        "coverage_score": [0.8, 0.9],
        "throughput_index_total": [0.5, 1.0],
    })
    coefficients = {"intercept": 0.5, "gc_total_coef": 1.2, "coverage_coef": 0.3}

    corrected = apply_bias_correction(metrics_df, coefficients)
    assert "throughput_index_corrected" in corrected.columns
    assert "bias_factor" in corrected.columns


def test_observation_summary_and_daily_metrics_include_coverage_metadata():
    aoi = load_aoi("configs/aoi_hormuz.geojson")
    manifest_df = pd.DataFrame({
        "datetime": ["2024-01-15T01:00:00Z", "2024-01-15T07:00:00Z"],
        "geometry": [
            json.dumps({
                "type": "Polygon",
                "coordinates": [[[56.2, 26.8], [56.7, 26.8], [56.7, 27.2], [56.2, 27.2], [56.2, 26.8]]],
            }),
            json.dumps({
                "type": "Polygon",
                "coordinates": [[[56.6, 26.9], [57.0, 26.9], [57.0, 27.3], [56.6, 27.3], [56.6, 26.9]]],
            }),
        ],
        "orbit": ["ascending", "descending"],
        "incidence_angle": [32.0, 38.0],
    })
    load_log_df = pd.DataFrame({
        "scene_id": ["a", "b"],
        "status": ["loaded", "error"],
    })

    summary = summarize_observation_day(manifest_df, load_log_df, aoi["features"][0]["geometry"])
    assert summary["coverage_score"] > 0
    assert summary["num_scenes"] == 2
    assert summary["loaded_scenes"] == 1
    assert summary["max_scene_gap_hours"] == 6.0
    assert "ascending:1" in summary["orbit_summary"]
    assert summary["incidence_angle_mean"] == 35.0

    metrics_df = aggregate_daily_metrics(
        target_date="2024-01-15",
        manifest_df=manifest_df,
        load_log_df=load_log_df,
        crossings_df=pd.DataFrame(),
        aoi_geom=aoi["features"][0]["geometry"],
    )
    assert metrics_df.loc[0, "coverage_score"] > 0
    assert metrics_df.loc[0, "num_scenes"] == 2
    assert metrics_df.loc[0, "loaded_scenes"] == 1


def test_detection_geojson_export_and_html_summary(tmp_path):
    detections_df = pd.DataFrame([{
        "date": "2024-01-15",
        "scene_id": "scene-1",
        "datetime": "2024-01-15T00:00:00Z",
        "detection_id": "scene-1_0",
        "bbox_geom_wkt": "POLYGON ((56.4 27.0, 56.5 27.0, 56.5 27.1, 56.4 27.1, 56.4 27.0))",
        "score": 0.9,
        "centroid_lon": 56.45,
        "centroid_lat": 27.05,
        "area_px": 5,
        "area_km2": 0.01,
        "mean_intensity_db": -12.0,
        "bbox_width_px": 2,
        "bbox_height_px": 3,
        "aspect_ratio": 0.66,
    }], columns=DETECTION_COLUMNS)
    geojson_path = tmp_path / "daily_detections.geojson"
    detections_to_geojson(detections_df, str(geojson_path))
    geojson = json.loads(geojson_path.read_text())
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 1

    preview = tmp_path / "scene-1.png"
    preview.write_bytes(b"png")
    html_path = tmp_path / "index.html"
    write_daily_html(
        target_date="2024-01-15",
        output_path=str(html_path),
        metrics_df=pd.DataFrame([{"date": "2024-01-15", "coverage_score": 0.75}]),
        load_log_df=pd.DataFrame([{"scene_id": "scene-1", "status": "loaded"}]),
        detections_df=detections_df,
        preview_paths=[str(preview)],
        manifest_path="manifest.parquet",
        detections_geojson_path=str(geojson_path),
        metrics_path="daily.parquet",
        load_log_path="scene_load_log.parquet",
    )
    html = html_path.read_text()
    assert "QuantTrade QA: 2024-01-15" in html
    assert "daily_detections.geojson" in html


def test_ui_data_lists_and_loads_day_artifacts(tmp_path):
    day_dir = tmp_path / "2024-01-15"
    (day_dir / "manifests").mkdir(parents=True)
    (day_dir / "logs").mkdir(parents=True)
    (day_dir / "detections").mkdir(parents=True)
    (day_dir / "metrics").mkdir(parents=True)
    (day_dir / "qa").mkdir(parents=True)

    pd.DataFrame([{"date": "2024-01-15", "coverage_score": 0.8}]).to_parquet(day_dir / "metrics" / "daily.parquet", index=False)
    pd.DataFrame([{"scene_id": "scene-1"}]).to_parquet(day_dir / "manifests" / "manifest.parquet", index=False)
    pd.DataFrame([{"scene_id": "scene-1", "status": "loaded"}]).to_parquet(day_dir / "logs" / "scene_load_log.parquet", index=False)
    pd.DataFrame(columns=DETECTION_COLUMNS).to_parquet(day_dir / "detections" / "daily_detections.parquet", index=False)
    (day_dir / "qa" / "run_report.json").write_text(json.dumps({"status": "completed"}))
    (day_dir / "qa" / "scene-1.png").write_bytes(b"png")

    assert list_available_days(str(tmp_path)) == ["2024-01-15"]
    paths = build_day_artifact_index(str(day_dir))
    assert paths["metrics"].name == "daily.parquet"
    assert len(paths["previews"]) == 1

    bundle = load_day_bundle(str(day_dir))
    assert bundle["report"]["status"] == "completed"
    assert len(bundle["metrics"]) == 1


def _make_dataset():
    data = xr.Dataset(
        {
            "vv": (("y", "x"), [[1.0, 2.0], [3.0, 4.0]]),
            "vh": (("y", "x"), [[1.5, 1.0], [2.0, 2.5]]),
        },
        coords={"x": [56.3, 56.4], "y": [27.0, 27.1]},
    )
    return data


class _FakeItem:
    def __init__(self, item_id: str, dt: datetime):
        self.id = item_id
        self.datetime = dt
        self.geometry = {
            "type": "Polygon",
            "coordinates": [[[56.2, 26.8], [56.7, 26.8], [56.7, 27.2], [56.2, 27.2], [56.2, 26.8]]],
        }


def _install_fake_odc(monkeypatch, dataset_factory):
    fake_stac = types.SimpleNamespace(load=dataset_factory)
    monkeypatch.setitem(sys.modules, "odc", types.SimpleNamespace(stac=fake_stac))


def test_run_single_day_writes_full_artifact_set_with_mocked_pipeline(tmp_path, monkeypatch):
    item = _FakeItem("scene-1", datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc))
    manifest_df = pd.DataFrame({
        "item_id": ["scene-1"],
        "datetime": ["2024-01-15T00:00:00Z"],
        "geometry": [json.dumps({
            "type": "Polygon",
            "coordinates": [[[56.2, 26.8], [56.7, 26.8], [56.7, 27.2], [56.2, 27.2], [56.2, 26.8]]],
        })],
        "orbit": ["ascending"],
        "incidence_angle": [34.0],
    })
    items_path = tmp_path / "items.ndjson"
    items_path.write_text("{}\n")

    monkeypatch.setattr("pipeline.run.run_manifest_builder", lambda **kwargs: (manifest_df, items_path))
    monkeypatch.setattr("pipeline.loader.load_stac_items", lambda path: [item])
    monkeypatch.setattr("pipeline.run.sign_items", lambda items: items)
    _install_fake_odc(monkeypatch, lambda *args, **kwargs: _make_dataset())

    detections_df = pd.DataFrame([{
        "date": "2024-01-15",
        "scene_id": "scene-1",
        "datetime": "2024-01-15T00:00:00Z",
        "detection_id": "scene-1_0",
        "bbox_geom_wkt": "POLYGON ((56.4 27.0, 56.5 27.0, 56.5 27.1, 56.4 27.1, 56.4 27.0))",
        "score": 0.9,
        "centroid_lon": 56.45,
        "centroid_lat": 27.05,
        "area_px": 5,
        "area_km2": 0.01,
        "mean_intensity_db": -12.0,
        "bbox_width_px": 2,
        "bbox_height_px": 3,
        "aspect_ratio": 0.66,
    }], columns=DETECTION_COLUMNS)
    monkeypatch.setattr(
        "pipeline.run.run_detection_pipeline",
        lambda *args, **kwargs: (detections_df.copy(), {"threshold_db": -18.0, "connected_components": 2, "detections_after_filter": 1}),
    )

    report = run_single_day(date(2024, 1, 15), output_base=str(tmp_path))
    date_dir = tmp_path / "2024-01-15"
    assert report["status"] == "completed"
    assert (date_dir / "manifests" / "manifest.parquet").exists()
    assert (date_dir / "logs" / "scene_load_log.parquet").exists()
    assert (date_dir / "detections" / "daily_detections.parquet").exists()
    assert (date_dir / "detections" / "daily_detections.geojson").exists()
    assert (date_dir / "metrics" / "daily.parquet").exists()
    assert (date_dir / "qa" / "scene-1.png").exists()
    assert (date_dir / "qa" / "index.html").exists()


def test_run_single_day_preserves_artifacts_on_scene_load_failure(tmp_path, monkeypatch):
    item = _FakeItem("scene-err", datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc))
    manifest_df = pd.DataFrame({
        "item_id": ["scene-err"],
        "datetime": ["2024-01-15T00:00:00Z"],
        "geometry": [json.dumps({
            "type": "Polygon",
            "coordinates": [[[56.2, 26.8], [56.7, 26.8], [56.7, 27.2], [56.2, 27.2], [56.2, 26.8]]],
        })],
        "orbit": ["ascending"],
        "incidence_angle": [34.0],
    })
    items_path = tmp_path / "items.ndjson"
    items_path.write_text("{}\n")

    monkeypatch.setattr("pipeline.run.run_manifest_builder", lambda **kwargs: (manifest_df, items_path))
    monkeypatch.setattr("pipeline.loader.load_stac_items", lambda path: [item])
    monkeypatch.setattr("pipeline.run.sign_items", lambda items: items)
    _install_fake_odc(monkeypatch, lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("load failed")))

    report = run_single_day(date(2024, 1, 15), output_base=str(tmp_path))
    date_dir = tmp_path / "2024-01-15"
    assert report["status"] == "completed_with_errors"
    assert (date_dir / "manifests" / "manifest.parquet").exists()
    assert (date_dir / "logs" / "scene_load_log.parquet").exists()
    assert (date_dir / "detections" / "daily_detections.parquet").exists()
    assert (date_dir / "detections" / "daily_detections.geojson").exists()
    assert (date_dir / "metrics" / "daily.parquet").exists()
    assert (date_dir / "qa" / "index.html").exists()
