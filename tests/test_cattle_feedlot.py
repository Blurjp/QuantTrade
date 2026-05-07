"""
Tests for cattle feedlot satellite monitoring.

Tests registry loading, thermal analysis, pasture analysis,
signal generation, and the run() entry point.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from pipeline.cattle_feedlot import (
    CattleFeedlotMonitor,
    _load_feedlot_regions,
    _load_pasture_regions,
    INSTRUMENTS,
    run,
)


class TestRegistryLoading:
    def test_feedlot_regions_loaded(self):
        regions = _load_feedlot_regions()
        assert len(regions) >= 8
        assert "texas_panhandle" in regions
        assert "sw_kansas" in regions
        assert "central_nebraska" in regions

    def test_feedlot_region_fields(self):
        regions = _load_feedlot_regions()
        for rid, cfg in regions.items():
            assert "name" in cfg, f"{rid} missing name"
            assert "capacity_share" in cfg, f"{rid} missing capacity_share"
            assert "state" in cfg, f"{rid} missing state"
            assert cfg["capacity_share"] > 0, f"{rid} has zero capacity_share"

    def test_pasture_regions_loaded(self):
        regions = _load_pasture_regions()
        assert len(regions) >= 3
        assert "flint_hills" in regions
        assert "sandhills_ne" in regions

    def test_thermal_id_mapping(self):
        regions = _load_feedlot_regions()
        assert regions["texas_panhandle"]["thermal_id"] == "feedlot_texas_panhandle"
        assert regions["sw_kansas"]["thermal_id"] == "feedlot_sw_kansas"
        assert regions["central_nebraska"]["thermal_id"] == "feedlot_central_nebraska"
        assert regions["central_iowa"].get("thermal_id") is None

    def test_veg_id_mapping(self):
        regions = _load_feedlot_regions()
        assert regions["texas_panhandle"]["veg_id"] == "usa_texas_panhandle_feedlot"
        pastures = _load_pasture_regions()
        assert pastures["flint_hills"]["veg_id"] == "usa_flint_hills_pasture"


class TestCattleFeedlotMonitor:
    @pytest.fixture
    def monitor(self, tmp_path):
        return CattleFeedlotMonitor(output_base=str(tmp_path))

    def test_init(self, monitor, tmp_path):
        assert monitor.output_base == tmp_path
        assert len(monitor.regions) >= 8
        assert len(monitor.pasture_regions) >= 3

    def test_thermal_no_data(self, monitor):
        result = monitor.analyze_feedlot_thermal("central_iowa")
        assert result["thermal_anomaly"] == 0
        assert result["source"] == "no_data"

    def test_thermal_unknown_region(self, monitor):
        result = monitor.analyze_feedlot_thermal("nonexistent_region")
        assert result == {}

    def test_thermal_with_data(self, monitor, tmp_path):
        thermal_dir = tmp_path / "thermal_infrared"
        thermal_dir.mkdir(parents=True)
        signal_file = thermal_dir / "signal_feedlot_texas_panhandle_20260427.json"
        signal_file.write_text(json.dumps({
            "anomaly_pct": 55.0,
            "date": "2026-04-27",
        }))
        result = monitor.analyze_feedlot_thermal("texas_panhandle")
        assert result["thermal_anomaly"] == 55.0
        assert result["source"] == "thermal_infrared"

    def test_pasture_no_data(self, monitor):
        results = monitor.analyze_pasture_health()
        assert len(results) >= 3
        for rid, data in results.items():
            assert data["ndvi"] == 0
            assert data["status"] == "no_data"

    def test_pasture_with_data(self, monitor, tmp_path):
        veg_dir = tmp_path / "vegetation"
        veg_dir.mkdir(parents=True)
        veg_file = veg_dir / "signal_usa_flint_hills_pasture_20260427.json"
        veg_file.write_text(json.dumps({
            "current_ndvi": 0.55,
            "ndvi_anomaly_pct": 10.0,
            "status": "normal",
        }))
        results = monitor.analyze_pasture_health()
        assert results["flint_hills"]["ndvi"] == 0.55
        assert results["flint_hills"]["status"] == "normal"

    def test_generate_signal_structure(self, monitor):
        signals = monitor.generate_signal()
        assert len(signals) >= 9

        aggregate = [s for s in signals if s["region_id"] == "usa_cattle_aggregate"]
        assert len(aggregate) == 1
        agg = aggregate[0]
        assert agg["signal_type"] == "cattle_feedlot"
        assert agg["direction"] in ("long", "short", "neutral")
        assert 10 <= agg["confidence"] <= 100
        assert "rationale" in agg
        assert "weighted_thermal" in agg
        assert "avg_pasture_ndvi" in agg

    def test_generate_signal_per_region_fields(self, monitor):
        signals = monitor.generate_signal()
        per_region = [s for s in signals if s["region_id"] != "usa_cattle_aggregate"]
        assert len(per_region) >= 8
        for s in per_region:
            assert s["region_type"] == "livestock"
            assert s["country"] == "USA"
            assert "state" in s
            assert s["signal_type"] == "cattle_feedlot"
            assert s["direction"] in ("long", "short", "neutral")
            assert "confidence_label" in s
            assert s["confidence_label"] in ("High", "Medium", "Low")

    def test_signal_scoring_high_thermal_low_ndvi_is_long(self, monitor):
        thermal = {rid: {"region_id": rid, "thermal_anomaly": 60, "source": "test"}
                   for rid in monitor.regions}
        with patch.object(monitor, 'analyze_feedlot_thermal', side_effect=lambda rid: thermal[rid]), \
             patch.object(monitor, 'analyze_pasture_health', return_value={
                 "flint_hills": {"ndvi": 0.2, "anomaly_pct": 0, "status": "poor"},
             }):
            signals = monitor.generate_signal()
        agg = [s for s in signals if s["region_id"] == "usa_cattle_aggregate"][0]
        assert agg["direction"] == "long"

    def test_signal_scoring_good_pasture_is_short(self, monitor):
        thermal = {rid: {"region_id": rid, "thermal_anomaly": 0, "source": "test"}
                   for rid in monitor.regions}
        with patch.object(monitor, 'analyze_feedlot_thermal', side_effect=lambda rid: thermal[rid]), \
             patch.object(monitor, 'analyze_pasture_health', return_value={
                 "flint_hills": {"ndvi": 0.7, "anomaly_pct": 0, "status": "good"},
             }):
            signals = monitor.generate_signal()
        agg = [s for s in signals if s["region_id"] == "usa_cattle_aggregate"][0]
        assert agg["direction"] == "short"

    def test_signal_scoring_neutral(self, monitor):
        thermal = {rid: {"region_id": rid, "thermal_anomaly": 0, "source": "test"}
                   for rid in monitor.regions}
        with patch.object(monitor, 'analyze_feedlot_thermal', side_effect=lambda rid: thermal[rid]), \
             patch.object(monitor, 'analyze_pasture_health', return_value={
                 "flint_hills": {"ndvi": 0, "anomaly_pct": 0, "status": "no_data"},
             }):
            signals = monitor.generate_signal()
        agg = [s for s in signals if s["region_id"] == "usa_cattle_aggregate"][0]
        assert agg["direction"] == "neutral"
        assert agg["supply_score"] == 0


class TestRunEntryPoint:
    def test_run_saves_signals(self, tmp_path):
        signals = run(output_base=str(tmp_path))
        assert len(signals) >= 9

        out_dir = tmp_path / "cattle_feedlot"
        assert out_dir.exists()
        files = list(out_dir.glob("signals_*.json"))
        assert len(files) == 1

        saved = json.loads(files[0].read_text())
        assert len(saved) == len(signals)

    def test_instruments_defined(self):
        assert "beef_bullish" in INSTRUMENTS
        assert "beef_bearish" in INSTRUMENTS
        assert "feed_proxy" in INSTRUMENTS
        assert "LE=F" in INSTRUMENTS["beef_bullish"]
