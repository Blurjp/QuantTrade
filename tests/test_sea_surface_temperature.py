"""
Tests for sea surface temperature monitoring.

Tests SST data fetching, ENSO detection, and signal generation.
"""

import json
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestSeaSurfaceTemperatureMonitor:
    """Tests for SeaSurfaceTemperatureMonitor class."""

    @pytest.fixture
    def monitor(self, tmp_path):
        """Create a SeaSurfaceTemperatureMonitor instance."""
        from pipeline.sea_surface_temperature import SeaSurfaceTemperatureMonitor
        return SeaSurfaceTemperatureMonitor(output_base=str(tmp_path))

    def test_initialization(self, monitor, tmp_path):
        """Test monitor initialization."""
        assert monitor.output_base == tmp_path
        assert monitor.cache_days == 30
        assert "nino34" in monitor.regions
        assert "gulf_mexico" in monitor.regions

    def test_regions_have_required_fields(self, monitor):
        """Test that all regions have required configuration fields."""
        required_fields = ["name", "bbox", "ocean", "type", "instruments", "baseline_sst"]

        for region_id, region in monitor.regions.items():
            for field in required_fields:
                assert field in region, f"Region {region_id} missing field {field}"

    def test_get_regional_summary(self, monitor):
        """Test regional summary generation."""
        summary = monitor.get_regional_summary()

        assert summary["monitoring_type"] == "sea_surface_temperature"
        assert "satellites" in summary
        assert "metrics" in summary
        assert summary["total_regions"] == len(monitor.regions)
        assert len(summary["trading_instruments"]) > 0

    def test_fetch_sst_data(self, monitor):
        """Test SST data fetching."""
        data = monitor.fetch_sst_data("nino34", "2024-03-15")

        assert data is not None
        assert "sst_celsius" in data
        assert "sst_anomaly" in data
        assert "baseline_sst" in data
        assert "enso_state" in data
        assert 10 <= data["sst_celsius"] <= 35

    def test_fetch_sst_data_different_seasons(self, monitor):
        """Test that different seasons produce different SST."""
        summer_data = monitor.fetch_sst_data("nino34", "2024-07-15")
        winter_data = monitor.fetch_sst_data("nino34", "2024-01-15")

        assert summer_data["sst_celsius"] != winter_data["sst_celsius"]

    def test_fetch_sst_unknown_region(self, monitor):
        """Test fetching SST for unknown region returns None."""
        data = monitor.fetch_sst_data("unknown_region", "2026-03-15")

        assert data is None

    def test_enso_state_detection(self, monitor):
        """Test ENSO state detection logic."""
        data = monitor.fetch_sst_data("nino34", "2024-03-15")

        assert data["enso_state"] in ["el_nino", "la_nina", "neutral"]

    def test_calculate_baseline(self, monitor):
        """Test baseline calculation."""
        # Mock fetch_sst_data to avoid slow network calls
        mock_data = {
            "sst_celsius": 28.0,
            "sst_anomaly": 0.0,
            "quality": "good",
        }

        with patch.object(monitor, 'fetch_sst_data', return_value=mock_data):
            baseline = monitor.calculate_baseline("nino34", days=10)

        assert "sst" in baseline
        assert "anomaly" in baseline
        assert baseline["sst"]["mean"] > 0

    def test_detect_anomaly(self, monitor):
        """Test anomaly detection."""
        current_data = {
            "sst_celsius": 28.5,
            "sst_anomaly": 1.0,
        }
        baseline = {
            "sst": {"mean": 27.5, "std": 0.5},
            "anomaly": {"mean": 0.0, "std": 0.3},
        }

        anomaly = monitor.detect_anomaly(current_data, baseline)

        assert "sst_z_score" in anomaly
        assert "combined_z_score" in anomaly

    def test_generate_signal(self, monitor):
        """Test signal generation."""
        # Mock fetch_sst_data and calculate_baseline to avoid slow operations
        mock_data = {
            "sst_celsius": 28.0,
            "sst_anomaly": 0.0,
            "baseline_sst": 27.5,
            "enso_state": "neutral",
            "quality": "good",
            "impact": "neutral",
            "region_id": "nino34",
            "region_name": "Niño 3.4 Region",
            "region_type": "enso_indicator",
            "ocean": "Pacific",
            "date": "2026-03-15",
        }
        mock_baseline = {
            "sst": {"mean": 27.5, "std": 0.5},
            "anomaly": {"mean": 0.0, "std": 0.3},
        }

        with patch.object(monitor, 'fetch_sst_data', return_value=mock_data):
            with patch.object(monitor, 'calculate_baseline', return_value=mock_baseline):
                signal = monitor.generate_signal("nino34", "2026-03-15")

        assert signal is not None
        assert "direction" in signal
        assert "confidence" in signal
        assert signal["direction"] in ["long", "short", "neutral"]

    def test_generate_signal_saves_file(self, monitor, tmp_path):
        """Test that signal generation saves output file."""
        # Mock fetch_sst_data and calculate_baseline to avoid slow operations
        mock_data = {
            "sst_celsius": 28.0,
            "sst_anomaly": 0.0,
            "baseline_sst": 27.5,
            "enso_state": "neutral",
            "quality": "good",
            "impact": "neutral",
            "region_id": "nino34",
            "region_name": "Niño 3.4 Region",
            "region_type": "enso_indicator",
            "ocean": "Pacific",
            "date": "2026-03-15",
        }
        mock_baseline = {
            "sst": {"mean": 27.5, "std": 0.5},
            "anomaly": {"mean": 0.0, "std": 0.3},
        }

        with patch.object(monitor, 'fetch_sst_data', return_value=mock_data):
            with patch.object(monitor, 'calculate_baseline', return_value=mock_baseline):
                signal = monitor.generate_signal("nino34", "2026-03-15")

        signal_file = tmp_path / "sea_surface_temperature" / "signal_nino34_2026-03-15.json"
        assert signal_file.exists()

        # Verify file content
        saved = json.loads(signal_file.read_text())
        assert saved["region_id"] == "nino34"

    def test_generate_all_signals(self, monitor):
        """Test generating signals for all regions."""
        # Mock generate_signal to avoid slow operations
        mock_signal = {
            "region_id": "nino34",
            "region_name": "Niño 3.4 Region",
            "region_type": "enso_indicator",
            "direction": "neutral",
            "confidence": 50.0,
            "enso_state": "neutral",
        }

        with patch.object(monitor, 'generate_signal', return_value=mock_signal):
            signals = monitor.generate_all_signals("2026-03-15")

        assert isinstance(signals, list)
        assert len(signals) <= len(monitor.regions)  # Some may fail


class TestSSTRegions:
    """Tests for region configurations."""

    @pytest.fixture
    def monitor(self, tmp_path):
        from pipeline.sea_surface_temperature import SeaSurfaceTemperatureMonitor
        return SeaSurfaceTemperatureMonitor(output_base=str(tmp_path))

    def test_all_regions_have_valid_bboxes(self, monitor):
        """Test that all region bboxes are valid."""
        for region_id, region in monitor.regions.items():
            bbox = region["bbox"]
            assert len(bbox) == 4, f"{region_id} bbox should have 4 values"
            assert bbox[0] < bbox[2], f"{region_id} min_lon < max_lon"
            assert bbox[1] < bbox[3], f"{region_id} min_lat < max_lat"

    def test_all_regions_have_valid_baseline_sst(self, monitor):
        """Test that baseline SST values are valid."""
        for region_id, region in monitor.regions.items():
            baseline = region["baseline_sst"]
            assert 10 < baseline < 35, f"{region_id} baseline SST should be reasonable"

    def test_enso_regions_have_threshold(self, monitor):
        """Test that ENSO regions have threshold defined."""
        enso_regions = ["nino34", "nino3", "nino4"]

        for region_id in enso_regions:
            if region_id in monitor.regions:
                region = monitor.regions[region_id]
                assert "enso_threshold" in region, f"{region_id} should have ENSO threshold"

    def test_all_regions_have_instruments(self, monitor):
        """Test that all regions have trading instruments."""
        for region_id, region in monitor.regions.items():
            instruments = region["instruments"]
            assert len(instruments) > 0, f"{region_id} should have at least one instrument"
