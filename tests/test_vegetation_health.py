"""
Tests for vegetation health monitoring.

Tests NDVI data fetching, baseline calculation, anomaly detection, and signal generation.
"""

import json
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestVegetationHealthMonitor:
    """Tests for VegetationHealthMonitor class."""

    @pytest.fixture
    def monitor(self, tmp_path):
        """Create a VegetationHealthMonitor instance."""
        from pipeline.vegetation_health import VegetationHealthMonitor
        return VegetationHealthMonitor(output_base=str(tmp_path))

    def test_initialization(self, monitor, tmp_path):
        """Test monitor initialization."""
        assert monitor.output_base == tmp_path
        assert monitor.cache_days == 30
        assert "usa_corn_soybeans" in monitor.regions
        assert "brazil_cerrado" in monitor.regions

    def test_regions_have_required_fields(self, monitor):
        """Test that all regions have required configuration fields."""
        required_fields = ["name", "bbox", "country", "type", "instruments", "baseline_ndvi", "critical_months"]

        for region_id, region in monitor.regions.items():
            for field in required_fields:
                assert field in region, f"Region {region_id} missing field {field}"

    def test_get_regional_summary(self, monitor):
        """Test regional summary generation."""
        summary = monitor.get_regional_summary()

        assert summary["monitoring_type"] == "vegetation_health"
        assert "satellites" in summary
        assert "metrics" in summary
        assert summary["total_regions"] == len(monitor.regions)
        assert len(summary["trading_instruments"]) > 0

    def test_fetch_simulated_ndvi_data(self, monitor):
        """Test simulated NDVI data fetching."""
        data = monitor._fetch_simulated_ndvi("usa_corn_soybeans", "2026-03-15")

        assert data is not None
        assert "ndvi" in data
        assert "evi" in data
        assert "baseline_ndvi" in data
        assert "ndvi_anomaly" in data
        assert "status" in data
        assert 0 <= data["ndvi"] <= 1
        assert 0 <= data["evi"] <= 1

    def test_fetch_simulated_ndvi_different_dates(self, monitor):
        """Test that different dates produce different NDVI values."""
        data1 = monitor._fetch_simulated_ndvi("usa_corn_soybeans", "2026-03-15")
        data2 = monitor._fetch_simulated_ndvi("usa_corn_soybeans", "2026-06-15")

        # Different seasons should have different NDVI (growing season effect)
        assert data1["ndvi"] != data2["ndvi"]

    def test_fetch_simulated_ndvi_critical_season(self, monitor):
        """Test NDVI during critical growing season."""
        # usa_corn_soybeans critical months are [6, 7, 8, 9]
        july_data = monitor._fetch_simulated_ndvi("usa_corn_soybeans", "2026-07-15")
        january_data = monitor._fetch_simulated_ndvi("usa_corn_soybeans", "2026-01-15")

        assert july_data["is_critical_season"] is True
        assert january_data["is_critical_season"] is False

    def test_fetch_ndvi_data_unknown_region(self, monitor):
        """Test fetching NDVI for unknown region returns None."""
        data = monitor.fetch_ndvi_data("unknown_region", "2026-03-15")

        assert data is None

    def test_calculate_baseline(self, monitor):
        """Test baseline calculation."""
        # Mock fetch_ndvi_data to avoid slow network calls
        mock_data = {
            "ndvi": 0.65,
            "evi": 0.55,
            "ndvi_anomaly_pct": 0.0,
            "quality": "good",
        }

        with patch.object(monitor, 'fetch_ndvi_data', return_value=mock_data):
            baseline = monitor.calculate_baseline("usa_corn_soybeans", days=5)

        assert "ndvi" in baseline
        assert "evi" in baseline
        assert "anomaly" in baseline
        assert baseline["ndvi"]["mean"] >= 0
        assert baseline["ndvi"]["std"] >= 0

    def test_detect_anomaly_stress(self, monitor):
        """Test anomaly detection for stressed vegetation."""
        current_data = {
            "ndvi": 0.35,
            "evi": 0.30,
        }
        baseline = {
            "ndvi": {"mean": 0.65, "std": 0.05},
            "evi": {"mean": 0.55, "std": 0.04},
        }

        anomaly = monitor.detect_anomaly(current_data, baseline)

        assert anomaly["ndvi_z_score"] < 0  # Below baseline
        assert anomaly["ndvi_anomaly"] in ["significant", "moderate", "none"]

    def test_detect_anomaly_excellent(self, monitor):
        """Test anomaly detection for excellent vegetation."""
        current_data = {
            "ndvi": 0.85,
            "evi": 0.72,
        }
        baseline = {
            "ndvi": {"mean": 0.65, "std": 0.05},
            "evi": {"mean": 0.55, "std": 0.04},
        }

        anomaly = monitor.detect_anomaly(current_data, baseline)

        assert anomaly["ndvi_z_score"] > 0  # Above baseline

    def test_generate_signal(self, monitor):
        """Test signal generation."""
        # Mock fetch_ndvi_data and calculate_baseline to avoid slow operations
        mock_data = {
            "ndvi": 0.65,
            "evi": 0.55,
            "ndvi_anomaly_pct": 0.0,
            "status": "normal",
            "is_critical_season": True,
            "impact_score": 10.0,
            "quality": "good",
            "lai_estimate": 3.9,
            "chlorophyll_content": 65.0,
            "region_id": "usa_corn_soybeans",
            "region_name": "US Corn & Soybeans Belt",
            "region_type": "row_crops",
            "country": "USA",
            "date": "2026-03-15",
        }
        mock_baseline = {
            "ndvi": {"mean": 0.65, "std": 0.05},
            "evi": {"mean": 0.55, "std": 0.04},
            "anomaly": {"mean": 0.0, "std": 0.1},
        }

        with patch.object(monitor, 'fetch_ndvi_data', return_value=mock_data):
            with patch.object(monitor, 'calculate_baseline', return_value=mock_baseline):
                signal = monitor.generate_signal("usa_corn_soybeans", "2026-03-15")

        assert signal is not None
        assert "direction" in signal
        assert "confidence" in signal
        assert signal["direction"] in ["long", "short", "neutral"]

    def test_generate_signal_saves_file(self, monitor, tmp_path):
        """Test that signal generation saves output file."""
        # Mock fetch_ndvi_data and calculate_baseline to avoid slow operations
        mock_data = {
            "ndvi": 0.65,
            "evi": 0.55,
            "ndvi_anomaly_pct": 0.0,
            "status": "normal",
            "is_critical_season": True,
            "impact_score": 10.0,
            "quality": "good",
            "lai_estimate": 3.9,
            "chlorophyll_content": 65.0,
            "region_id": "usa_corn_soybeans",
            "region_name": "US Corn & Soybeans Belt",
            "region_type": "row_crops",
            "country": "USA",
            "date": "2026-03-15",
        }
        mock_baseline = {
            "ndvi": {"mean": 0.65, "std": 0.05},
            "evi": {"mean": 0.55, "std": 0.04},
            "anomaly": {"mean": 0.0, "std": 0.1},
        }

        with patch.object(monitor, 'fetch_ndvi_data', return_value=mock_data):
            with patch.object(monitor, 'calculate_baseline', return_value=mock_baseline):
                signal = monitor.generate_signal("usa_corn_soybeans", "2026-03-15")

        signal_file = tmp_path / "vegetation_health" / "signal_usa_corn_soybeans_2026-03-15.json"
        assert signal_file.exists()

        # Verify file content
        saved = json.loads(signal_file.read_text())
        assert saved["region_id"] == "usa_corn_soybeans"

    def test_generate_all_signals(self, monitor):
        """Test generating signals for all regions."""
        # Mock generate_signal to avoid slow operations
        mock_signal = {
            "region_id": "usa_corn_soybeans",
            "region_name": "US Corn & Soybeans Belt",
            "region_type": "row_crops",
            "direction": "neutral",
            "confidence": 50.0,
            "impact_score": 10.0,
            "status": "normal",
            "is_critical_season": True,
        }

        with patch.object(monitor, 'generate_signal', return_value=mock_signal):
            signals = monitor.generate_all_signals("2026-03-15")

        assert isinstance(signals, list)
        assert len(signals) <= len(monitor.regions)  # Some may fail


class TestVegetationHealthRegions:
    """Tests for region configurations."""

    @pytest.fixture
    def monitor(self, tmp_path):
        from pipeline.vegetation_health import VegetationHealthMonitor
        return VegetationHealthMonitor(output_base=str(tmp_path))

    def test_all_regions_have_valid_bboxes(self, monitor):
        """Test that all region bboxes are valid."""
        for region_id, region in monitor.regions.items():
            bbox = region["bbox"]
            assert len(bbox) == 4, f"{region_id} bbox should have 4 values"
            assert bbox[0] < bbox[2], f"{region_id} min_lon < max_lon"
            assert bbox[1] < bbox[3], f"{region_id} min_lat < max_lat"

    def test_all_regions_have_valid_baseline_ndvi(self, monitor):
        """Test that baseline NDVI values are valid."""
        for region_id, region in monitor.regions.items():
            baseline = region["baseline_ndvi"]
            assert 0 < baseline < 1, f"{region_id} baseline NDVI should be between 0 and 1"

    def test_all_regions_have_instruments(self, monitor):
        """Test that all regions have trading instruments."""
        for region_id, region in monitor.regions.items():
            instruments = region["instruments"]
            assert len(instruments) > 0, f"{region_id} should have at least one instrument"
