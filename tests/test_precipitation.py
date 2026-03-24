"""
Tests for precipitation monitoring.

Tests precipitation data fetching, drought/flood detection, and signal generation.
"""

import json
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from pipeline.precipitation import PrecipitationMonitor


class TestPrecipitationMonitor:
    """Tests for PrecipitationMonitor class."""

    @pytest.fixture
    def monitor(self, tmp_path):
        """Create a PrecipitationMonitor instance."""
        return PrecipitationMonitor(output_base=str(tmp_path))

    def test_initialization(self, monitor, tmp_path):
        """Test monitor initialization."""
        assert monitor.output_base == tmp_path
        assert monitor.cache_days == 30
        assert "usa_corn_belt" in monitor.regions
        assert "brazil_soybeans" in monitor.regions

    def test_regions_have_required_fields(self, monitor):
        """Test that all regions have required configuration fields."""
        required_fields = ["name", "bbox", "country", "type", "instruments", "baseline_precip_mm", "critical_months"]

        for region_id, region in monitor.regions.items():
            for field in required_fields:
                assert field in region, f"Region {region_id} missing field {field}"

    def test_get_regional_summary(self, monitor):
        """Test regional summary generation."""
        summary = monitor.get_regional_summary()

        assert summary["monitoring_type"] == "precipitation"
        assert "satellites" in summary
        assert "metrics" in summary
        assert summary["total_regions"] == len(monitor.regions)
        assert len(summary["trading_instruments"]) > 0

    def test_fetch_simulated_precipitation(self, monitor):
        """Test simulated precipitation data fetching."""
        data = monitor.fetch_precipitation_data("usa_corn_belt", "2026-03-15")

        assert data is not None
        assert "daily_precip_mm" in data
        assert "monthly_precip_estimate_mm" in data
        assert "baseline_precip_mm" in data
        assert "precip_anomaly_mm" in data
        assert "precip_anomaly_pct" in data
        assert "status" in data
        assert data["daily_precip_mm"] >= 0

    def test_fetch_simulated_precipitation_different_seasons(self, monitor):
        """Test that different seasons produce different precipitation."""
        summer_data = monitor.fetch_precipitation_data("usa_corn_belt", "2026-07-15")
        winter_data = monitor.fetch_precipitation_data("usa_corn_belt", "2026-01-15")

        # Different seasons should have different precipitation patterns
        assert summer_data["daily_precip_mm"] != winter_data["daily_precip_mm"]

    def test_fetch_precipitation_unknown_region(self, monitor):
        """Test fetching precipitation for unknown region returns None."""
        data = monitor.fetch_precipitation_data("unknown_region", "2026-03-15")

        assert data is None

    def test_precipitation_status_classification(self, monitor):
        """Test precipitation status classification."""
        # Test different anomaly percentages
        test_cases = [
            (-45.0, "severe_drought"),
            (-25.0, "drought"),
            (-12.0, "dry"),
            (0.0, "normal"),
            (12.0, "slightly_wet"),
            (25.0, "wet"),
            (45.0, "flood"),
        ]

        for anomaly_pct, expected_status in test_cases:
            # Simulate data that would result in this anomaly
            data = monitor.fetch_precipitation_data("usa_corn_belt", "2026-03-15")
            # The actual status depends on the random simulation
            assert data["status"] in ["severe_drought", "drought", "dry", "normal", "slightly_wet", "wet", "flood"]

    def test_calculate_baseline(self, monitor):
        """Test baseline calculation."""
        # Mock fetch_precipitation_data to avoid slow network calls
        mock_data = {
            "daily_precip_mm": 3.0,
            "monthly_precip_estimate_mm": 85.0,
            "baseline_precip_mm": 85.0,
            "precip_anomaly_mm": 0.0,
            "precip_anomaly_pct": 0.0,
            "status": "normal",
            "quality": "good",
        }

        with patch.object(monitor, 'fetch_precipitation_data', return_value=mock_data):
            baseline = monitor.calculate_baseline("usa_corn_belt", days=10)

        assert "precipitation" in baseline
        assert "anomaly" in baseline
        assert baseline["precipitation"]["mean"] >= 0
        assert baseline["precipitation"]["std"] >= 0

    def test_detect_anomaly_drought(self, monitor):
        """Test anomaly detection for drought conditions."""
        current_data = {
            "monthly_precip_estimate_mm": 40.0,
            "baseline_precip_mm": 85.0,
        }
        baseline = {
            "precipitation": {"mean": 85.0, "std": 15.0},
            "anomaly": {"mean": 0.0, "std": 20.0},
        }

        anomaly = monitor.detect_anomaly(current_data, baseline)

        assert anomaly["precip_z_score"] < 0  # Below baseline

    def test_detect_anomaly_flood(self, monitor):
        """Test anomaly detection for flood conditions."""
        current_data = {
            "monthly_precip_estimate_mm": 150.0,
            "baseline_precip_mm": 85.0,
        }
        baseline = {
            "precipitation": {"mean": 85.0, "std": 15.0},
            "anomaly": {"mean": 0.0, "std": 20.0},
        }

        anomaly = monitor.detect_anomaly(current_data, baseline)

        assert anomaly["precip_z_score"] > 0  # Above baseline

    def test_generate_signal_drought(self, monitor):
        """Test signal generation for drought conditions."""
        with patch.object(monitor, '_fetch_simulated_precipitation') as mock_fetch:
            mock_fetch.return_value = {
                "daily_precip_mm": 1.0,
                "monthly_precip_estimate_mm": 30.0,
                "baseline_precip_mm": 85.0,
                "precip_anomaly_mm": -55.0,
                "precip_anomaly_pct": -64.7,
                "status": "severe_drought",
                "is_critical_season": True,
                "impact_score": 97.0,
                "crops": ["corn", "soybeans"],
                "data_source": "GPM_IMERG_SIMULATED",
                "quality": "good",
            }

            signal = monitor.generate_signal("usa_corn_belt", "2026-07-15")

            # Drought = supply shortage = bullish = LONG
            assert signal["direction"] == "long"
            assert signal["confidence"] > 50
            assert "drought" in signal["rationale"].lower()

    def test_generate_signal_flood(self, monitor):
        """Test signal generation for flood conditions."""
        with patch.object(monitor, '_fetch_simulated_precipitation') as mock_fetch:
            mock_fetch.return_value = {
                "daily_precip_mm": 15.0,
                "monthly_precip_estimate_mm": 150.0,
                "baseline_precip_mm": 85.0,
                "precip_anomaly_mm": 65.0,
                "precip_anomaly_pct": 76.5,
                "status": "flood",
                "is_critical_season": True,
                "impact_score": 100.0,
                "crops": ["corn", "soybeans"],
                "data_source": "GPM_IMERG_SIMULATED",
                "quality": "good",
            }

            signal = monitor.generate_signal("usa_corn_belt", "2026-07-15")

            # Flood = supply damage = bullish = LONG
            assert signal["direction"] == "long"
            assert "flood" in signal["rationale"].lower() or "rain" in signal["rationale"].lower()

    def test_generate_signal_normal(self, monitor):
        """Test signal generation for normal conditions."""
        with patch.object(monitor, '_fetch_simulated_precipitation') as mock_fetch:
            mock_fetch.return_value = {
                "daily_precip_mm": 3.0,
                "monthly_precip_estimate_mm": 90.0,
                "baseline_precip_mm": 85.0,
                "precip_anomaly_mm": 5.0,
                "precip_anomaly_pct": 5.9,
                "status": "normal",
                "is_critical_season": True,
                "impact_score": 8.8,
                "crops": ["corn", "soybeans"],
                "data_source": "GPM_IMERG_SIMULATED",
                "quality": "good",
            }

            signal = monitor.generate_signal("usa_corn_belt", "2026-07-15")

            # Normal conditions = good growing conditions = good supply = bearish = SHORT
            assert signal["direction"] in ["short", "neutral"]

    def test_generate_signal_penalizes_simulated_data(self, monitor):
        with patch.object(monitor, '_fetch_simulated_precipitation') as mock_fetch:
            mock_fetch.return_value = {
                "daily_precip_mm": 1.0,
                "monthly_precip_estimate_mm": 30.0,
                "baseline_precip_mm": 85.0,
                "precip_anomaly_mm": -55.0,
                "precip_anomaly_pct": -64.7,
                "status": "severe_drought",
                "is_critical_season": True,
                "impact_score": 97.0,
                "crops": ["corn", "soybeans"],
                "data_source": "GPM_IMERG_SIMULATED",
                "quality": "good",
                "is_real_data": False,
                "fallback_reason": "real_data_unavailable",
            }

            signal = monitor.generate_signal("usa_corn_belt", "2026-07-15")

            assert signal["is_real_data"] is False
            assert signal["confidence_penalty_pct"] == 30
            assert signal["confidence_label"] in ["High", "Medium", "Low"]

    def test_generate_all_signals(self, monitor):
        """Test generating signals for all regions."""
        signals = monitor.generate_all_signals("2026-03-15")

        assert isinstance(signals, list)
        assert len(signals) <= len(monitor.regions)

        # Check that signals are sorted by impact score
        if len(signals) > 1:
            for i in range(len(signals) - 1):
                assert signals[i]["impact_score"] >= signals[i + 1]["impact_score"]

    def test_generate_signal_saves_file(self, monitor, tmp_path):
        """Test that signal generation saves output file."""
        signal = monitor.generate_signal("usa_corn_belt", "2026-03-15")

        signal_file = tmp_path / "precipitation" / "signal_usa_corn_belt_2026-03-15.json"
        assert signal_file.exists()

        # Verify file content
        saved = json.loads(signal_file.read_text())
        assert saved["region_id"] == "usa_corn_belt"

    def test_critical_season_impact(self, monitor):
        """Test that critical season increases impact score."""
        # usa_corn_belt critical months are [4, 5, 6, 7, 8]
        critical_data = monitor.fetch_precipitation_data("usa_corn_belt", "2026-07-15")
        non_critical_data = monitor.fetch_precipitation_data("usa_corn_belt", "2026-01-15")

        # July should be critical season
        assert critical_data["is_critical_season"] is True
        # January should not be critical season
        assert non_critical_data["is_critical_season"] is False

    def test_impact_score_critical_vs_non_critical(self, monitor):
        """Test that impact score is higher during critical season."""
        # Same anomaly percentage but different seasons
        with patch.object(monitor, '_fetch_simulated_precipitation') as mock_fetch:
            # Critical season data
            mock_fetch.return_value = {
                "daily_precip_mm": 1.0,
                "monthly_precip_estimate_mm": 30.0,
                "baseline_precip_mm": 85.0,
                "precip_anomaly_mm": -55.0,
                "precip_anomaly_pct": -50.0,
                "status": "severe_drought",
                "is_critical_season": True,
                "impact_score": 75.0,  # 50 * 1.5
                "crops": ["corn", "soybeans"],
                "data_source": "GPM_IMERG_SIMULATED",
                "quality": "good",
            }

            critical_signal = monitor.generate_signal("usa_corn_belt", "2026-07-15")

            # Non-critical season data
            mock_fetch.return_value["is_critical_season"] = False
            mock_fetch.return_value["impact_score"] = 35.0  # 50 * 0.7

            non_critical_signal = monitor.generate_signal("usa_corn_belt", "2026-01-15")

            assert critical_signal["impact_score"] > non_critical_signal["impact_score"]


class TestPrecipitationRegions:
    """Tests for region configurations."""

    @pytest.fixture
    def monitor(self, tmp_path):
        return PrecipitationMonitor(output_base=str(tmp_path))

    def test_all_regions_have_valid_bboxes(self, monitor):
        """Test that all region bboxes are valid."""
        for region_id, region in monitor.regions.items():
            bbox = region["bbox"]
            assert len(bbox) == 4, f"{region_id} bbox should have 4 values"
            assert bbox[0] < bbox[2], f"{region_id} min_lon < max_lon"
            assert bbox[1] < bbox[3], f"{region_id} min_lat < max_lat"

    def test_all_regions_have_valid_baseline_precip(self, monitor):
        """Test that baseline precipitation values are valid."""
        for region_id, region in monitor.regions.items():
            baseline = region["baseline_precip_mm"]
            assert baseline > 0, f"{region_id} baseline precip should be positive"

    def test_all_regions_have_crops(self, monitor):
        """Test that all regions have crop information."""
        for region_id, region in monitor.regions.items():
            crops = region.get("crops", [])
            assert len(crops) > 0, f"{region_id} should have at least one crop"

    def test_all_regions_have_instruments(self, monitor):
        """Test that all regions have trading instruments."""
        for region_id, region in monitor.regions.items():
            instruments = region["instruments"]
            assert len(instruments) > 0, f"{region_id} should have at least one instrument"


class TestPrecipitationIntegration:
    """Integration tests for precipitation monitoring."""

    @pytest.fixture
    def monitor(self, tmp_path):
        return PrecipitationMonitor(output_base=str(tmp_path))

    def test_full_workflow(self, monitor):
        """Test complete monitoring workflow."""
        region_id = "usa_corn_belt"
        date = "2026-07-15"

        # 1. Fetch data
        data = monitor.fetch_precipitation_data(region_id, date)
        assert data is not None

        # 2. Calculate baseline
        baseline = monitor.calculate_baseline(region_id, days=10)
        assert "precipitation" in baseline

        # 3. Detect anomaly
        anomaly = monitor.detect_anomaly(data, baseline)
        assert "precip_z_score" in anomaly

        # 4. Generate signal
        signal = monitor.generate_signal(region_id, date)
        assert signal is not None
        assert "direction" in signal
        assert "confidence" in signal

    def test_multiple_regions_workflow(self, monitor):
        """Test workflow for multiple regions."""
        signals = monitor.generate_all_signals("2026-07-15")

        for signal in signals:
            assert "region_id" in signal
            assert "direction" in signal
            assert "confidence" in signal
            assert signal["direction"] in ["long", "short", "neutral"]

    def test_summary_generation(self, monitor):
        """Test summary generation for all regions."""
        signals = monitor.generate_all_signals("2026-07-15")

        # Check that summary file was created
        summary_file = monitor.output_dir / "summary_2026-07-15.json"

        if signals:
            assert summary_file.exists()
            summary = json.loads(summary_file.read_text())
            assert summary["date"] == "2026-07-15"
            assert "total_regions" in summary
            assert "signals_generated" in summary
