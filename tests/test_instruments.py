"""
Tests for instrument mapping utilities.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

from pipeline.instruments import (
    load_instrument_registry,
    list_region_instruments,
    get_primary_instrument,
    INSTRUMENTS_PATH,
)


class TestLoadInstrumentRegistry:
    """Tests for load_instrument_registry function."""

    
    def test_load_default_path(self, tmp_path):
        """Test loading from default path."""
        instruments_file = tmp_path / "configs" / "instruments.json"
        instruments_file.parent.mkdir(parents=True, exist_ok=True)
        
        test_data = {
            "regions": {
                "usa_midwest": [
                    {"symbol": "CORN", "name": "Corn Futures", "primary": True},
                    {"symbol": "SOYB", "name": "Soybean ETF", "primary": False}
                ]
            }
        }
        instruments_file.write_text(json.dumps(test_data))
        
        result = load_instrument_registry(instruments_file)
        
        assert "regions" in result
        assert "usa_midwest" in result["regions"]
        assert len(result["regions"]["usa_midwest"]) == 2
        assert result["regions"]["usa_midwest"][0]["symbol"] == "CORN"

    
    def test_load_custom_path(self, tmp_path):
        """Test loading from custom path."""
        custom_file = tmp_path / "custom_instruments.json"
        custom_file.write_text(json.dumps({"regions": {"custom": []}}))
        
        result = load_instrument_registry(custom_file)
        
        assert "custom" in result["regions"]
    
    def test_file_not_found(self):
        """Test handling of missing file."""
        with patch("builtins.open", mock_open(read_data='{"regions": {}}')) as mock_file:
            result = load_instrument_registry(Path("/nonexistent.json"))
            assert result == {"regions": {}}


class TestListRegionInstruments:
    """Tests for list_region_instruments function."""
    
    @pytest.fixture
    def registry_data(self, tmp_path):
        """Create test registry file."""
        instruments_file = tmp_path / "configs" / "instruments.json"
        instruments_file.parent.mkdir(parents=True, exist_ok=True)
        
        test_data = {
            "regions": {
                "brazil_soybean": [
                    {"symbol": "SOYB", "name": "Soybean ETF", "enabled_for_backtest": True, "enabled_for_alerts": True, "primary": True},
                    {"symbol": "WEAT", "name": "Wheat ETF", "enabled_for_backtest": False, "enabled_for_alerts": True}
                ],
                "usa_corn": [
                    {"symbol": "CORN", "name": "Corn Futures", "enabled_for_backtest": True, "enabled_for_alerts": False}
                ]
            }
        }
        instruments_file.write_text(json.dumps(test_data))
        
        with patch("pipeline.instruments.INSTRUMENTS_PATH", instruments_file):
            yield instruments_file
    
    def test_list_all_instruments(self, registry_data):
        """Test listing all instruments for a region."""
        instruments = list_region_instruments("brazil_soybean")
        
        assert len(instruments) == 2
        symbols = [i["symbol"] for i in instruments]
        assert "SOYB" in symbols
        assert "WEAT" in symbols
    
    def test_filter_by_backtest(self, registry_data):
        """Test filtering instruments by backtest status."""
        instruments = list_region_instruments("brazil_soybean", enabled_for_backtest=True)
        
        assert len(instruments) == 1
        assert instruments[0]["symbol"] == "SOYB"
    
    def test_filter_by_alerts(self, registry_data):
        """Test filtering instruments by alerts status."""
        instruments = list_region_instruments("brazil_soybean", enabled_for_alerts=True)
        
        assert len(instruments) == 2
    
    def test_unknown_region(self, registry_data):
        """Test listing instruments for unknown region."""
        instruments = list_region_instruments("unknown_region")
        
        assert instruments == []
    
    def test_empty_region(self, tmp_path):
        """Test listing instruments for region with no instruments."""
        instruments_file = tmp_path / "configs" / "instruments.json"
        instruments_file.parent.mkdir(parents=True, exist_ok=True)
        instruments_file.write_text(json.dumps({"regions": {"empty": []}}))
        
        with patch("pipeline.instruments.INSTRUMENTS_PATH", instruments_file):
            instruments = list_region_instruments("empty")
            assert instruments == []


class TestGetPrimaryInstrument:
    """Tests for get_primary_instrument function."""
    
    @pytest.fixture
    def registry_data(self, tmp_path):
        """Create test registry file."""
        instruments_file = tmp_path / "configs" / "instruments.json"
        instruments_file.parent.mkdir(parents=True, exist_ok=True)
        
        test_data = {
            "regions": {
                "region_with_primary": [
                    {"symbol": "PRIMARY", "name": "Primary Instrument", "primary": True},
                    {"symbol": "SECONDARY", "name": "Secondary Instrument", "primary": False}
                ],
                "region_without_primary": [
                    {"symbol": "FIRST", "name": "First Instrument"},
                    {"symbol": "SECOND", "name": "Second Instrument"}
                ],
                "empty_region": []
            }
        }
        instruments_file.write_text(json.dumps(test_data))
        
        with patch("pipeline.instruments.INSTRUMENTS_PATH", instruments_file):
            yield instruments_file
    
    def test_get_primary_exists(self, registry_data):
        """Test getting primary instrument when one is marked."""
        primary = get_primary_instrument("region_with_primary")
        
        assert primary is not None
        assert primary["symbol"] == "PRIMARY"
        assert primary["primary"] is True
    
    def test_get_first_as_primary(self, registry_data):
        """Test getting first instrument when no primary is marked."""
        primary = get_primary_instrument("region_without_primary")
        
        assert primary is not None
        assert primary["symbol"] == "FIRST"
    
    def test_empty_region_returns_none(self, registry_data):
        """Test that empty region returns None."""
        primary = get_primary_instrument("empty_region")
        
        assert primary is None
    
    def test_unknown_region_returns_none(self, registry_data):
        """Test that unknown region returns None."""
        primary = get_primary_instrument("unknown_region")
        
        assert primary is None
