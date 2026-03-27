"""
Tests for region registry utilities.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open
import pytest
from pipeline.regions import (
    load_registry,
    load_region_registry,
    list_regions,
    get_region_config,
    get_active_regions,
    resolve_region_output_base,
    REGISTRY_V2_PATH,
    REGISTRY_PATH,
)


class TestLoadRegistry:
    """Tests for load_registry function."""
    
    def test_load_v2_registry(self, tmp_path):
        """Test loading from v2 registry."""
        registry_file = tmp_path / "configs" / "regions" / "registry_v2.json"
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        test_data = {
            "regions": {
                "hormuz": {
                    "name": "Strait of Hormuz",
                    "description": "Key oil chokepoint",
                    "active": True
                },
                "malacca": {
                    "name": "Strait of Malacca",
                    "description": "Key oil transit route",
                    "active": False
                }
            }
        }
        registry_file.write_text(json.dumps(test_data))
        
        with patch("pipeline.regions.REGISTRY_V2_PATH", registry_file):
            with patch("pipeline.regions.REGISTRY_PATH", tmp_path / "registry.json"):
                result = load_registry()
        
        assert "regions" in result
        assert "hormuz" in result["regions"]
        assert result["regions"]["hormuz"]["name"] == "Strait of Hormuz"
    
    def test_load_fallback_registry(self, tmp_path):
        """Test fallback to legacy registry when v2 doesn't exist."""
        registry_file = tmp_path / "configs" / "regions" / "registry.json"
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        test_data = {
            "regions": {
                "malacca": {
                    "name": "Strait of Malacca",
                    "description": "Oil transit"
                    "active": True
                }
            }
        }
        registry_file.write_text(json.dumps(test_data))
        
        with patch("pipeline.regions.REGISTRY_V2_PATH", Path("/nonexistent/v2.json")):
            with patch("pipeline.regions.REGISTRY_PATH", registry_file):
                result = load_registry()
        
        assert "regions" in result
        assert "malacca" in result["regions"]


class TestLoadRegionRegistry:
    """Tests for load_region_registry function."""
    
    def test_load_region_registry_success(self, tmp_path):
        """Test loading region registry."""
        registry_file = tmp_path / "configs" / "regions" / "registry_v2.json"
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        test_data = {
            "regions": {
                "hormuz": {
                    "name": "Strait of Hormuz",
                    "description": "Key chokepoint",
                    "active": True
                }
            }
        }
        registry_file.write_text(json.dumps(test_data))
        
        with patch("pipeline.regions.REGISTRY_V2_PATH", registry_file):
            result = load_region_registry()
        
        assert "hormuz" in result
        assert result["hormuz"]["name"] == "Strait of Hormuz"
    
    def test_load_region_registry_empty(self, tmp_path):
        """Test with empty registry."""
        registry_file = tmp_path / "configs" / "regions" / "registry_v2.json"
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        registry_file.write_text("{}")
        
        with patch("pipeline.regions.REGISTRY_V2_PATH", registry_file):
            result = load_region_registry()
        
        assert result == {}


class TestListRegions:
    """Tests for list_regions function."""
    
    def test_list_regions_success(self, tmp_path):
        """Test listing regions."""
        registry_file = tmp_path / "configs" / "regions" / "registry_v2.json"
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        test_data = {
            "regions": {
                "hormuz": {
                    "name": "Strait of Hormuz",
                    "description": "Key chokepoint"
                    "active": True
                },
                "malacca": {
                    "name": "Strait of Malacca",
                    "description": "Transit route"
                    "active": False
                }
            }
        }
        registry_file.write_text(json.dumps(test_data))
        
        with patch("pipeline.regions.REGISTRY_V2_PATH", registry_file):
            regions = list_regions()
        
        assert len(regions) == 2
        
        region_ids = [r["id"] for r in regions]
        assert "hormuz" in region_ids
        assert "malacca" in region_ids
    
    def test_list_regions_empty(self, tmp_path):
        """Test listing regions with empty registry."""
        registry_file = tmp_path / "configs" / "regions" / "registry_v2.json"
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        registry_file.write_text("{}")
        
        with patch("pipeline.regions.REGISTRY_V2_PATH", registry_file):
            regions = list_regions()
        
        assert regions == []


class TestGetRegionConfig:
    """Tests for get_region_config function."""
    
    def test_get_region_config_success(self, tmp_path):
        """Test getting region config."""
        registry_file = tmp_path / "configs" / "regions" / "registry_v2.json"
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        test_data = {
            "regions": {
                "hormuz": {
                    "name": "Strait of Hormuz",
                    "description": "Key chokepoint",
                    "active": True
                }
            }
        }
        registry_file.write_text(json.dumps(test_data))
        
        with patch("pipeline.regions.REGISTRY_V2_PATH", registry_file):
            config = get_region_config("hormuz")
        
        assert config["id"] == "hormuz"
        assert config["name"] == "Strait of Hormuz"
        assert config["description"] == "Key chokepoint"
    
    def test_get_region_config_unknown(self, tmp_path):
        """Test getting config for unknown region."""
        registry_file = tmp_path / "configs" / "regions" / "registry_v2.json"
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        registry_file.write_text('{"regions": {}}')
        
        with patch("pipeline.regions.REGISTRY_V2_PATH", registry_file):
            with pytest.raises(KeyError):
                get_region_config("unknown_region")
    
    def test_get_region_config_adds_id(self, tmp_path):
        """Test that get_region_config adds id to config."""
        registry_file = tmp_path / "configs" / "regions" / "registry_v2.json"
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        test_data = {
            "regions": {
                "hormuz": {
                    "name": "Strait of Hormuz",
                    "description": "Key chokepoint"
                }
            }
        }
        registry_file.write_text(json.dumps(test_data))
        
        with patch("pipeline.regions.REGISTRY_V2_PATH", registry_file):
            config = get_region_config("hormuz")
        
        assert "id" in config
        assert config["id"] == "hormuz"


class TestGetActiveRegions:
    """Tests for get_active_regions function."""
    
    def test_get_active_regions_filters(self, tmp_path):
        """Test that get_active_regions filters by active status."""
        registry_file = tmp_path / "configs" / "regions" / "registry_v2.json"
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        test_data = {
            "regions": {
                "hormuz": {
                    "name": "Strait of Hormuz",
                    "active": True
                },
                "malacca": {
                    "name": "Strait of Malacca",
                    "active": False
                },
                "suez": {
                    "name": "Suez Canal",
                    "active": True
                }
            }
        }
        registry_file.write_text(json.dumps(test_data))
        
        with patch("pipeline.regions.REGISTRY_V2_PATH", registry_file):
            active = get_active_regions()
        
        assert len(active) == 2
        assert "hormuz" in active
        assert "suez" in active
        assert "malacca" not in active
    
    def test_get_active_regions_empty(self, tmp_path):
        """Test with empty registry."""
        registry_file = tmp_path / "configs" / "regions" / "registry_v2.json"
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        registry_file.write_text("{}")
        
        with patch("pipeline.regions.REGISTRY_V2_PATH", registry_file):
            active = get_active_regions()
        
        assert active == {}


class TestResolveRegionOutputBase:
    """Tests for resolve_region_output_base function."""
    
    def test_resolve_default_hormuz(self):
        """Test default output base for Hormuz (backward compat)."""
        result = resolve_region_output_base("outputs", "hormuz")
        
        assert result == "outputs"
    
    def test_resolve_other_region(self):
        """Test output base for other regions."""
        result = resolve_region_output_base("outputs", "malacca")
        
        assert result == "outputs/regions/malacca"
    
    def test_resolve_custom_base(self):
        """Test with custom base path."""
        result = resolve_region_output_base("/custom", "hormuz")
        
        assert result == "/custom/regions/hormuz"
