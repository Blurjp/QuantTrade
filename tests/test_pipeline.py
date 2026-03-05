"""
Unit tests for QuantTrade pipeline
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date
from shapely.geometry import Point, LineString

from pipeline.manifest import load_aoi
from pipeline.crossings import compute_side_of_gate


def test_load_aoi():
    """Test AOI loading."""
    aoi = load_aoi("configs/aoi_hormuz.geojson")

    assert aoi is not None
    assert "features" in aoi
    assert len(aoi["features"]) > 0


def test_compute_side_of_gate():
    """Test gate side computation."""
    gate_line = LineString([[56.5, 26.9], [56.5, 27.1]])

    point_left = Point(56.4, 27.0)
    point_right = Point(56.6, 27.0)

    side_left = compute_side_of_gate(point_left, gate_line)
    side_right = compute_side_of_gate(point_right, gate_line)

    assert side_left != side_right


def test_distance_computation():
    """Test Haversine distance."""
    from pipeline.tracking import compute_distance

    # Same point
    d = compute_distance(26.9, 56.5, 26.9, 56.5)
    assert d == 0.0

    # Different points (roughly 11km)
    d = compute_distance(26.9, 56.5, 27.0, 56.5)
    assert 10 < d < 12


def test_bias_correction():
    """Test bias correction application."""
    from pipeline.calibration import apply_bias_correction

    metrics_df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "gc_total": [10, 20],
        "gc_in": [5, 10],
        "gc_out": [5, 10],
        "coverage_score": [0.8, 0.9],
        "throughput_index_total": [0.5, 1.0]
    })

    coefficients = {
        "intercept": 0.5,
        "gc_total_coef": 1.2,
        "coverage_coef": 0.3
    }

    corrected = apply_bias_correction(metrics_df, coefficients)

    assert "throughput_index_corrected" in corrected.columns
    assert "bias_factor" in corrected.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
