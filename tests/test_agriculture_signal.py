from unittest.mock import patch


def test_build_agriculture_signals_long_with_real_data(tmp_path):
    from pipeline.agriculture_signal import build_agriculture_signals

    veg_signal = {
        "confidence": 82.0,
        "is_real_data": True,
        "is_critical_season": True,
        "status": "stress",
        "ndvi_anomaly_pct": -14.0,
    }
    precip_signal = {
        "confidence": 78.0,
        "is_real_data": True,
        "is_critical_season": True,
        "status": "drought",
        "precip_anomaly_pct": -24.0,
    }

    with patch("pipeline.agriculture_signal.VegetationHealthMonitor.generate_signal", return_value=veg_signal):
        with patch("pipeline.agriculture_signal.PrecipitationMonitor.generate_signal", return_value=precip_signal):
            signals = build_agriculture_signals("2026-07-15", output_base=str(tmp_path))

    assert signals["agriculture_us_corn_soy"]["trading_action"] == "LONG"
    assert signals["agriculture_us_corn_soy"]["actionability"] == "Actionable"
    assert signals["agriculture_us_corn_soy"]["real_data_ratio"] == 1.0


def test_build_agriculture_signals_ignores_simulated_only(tmp_path):
    from pipeline.agriculture_signal import build_agriculture_signals

    veg_signal = {
        "confidence": 52.0,
        "is_real_data": False,
        "is_critical_season": True,
        "status": "stress",
        "ndvi_anomaly_pct": -12.0,
    }
    precip_signal = {
        "confidence": 50.0,
        "is_real_data": False,
        "is_critical_season": True,
        "status": "drought",
        "precip_anomaly_pct": -22.0,
    }

    with patch("pipeline.agriculture_signal.VegetationHealthMonitor.generate_signal", return_value=veg_signal):
        with patch("pipeline.agriculture_signal.PrecipitationMonitor.generate_signal", return_value=precip_signal):
            signals = build_agriculture_signals("2026-07-15", output_base=str(tmp_path))

    assert signals["agriculture_us_corn_soy"]["trading_action"] == "LONG"
    assert signals["agriculture_us_corn_soy"]["actionability"] == "Ignore"
    assert signals["agriculture_us_corn_soy"]["data_quality_mode"] == "simulated"
