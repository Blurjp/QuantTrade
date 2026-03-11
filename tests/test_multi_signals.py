import json

import pandas as pd

from pipeline.backtest import generate_historical_signals, optimize_thresholds
from pipeline.signals_multi import generate_signal


def test_agriculture_signal_uses_seasonal_baseline():
    data = pd.DataFrame([
        {"date": "2024-03-01", "ndvi_mean": 0.72},
        {"date": "2024-03-05", "ndvi_mean": 0.71},
        {"date": "2025-03-02", "ndvi_mean": 0.70},
        {"date": "2025-03-06", "ndvi_mean": 0.69},
        {"date": "2026-03-03", "ndvi_mean": 0.60},
    ])

    signal = generate_signal("agriculture", data)

    assert signal["trading_action"] == "LONG"
    assert signal["signal"] == "Long crop (supply concerns)"
    assert signal["baseline_samples"] >= 3
    assert signal["ndvi_change"] < -0.05


def test_generate_historical_signals_supports_agriculture_alias():
    detection_data = {
        "type": "agriculture",
        "weekly_stats": [
            {"date": "2024-03-01", "ndvi_mean": 0.72},
            {"date": "2024-03-08", "ndvi_mean": 0.71},
            {"date": "2025-03-01", "ndvi_mean": 0.70},
            {"date": "2025-03-08", "ndvi_mean": 0.69},
            {"date": "2026-03-01", "ndvi_mean": 0.60},
        ],
    }

    signals = generate_historical_signals(detection_data, "agriculture")

    assert signals.iloc[-1]["signal_direction"] == "long"
    assert signals.iloc[-1]["signal_raw"] < -0.03


def test_optimize_thresholds_applies_thresholds(tmp_path, monkeypatch):
    output_base = tmp_path / "outputs"
    backfill_dir = output_base / "backfill"
    backfill_dir.mkdir(parents=True)

    payload = {
        "region": "brazil_soy",
        "type": "agriculture",
        "weekly_stats": [
            {"date": "2024-03-01", "ndvi_mean": 0.72},
            {"date": "2024-03-08", "ndvi_mean": 0.71},
            {"date": "2025-03-01", "ndvi_mean": 0.70},
            {"date": "2025-03-08", "ndvi_mean": 0.69},
            {"date": "2026-03-01", "ndvi_mean": 0.67},
        ],
    }
    (backfill_dir / "brazil_soy_backfill.json").write_text(json.dumps(payload))

    prices = pd.DataFrame([
        {"Date": "2024-03-01", "Open": 10, "High": 10, "Low": 10, "Close": 100, "Volume": 1},
        {"Date": "2024-03-08", "Open": 10, "High": 10, "Low": 10, "Close": 101, "Volume": 1},
        {"Date": "2025-03-01", "Open": 10, "High": 10, "Low": 10, "Close": 102, "Volume": 1},
        {"Date": "2025-03-08", "Open": 10, "High": 10, "Low": 10, "Close": 103, "Volume": 1},
        {"Date": "2026-03-01", "Open": 10, "High": 10, "Low": 10, "Close": 110, "Volume": 1},
        {"Date": "2026-03-06", "Open": 10, "High": 10, "Low": 10, "Close": 120, "Volume": 1},
        {"Date": "2026-03-10", "Open": 10, "High": 10, "Low": 10, "Close": 121, "Volume": 1},
    ])

    monkeypatch.setattr("pipeline.backtest.fetch_historical_prices", lambda *args, **kwargs: prices.copy())
    monkeypatch.setattr(
        "pipeline.backtest.backtest_signals",
        lambda signals, prices, forward_days: {
            "overall_accuracy": float((signals["signal_direction"] != "neutral").sum())
        },
    )

    result = optimize_thresholds(
        "brazil_soy",
        "Soybeans",
        output_base=str(output_base),
        threshold_range=[0.02, 0.05],
    )

    accuracies = {entry["threshold"]: entry["accuracy"] for entry in result["all_results"]}

    assert accuracies[0.02] != accuracies[0.05]
    assert result["optimal_threshold"] in accuracies
