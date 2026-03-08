import json
from datetime import date
from pathlib import Path

import pandas as pd

from automation.status import load_region_status
from backtesting.run import run_region_symbol_backtest
from pipeline.instruments import get_primary_instrument, list_region_instruments
from pipeline.signals import build_region_signal_table, latest_region_signal


def _write_region_metrics(base: Path, region_id: str, rows: list[dict]) -> None:
    region_root = base if region_id == "hormuz" else base / "regions" / region_id
    (region_root / "metrics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(region_root / "metrics" / "daily.parquet", index=False)


def test_instrument_registry_has_primary_assets():
    hormuz = get_primary_instrument("hormuz")
    suez = get_primary_instrument("suez_south")
    assert hormuz["ticker"] == "USO"
    assert suez["ticker"] == "ZIM"
    assert len(list_region_instruments("malacca", enabled_for_backtest=True)) == 1


def test_v2_signal_table_sets_confirmation_and_reroute(tmp_path):
    output_base = tmp_path / "outputs"
    _write_region_metrics(
        output_base,
        "hormuz",
        [
            {"date": "2024-01-01", "throughput_index_total": 1.0, "coverage_score": 0.9},
            {"date": "2024-01-02", "throughput_index_total": 1.0, "coverage_score": 0.9},
            {"date": "2024-01-03", "throughput_index_total": 0.2, "coverage_score": 0.9},
            {"date": "2024-01-04", "throughput_index_total": 0.1, "coverage_score": 0.9},
        ],
    )
    _write_region_metrics(
        output_base,
        "bab_el_mandeb",
        [
            {"date": "2024-01-01", "throughput_index_total": 0.2, "coverage_score": 0.9},
            {"date": "2024-01-02", "throughput_index_total": 0.2, "coverage_score": 0.9},
            {"date": "2024-01-03", "throughput_index_total": 0.9, "coverage_score": 0.9},
            {"date": "2024-01-04", "throughput_index_total": 1.0, "coverage_score": 0.9},
        ],
    )

    signal_df = build_region_signal_table("hormuz", output_base=str(output_base), version="v2")
    latest = signal_df.iloc[-1]
    assert latest["confirmation_days"] >= 2
    assert bool(latest["reroute_flag"]) is True
    assert latest["actionability"] == "Watchlist"


def test_run_region_symbol_backtest_writes_artifacts(tmp_path, monkeypatch):
    output_base = tmp_path / "outputs"
    _write_region_metrics(
        output_base,
        "suez_south",
        [
            {"date": "2024-01-01", "throughput_index_total": 1.0, "coverage_score": 0.9},
            {"date": "2024-01-02", "throughput_index_total": 0.1, "coverage_score": 0.9},
            {"date": "2024-01-03", "throughput_index_total": 0.0, "coverage_score": 0.9},
            {"date": "2024-01-04", "throughput_index_total": 1.0, "coverage_score": 0.9},
        ],
    )

    price_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "Open": [10.0, 10.5, 10.8, 10.7],
            "Close": [10.2, 10.7, 10.6, 11.0],
            "Volume": [1, 1, 1, 1],
        }
    )
    monkeypatch.setattr("backtesting.run.fetch_yahoo_prices", lambda *args, **kwargs: price_df.copy())

    summary = run_region_symbol_backtest(
        region_id="suez_south",
        symbol="ZIM",
        output_base=str(output_base),
        version="v2",
    )
    assert summary["symbol"] == "ZIM"
    assert Path(summary["equity_path"]).exists()
    assert Path(summary["summary_path"]).exists()


def test_run_daily_region_dry_run_updates_status(tmp_path, monkeypatch):
    output_base = tmp_path / "outputs"
    region_root = output_base / "regions" / "malacca"
    (region_root / "metrics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"date": "2024-01-01", "throughput_index_total": 1.0, "coverage_score": 0.9},
            {"date": "2024-01-02", "throughput_index_total": 0.0, "coverage_score": 0.9},
        ]
    ).to_parquet(region_root / "metrics" / "daily.parquet", index=False)

    monkeypatch.setattr(
        "automation.daily.run_single_day",
        lambda **kwargs: {
            "status": "completed",
            "coverage": {"coverage_score": 0.9},
        },
    )
    monkeypatch.setattr("automation.daily.run_backtests_if_new_signal", lambda *args, **kwargs: [])
    sent = {}

    def _capture(payload, dry_run=False):
        sent["payload"] = payload
        sent["dry_run"] = dry_run
        return {"message": "ok", "dry_run": dry_run}

    monkeypatch.setattr("automation.daily.send_signal_alert", _capture)

    from automation.daily import run_daily_region

    result = run_daily_region("malacca", date(2024, 1, 2), output_base=str(output_base), dry_run_alerts=True, version="v2")
    status = load_region_status(str(output_base), "malacca")
    assert result["signal"]["date"] == "2024-01-02"
    assert status["run_status"] == "completed"
    assert sent["dry_run"] is True
