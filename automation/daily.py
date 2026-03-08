"""
Daily automation runners and alert hooks.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone

from automation.alerts import append_alert_log, send_failure_alert, send_signal_alert
from automation.status import load_region_status, save_region_status
from backtesting.run import run_region_symbol_backtest
from pipeline.instruments import list_region_instruments
from pipeline.regions import list_regions, resolve_region_paths
from pipeline.run import run_single_day
from pipeline.signals import latest_region_signal


def run_backtests_if_new_signal(region_id: str, output_base: str = "outputs", version: str = "v2") -> list[dict]:
    latest = latest_region_signal(region_id, output_base=output_base, version=version)
    if latest is None:
        return []

    status = load_region_status(output_base, region_id)
    if status.get("last_signal_date") == latest["date"] and status.get("last_signal") == latest["signal"]:
        return []

    results = []
    for instrument in list_region_instruments(region_id, enabled_for_backtest=True):
        results.append(
            run_region_symbol_backtest(
                region_id=region_id,
                symbol=instrument["ticker"],
                output_base=output_base,
                version=version,
            )
        )
    return results


def run_daily_region(
    region_id: str,
    run_date: date,
    output_base: str = "outputs",
    dry_run_alerts: bool = False,
    version: str = "v2",
) -> dict:
    aoi_path, gate_path = resolve_region_paths(region_id)
    status = load_region_status(output_base, region_id)

    try:
        report = run_single_day(
            target_date=run_date,
            aoi_path=aoi_path,
            gate_path=gate_path,
            output_base=output_base,
            region_id=region_id,
        )
        latest = latest_region_signal(region_id, output_base=output_base, selected_day=run_date.isoformat(), version=version)
        backtests = run_backtests_if_new_signal(region_id, output_base=output_base, version=version)

        updated_status = {
            "region": region_id,
            "last_run_at": datetime.now(timezone.utc).isoformat(),
            "last_success_at": datetime.now(timezone.utc).isoformat(),
            "last_signal_date": latest["date"] if latest else None,
            "last_signal": latest["signal"] if latest else None,
            "latest_coverage": report.get("coverage", {}).get("coverage_score"),
            "run_status": report["status"],
        }
        save_region_status(output_base, region_id, updated_status)

        if latest:
            previous_signal = status.get("last_signal")
            should_alert = (
                previous_signal != latest["signal"]
                or latest["actionability"] == "Actionable"
                or (latest.get("coverage_score") or 0.0) < 0.55
            )
            if should_alert:
                payload = {
                    "region": region_id,
                    "date": latest["date"],
                    "signal": latest["signal"],
                    "confidence": latest["confidence"],
                    "throughput_value": latest["throughput_index_corrected"],
                    "baseline_value": latest["baseline_value"],
                    "coverage": latest["coverage_score"],
                    "instruments": [item["ticker"] for item in list_region_instruments(region_id, enabled_for_alerts=True)],
                    "ui_path": "http://localhost:8501",
                }
                append_alert_log(output_base, region_id, "signal", payload)
                send_signal_alert(payload, dry_run=dry_run_alerts)

        return {"report": report, "signal": latest, "backtests": backtests}
    except Exception as exc:
        payload = {
            "region": region_id,
            "stage": "daily_run",
            "error": str(exc),
            "last_success_at": status.get("last_success_at"),
        }
        append_alert_log(output_base, region_id, "failure", payload)
        send_failure_alert(payload, dry_run=dry_run_alerts)
        save_region_status(
            output_base,
            region_id,
            {
                "region": region_id,
                "last_run_at": datetime.now(timezone.utc).isoformat(),
                "last_success_at": status.get("last_success_at"),
                "last_signal_date": status.get("last_signal_date"),
                "latest_coverage": status.get("latest_coverage"),
                "run_status": "failed",
            },
        )
        raise


def run_daily_all_regions(run_date: date, output_base: str = "outputs", dry_run_alerts: bool = False, version: str = "v2") -> list[dict]:
    results = []
    for region in list_regions():
        try:
            results.append(run_daily_region(region["id"], run_date, output_base=output_base, dry_run_alerts=dry_run_alerts, version=version))
        except Exception:
            continue
    return results


def main():
    parser = argparse.ArgumentParser(description="QuantTrade daily automation runner")
    parser.add_argument("--date", type=str, default=date.today().isoformat())
    parser.add_argument("--region", type=str)
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--dry-run-alerts", action="store_true")
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2"])
    args = parser.parse_args()

    run_date = date.fromisoformat(args.date)
    if args.region:
        result = run_daily_region(args.region, run_date, output_base=args.output, dry_run_alerts=args.dry_run_alerts, version=args.version)
        print(result)
    else:
        results = run_daily_all_regions(run_date, output_base=args.output, dry_run_alerts=args.dry_run_alerts, version=args.version)
        print(results)


if __name__ == "__main__":
    main()
