"""
Standalone alert sender for dry-runs and manual testing.
"""

import argparse

from automation.alerts import send_failure_alert, send_signal_alert


def main():
    parser = argparse.ArgumentParser(description="QuantTrade alert sender")
    parser.add_argument("--type", choices=["signal", "failure"], required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--date")
    parser.add_argument("--signal", default="No trade")
    parser.add_argument("--confidence", default="Unknown")
    parser.add_argument("--throughput", type=float, default=0.0)
    parser.add_argument("--baseline", type=float, default=0.0)
    parser.add_argument("--coverage", type=float, default=0.0)
    parser.add_argument("--stage", default="manual")
    parser.add_argument("--error", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.type == "signal":
        payload = {
            "region": args.region,
            "date": args.date or "n/a",
            "signal": args.signal,
            "confidence": args.confidence,
            "throughput_value": args.throughput,
            "baseline_value": args.baseline,
            "coverage": args.coverage,
            "instruments": [],
            "ui_path": "http://localhost:8501",
        }
        print(send_signal_alert(payload, dry_run=args.dry_run))
    else:
        payload = {
            "region": args.region,
            "stage": args.stage,
            "error": args.error or "manual failure test",
            "last_success_at": "n/a",
        }
        print(send_failure_alert(payload, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
