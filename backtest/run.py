"""
Backtest CLI Runner

Usage:
    python -m backtest.run --from 2025-01-01 --to 2026-03-01
    python -m backtest.run --from 2025-06-01 --to 2026-01-01 --capital 50000
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.backtest_engine import BacktestEngine


def main():
    parser = argparse.ArgumentParser(description="QuantTrade Backtest Engine")
    parser.add_argument(
        "--from", dest="from_date", required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--to", dest="to_date", required=True,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--capital", type=float, default=100000,
        help="Initial capital (default: 100000)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file for report (default: stdout)"
    )
    parser.add_argument(
        "--json", dest="json_output", default=None,
        help="Output file for JSON results"
    )
    parser.add_argument(
        "--stop-loss", type=float, default=8,
        help="Stop loss percentage (default: 8)"
    )
    parser.add_argument(
        "--take-profit", type=float, default=15,
        help="Take profit percentage (default: 15)"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=75,
        help="Minimum signal confidence (default: 75)"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    engine = BacktestEngine(
        output_base="outputs",
        initial_capital=args.capital,
        stop_loss_pct=args.stop_loss / 100,
        take_profit_pct=args.take_profit / 100,
        min_confidence=args.min_confidence,
    )

    results = engine.run(args.from_date, args.to_date)

    if "error" in results:
        print(f"Backtest error: {results['error']}")
        sys.exit(1)

    # Print report
    report = engine.generate_report(results, output_file=args.output)
    print(report)

    # Save JSON
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(results, indent=2))
        print(f"\nJSON results saved to {args.json_output}")


if __name__ == "__main__":
    main()
