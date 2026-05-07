"""
Backtest runner for QuantTrade strategies.

Usage:
    # Backtest a single region-symbol pair
    python scripts/run_backtest.py --region brazil_soy_north --symbol SOYB

    # Backtest all symbols for a region
    python scripts/run_backtest.py --region brazil_soy_north --all-symbols

    # Backtest all regions
    python scripts/run_backtest.py --all-regions

    # Walk-forward validation
    python scripts/run_backtest.py --region brazil_soy_north --symbol SOYB --walk-forward
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple
import pandas as pd

from backtesting.engine import run_backtest
from backtesting.market_data import fetch_yahoo_prices
from backtesting.reports import save_backtest_report
from backtesting.signals import build_backtest_signal_table
from pipeline.instruments import list_region_instruments
from pipeline.regions import list_regions


def _prepare_price_frame(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame()

    price_column = "Open" if "Open" in price_df.columns else "Close"
    return price_df.rename(columns={price_column: "price_open", "Close": "price_close"})[
        ["date", "price_open", "price_close", "Volume"]
    ].copy()


def run_region_symbol_backtest(
    region_id: str,
    symbol: str,
    output_base: str = "outputs",
    start: str | None = None,
    end: str | None = None,
    version: str = "v2",
    use_corrected: bool | None = None,
    refresh_prices: bool = False,
) -> dict:
    """
    Run backtest for a single region-symbol pair.

    Args:
        region_id: Region identifier
        symbol: Ticker symbol
        output_base: Output directory
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        version: Signal version (v1 or v2)
        use_corrected: Use corrected signals
        refresh_prices: Refetch price data

    Returns:
        Backtest summary dict
    """
    signal_df = build_backtest_signal_table(
        region_id=region_id,
        symbol=symbol,
        output_base=output_base,
        version=version,
        use_corrected=use_corrected,
    )
    if signal_df.empty:
        raise ValueError(f"No signal data available for region={region_id} symbol={symbol}")

    price_df = fetch_yahoo_prices(symbol, start=start, end=end, output_base=output_base, refresh=refresh_prices)
    price_df = _prepare_price_frame(price_df)
    if price_df.empty:
        raise ValueError(f"No market data returned for {symbol}")

    merged = price_df.merge(signal_df, on="date", how="inner")
    if merged.empty:
        raise ValueError(f"No overlapping dates between signal and price data for {symbol}")

    equity_df, summary = run_backtest(
        price_df=merged[["date", "price_open", "price_close", "Volume"]].copy(),
        signal_df=merged[["date", "position", "signal", "confidence"]].copy(),
    )
    equity_df = equity_df.merge(
        merged[["date", "region", "signal", "confidence", "signal_strength", "price_symbol"]],
        on="date",
        how="left",
    )
    summary.update({
        "region": region_id,
        "symbol": symbol,
        "strategy_name": f"{version}_{symbol.lower()}",
    })
    artifacts = save_backtest_report(
        output_base=output_base,
        region_id=region_id,
        strategy_name=summary["strategy_name"],
        symbol=symbol,
        equity_df=equity_df,
        summary=summary,
    )
    summary.update(artifacts)
    return summary


def run_walk_forward_backtest(
    region_id: str,
    symbol: str,
    train_months: int = 12,
    validate_months: int = 3,
    output_base: str = "outputs",
    version: str = "v2",
) -> dict:
    """
    Run walk-forward validation backtest.

    Args:
        region_id: Region identifier
        symbol: Ticker symbol
        train_months: Training period in months
        validate_months: Validation period in months
        output_base: Output directory
        version: Signal version

    Returns:
        Walk-forward summary dict
    """
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta

    # Get full signal range
    signal_df = build_backtest_signal_table(
        region_id=region_id,
        symbol=symbol,
        output_base=output_base,
        version=version,
    )

    if signal_df.empty:
        raise ValueError(f"No signal data available for region={region_id} symbol={symbol}")

    signal_df["date"] = pd.to_datetime(signal_df["date"])
    min_date = signal_df["date"].min()
    max_date = signal_df["date"].max()

    # Generate walk-forward windows
    windows = []
    current_start = min_date

    while current_start < max_date:
        train_end = current_start + relativedelta(months=train_months)
        validate_end = train_end + relativedelta(months=validate_months)

        if validate_end > max_date:
            break

        windows.append({
            "train_start": current_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "validate_start": train_end.strftime("%Y-%m-%d"),
            "validate_end": validate_end.strftime("%Y-%m-%d"),
        })

        current_start = train_end

    print(f"Running {len(windows)} walk-forward windows...")
    results = []

    for i, window in enumerate(windows):
        print(f"\n[{i+1}/{len(windows)}] {window['validate_start']} to {window['validate_end']}")

        try:
            result = run_region_symbol_backtest(
                region_id=region_id,
                symbol=symbol,
                output_base=output_base,
                start=window["validate_start"],
                end=window["validate_end"],
                version=version,
            )
            result["window"] = window
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "region": region_id,
                "symbol": symbol,
                "window": window,
                "error": str(e),
                "total_return": None,
                "sharpe_ratio": None,
            })

    # Aggregate results
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        avg_return = pd.DataFrame([r["total_return"] for r in valid_results]).mean()
        avg_sharpe = pd.DataFrame([r["sharpe_ratio"] for r in valid_results]).mean()

        summary = {
            "region": region_id,
            "symbol": symbol,
            "walk_forward": True,
            "train_months": train_months,
            "validate_months": validate_months,
            "num_windows": len(windows),
            "successful_windows": len(valid_results),
            "avg_total_return": float(avg_return) if not pd.isna(avg_return) else None,
            "avg_sharpe_ratio": float(avg_sharpe) if not pd.isna(avg_sharpe) else None,
            "windows": results,
        }
    else:
        summary = {
            "region": region_id,
            "symbol": symbol,
            "walk_forward": True,
            "error": "All windows failed",
            "windows": results,
        }

    # Save walk-forward report
    output_path = Path(output_base) / "backtest" / region_id
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / f"{symbol}_walk_forward.json"

    import json
    report_file.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWalk-forward report saved to: {report_file}")

    return summary


def run_batch_backtest(
    targets: List[Tuple[str, str]],
    output_base: str = "outputs",
    start: str | None = None,
    end: str | None = None,
    version: str = "v2",
    refresh_prices: bool = False,
) -> dict:
    """
    Run backtest for multiple region-symbol pairs.

    Args:
        targets: List of (region_id, symbol) tuples
        output_base: Output directory
        start: Start date filter
        end: End date filter
        version: Signal version
        refresh_prices: Refetch price data

    Returns:
        Batch backtest summary
    """
    results = []
    failures = []

    print(f"Running {len(targets)} backtests...")

    for i, (region_id, symbol) in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}] {region_id} / {symbol}")

        try:
            result = run_region_symbol_backtest(
                region_id=region_id,
                symbol=symbol,
                output_base=output_base,
                start=start,
                end=end,
                version=version,
                refresh_prices=refresh_prices,
            )
            results.append(result)
            print(f"  Total Return: {result.get('total_return', 'N/A')}")
            print(f"  Sharpe Ratio: {result.get('sharpe_ratio', 'N/A')}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failures.append({"region": region_id, "symbol": symbol, "error": str(exc)})

    # Print summary
    print(f"\n{'='*60}")
    print("BACKTEST BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {len(results)}/{len(targets)}")
    print(f"Failed: {len(failures)}/{len(targets)}")

    if results:
        result_df = pd.DataFrame(results)
        print(f"\nResults:")
        print(result_df[["region", "symbol", "total_return", "sharpe_ratio", "win_rate"]].to_string(index=False))

    if failures:
        print(f"\nFailures:")
        print(pd.DataFrame(failures).to_string(index=False))

    # Save batch summary
    output_path = Path(output_base) / "backtest"
    output_path.mkdir(parents=True, exist_ok=True)

    import json
    batch_file = output_path / "batch_summary.json"
    batch_summary = {
        "successful": len(results),
        "failed": len(failures),
        "results": results,
        "failures": failures,
    }
    batch_file.write_text(json.dumps(batch_summary, indent=2, default=str))

    return batch_summary


def main():
    parser = argparse.ArgumentParser(
        description="QuantTrade backtest runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single backtest
  python scripts/run_backtest.py --region brazil_soy_north --symbol SOYB

  # All symbols for a region
  python scripts/run_backtest.py --region brazil_soy_north --all-symbols

  # All regions
  python scripts/run_backtest.py --all-regions

  # Walk-forward validation
  python scripts/run_backtest.py --region brazil_soy_north --symbol SOYB --walk-forward
        """
    )

    parser.add_argument("--region", type=str, help="Configured region ID")
    parser.add_argument("--symbol", type=str, help="Ticker to backtest")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2"], help="Signal version")
    parser.add_argument("--all-symbols", action="store_true", help="Backtest all symbols for region")
    parser.add_argument("--all-regions", action="store_true", help="Backtest all configured regions")
    parser.add_argument("--use-corrected", action="store_true", help="Use corrected signals")
    parser.add_argument("--use-raw", action="store_true", help="Use raw (uncorrected) signals")
    parser.add_argument("--refresh-prices", action="store_true", help="Refetch price data")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation")

    args = parser.parse_args()

    use_corrected = None
    if args.use_corrected:
        use_corrected = True
    if args.use_raw:
        use_corrected = False

    # Walk-forward mode
    if args.walk_forward:
        if not args.region or not args.symbol:
            parser.error("--region and --symbol required for walk-forward mode")

        run_walk_forward_backtest(
            region_id=args.region,
            symbol=args.symbol,
            output_base=args.output,
            version=args.version,
        )
        return

    # Determine targets
    targets: List[Tuple[str, str]] = []

    if args.all_regions:
        for region in list_regions():
            for instrument in list_region_instruments(region["id"], enabled_for_backtest=True):
                targets.append((region["id"], instrument["ticker"]))
    elif args.region and args.all_symbols:
        for instrument in list_region_instruments(args.region, enabled_for_backtest=True):
            targets.append((args.region, instrument["ticker"]))
    elif args.region and args.symbol:
        targets.append((args.region, args.symbol))
    else:
        parser.error("Use --region --symbol, --region --all-symbols, or --all-regions")

    # Run batch
    run_batch_backtest(
        targets=targets,
        output_base=args.output,
        start=args.start,
        end=args.end,
        version=args.version,
        refresh_prices=args.refresh_prices,
    )


if __name__ == "__main__":
    main()
