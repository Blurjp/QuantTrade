"""
CLI entrypoint for region-aware backtests.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description="QuantTrade backtest runner")
    parser.add_argument("--region", type=str, help="Configured region ID")
    parser.add_argument("--symbol", type=str, help="Ticker to backtest")
    parser.add_argument("--start", type=str)
    parser.add_argument("--end", type=str)
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2"])
    parser.add_argument("--all-symbols", action="store_true")
    parser.add_argument("--all-regions", action="store_true")
    parser.add_argument("--use-corrected", action="store_true")
    parser.add_argument("--use-raw", action="store_true")
    parser.add_argument("--refresh-prices", action="store_true")
    args = parser.parse_args()

    use_corrected = None
    if args.use_corrected:
        use_corrected = True
    if args.use_raw:
        use_corrected = False

    targets: list[tuple[str, str]] = []
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

    results = []
    failures = []
    for region_id, symbol in targets:
        try:
            results.append(
                run_region_symbol_backtest(
                    region_id=region_id,
                    symbol=symbol,
                    output_base=args.output,
                    start=args.start,
                    end=args.end,
                    version=args.version,
                    use_corrected=use_corrected,
                    refresh_prices=args.refresh_prices,
                )
            )
        except Exception as exc:
            failures.append({"region": region_id, "symbol": symbol, "error": str(exc)})

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        print(result_df.to_string(index=False))
    if failures:
        print("\nFailures:")
        print(pd.DataFrame(failures).to_string(index=False))


if __name__ == "__main__":
    main()
