"""Backtest the combined agriculture signal against forward instrument returns."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtesting.market_data import fetch_yahoo_prices
from pipeline.agriculture_signal import AGRICULTURE_SETUPS, build_agriculture_signals


@dataclass
class HorizonSummary:
    horizon_days: int
    sample_count: int
    hit_rate: float | None
    avg_forward_return: float | None
    median_forward_return: float | None
    avg_signal_return: float | None


def _prepare_price_frame(symbol: str, output_base: str, start: str, end: str, refresh: bool) -> pd.DataFrame:
    price_df = fetch_yahoo_prices(symbol, start=start, end=end, output_base=output_base, refresh=refresh)
    if price_df.empty:
        return pd.DataFrame()
    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.strftime("%Y-%m-%d")
    return price_df[["date", "Close"]].rename(columns={"Close": "close"})


def _direction_to_sign(action: str) -> int:
    if action == "LONG":
        return 1
    if action == "SHORT":
        return -1
    return 0


def build_signal_frame(start: str, end: str, output_base: str) -> pd.DataFrame:
    rows: list[dict] = []
    for current in pd.date_range(start=start, end=end, freq="D"):
        target_date = current.strftime("%Y-%m-%d")
        signals = build_agriculture_signals(target_date, output_base=output_base)
        for signal_id, payload in signals.items():
            rows.append(
                {
                    "date": target_date,
                    "signal_id": signal_id,
                    "action": payload.get("trading_action", "FLAT"),
                    "actionability": payload.get("actionability", "Ignore"),
                    "confidence": payload.get("confidence", "Low"),
                    "numeric_confidence": payload.get("numeric_confidence", 0.0),
                    "real_data_ratio": payload.get("real_data_ratio", 0.0),
                    "critical_season": payload.get("critical_season", False),
                    "data_quality_mode": payload.get("data_quality_mode", "unknown"),
                    "combined_score": payload.get("combined_score", 0),
                    "rationale": payload.get("rationale", ""),
                }
            )
    return pd.DataFrame(rows)


def summarize_horizon(df: pd.DataFrame, horizon_days: int) -> HorizonSummary:
    signal_return_col = f"signal_return_{horizon_days}d"
    forward_col = f"forward_return_{horizon_days}d"
    valid = df.dropna(subset=[signal_return_col, forward_col])
    if valid.empty:
        return HorizonSummary(horizon_days, 0, None, None, None, None)

    hits = (valid[signal_return_col] > 0).mean()
    return HorizonSummary(
        horizon_days=horizon_days,
        sample_count=int(len(valid)),
        hit_rate=round(float(hits), 4),
        avg_forward_return=round(float(valid[forward_col].mean()), 6),
        median_forward_return=round(float(valid[forward_col].median()), 6),
        avg_signal_return=round(float(valid[signal_return_col].mean()), 6),
    )


def backtest_signal(
    signal_id: str,
    symbol: str,
    start: str,
    end: str,
    output_base: str = "outputs",
    refresh_prices: bool = False,
) -> dict:
    signal_df = build_signal_frame(start=start, end=end, output_base=output_base)
    signal_df = signal_df[signal_df["signal_id"] == signal_id].copy()
    if signal_df.empty:
        raise ValueError(f"No signals generated for {signal_id}")

    signal_df = signal_df[signal_df["actionability"] == "Actionable"].copy()
    if signal_df.empty:
        return {
            "signal_id": signal_id,
            "symbol": symbol,
            "start": start,
            "end": end,
            "error": "No actionable signals in range",
        }

    price_df = _prepare_price_frame(symbol, output_base, start, end, refresh_prices)
    if price_df.empty:
        return {
            "signal_id": signal_id,
            "symbol": symbol,
            "start": start,
            "end": end,
            "error": "No market data available",
        }

    merged = signal_df.merge(price_df, on="date", how="inner")
    if merged.empty:
        return {
            "signal_id": signal_id,
            "symbol": symbol,
            "start": start,
            "end": end,
            "error": "No overlapping signal and price dates",
        }

    merged["close"] = pd.to_numeric(merged["close"], errors="coerce")
    merged = merged.sort_values("date").reset_index(drop=True)
    merged["signal_sign"] = merged["action"].map(_direction_to_sign)

    horizons = [5, 10, 20]
    for horizon in horizons:
        merged[f"forward_return_{horizon}d"] = merged["close"].shift(-horizon) / merged["close"] - 1.0
        merged[f"signal_return_{horizon}d"] = merged[f"forward_return_{horizon}d"] * merged["signal_sign"]

    summaries = [asdict(summarize_horizon(merged, horizon)) for horizon in horizons]
    report = {
        "signal_id": signal_id,
        "symbol": symbol,
        "start": start,
        "end": end,
        "sample_count": int(len(merged)),
        "real_data_coverage_pct": round(float((merged["real_data_ratio"] >= 0.5).mean() * 100), 2),
        "critical_season_pct": round(float(merged["critical_season"].mean() * 100), 2),
        "data_quality_modes": merged["data_quality_mode"].value_counts(dropna=False).to_dict(),
        "horizons": summaries,
    }

    output_dir = Path(output_base) / "backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{signal_id}_{symbol}_agriculture_backtest.json"
    output_file.write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the combined agriculture signal.")
    parser.add_argument("--signal-id", choices=sorted(AGRICULTURE_SETUPS.keys()), default="agriculture_us_corn_soy")
    parser.add_argument("--symbol", default="CORN")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--output-base", default="outputs")
    parser.add_argument("--refresh-prices", action="store_true")
    args = parser.parse_args()

    report = backtest_signal(
        signal_id=args.signal_id,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        output_base=args.output_base,
        refresh_prices=args.refresh_prices,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
