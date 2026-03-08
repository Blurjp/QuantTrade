"""
Persist backtest artifacts.
"""

import json
from pathlib import Path

import pandas as pd


def save_backtest_report(
    output_base: str,
    region_id: str,
    strategy_name: str,
    symbol: str,
    equity_df: pd.DataFrame,
    summary: dict,
) -> dict:
    output_dir = Path(output_base) / "regions" / region_id / "backtests" / strategy_name
    output_dir.mkdir(parents=True, exist_ok=True)

    equity_path = output_dir / f"{symbol}_equity.parquet"
    summary_path = output_dir / f"{symbol}_summary.json"
    trades_path = output_dir / f"{symbol}_signals.parquet"

    equity_df.to_parquet(equity_path, index=False)
    equity_df.to_parquet(trades_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return {
        "equity_path": str(equity_path),
        "summary_path": str(summary_path),
        "signals_path": str(trades_path),
    }
