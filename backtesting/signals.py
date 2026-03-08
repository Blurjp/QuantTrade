"""
Signal normalization for backtesting.
"""

from __future__ import annotations

import pandas as pd

from pipeline.signals import build_region_signal_table


def signal_to_position(signal: str, confidence: str) -> int:
    if confidence == "Low":
        return 0
    if signal == "Long disruption risk":
        return 1
    if signal == "Short disruption risk":
        return -1
    return 0


def build_backtest_signal_table(
    region_id: str,
    symbol: str,
    output_base: str = "outputs",
    version: str = "v2",
    use_corrected: bool | None = None,
) -> pd.DataFrame:
    signal_df = build_region_signal_table(region_id, output_base=output_base, version=version)
    if signal_df.empty:
        return pd.DataFrame()

    if use_corrected is True:
        signal_df = signal_df[signal_df["signal_source"] == "throughput_index_corrected"].copy()
    elif use_corrected is False:
        signal_df = signal_df[signal_df["signal_source"] == "throughput_index_total"].copy()

    if signal_df.empty:
        return pd.DataFrame()

    df = signal_df.copy()
    df["position"] = [
        signal_to_position(signal, confidence)
        for signal, confidence in zip(df["signal"], df["confidence"])
    ]
    df["price_symbol"] = symbol
    return df[
        [
            "date",
            "region",
            "signal",
            "confidence",
            "signal_strength",
            "price_symbol",
            "position",
            "actionability",
            "throughput_value",
            "baseline_value",
            "coverage_score",
            "reroute_flag",
            "signal_source",
        ]
    ].rename(columns={"throughput_value": "signal_value"})
