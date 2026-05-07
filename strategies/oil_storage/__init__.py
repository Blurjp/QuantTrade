"""
Oil Storage Strategy.

Monitors oil storage tank levels (Cushing, etc.).

High inventory = bearish for oil prices (oversupply)
Low inventory = bullish for oil prices (supply tightness)

Trade mapping:
- High storage → Short oil/energy
- Low storage → Long oil/energy
"""
from __future__ import annotations

from typing import Optional, Dict, Any, List
from datetime import datetime

import pandas as pd

from strategies.base import (
    BaseStrategy,
    ResearchSignal,
    TradeCandidate,
    StrategyConfig,
    Direction,
    AssetType,
)


class OilStorageStrategy(BaseStrategy):
    """
    Oil storage monitoring strategy.

    Uses satellite imagery to measure tank fill levels.

    Signal interpretation:
    - High fill levels → BEARISH for oil (oversupply)
    - Low fill levels → BULLISH for oil (supply constraint)
    """

    name: str = "oil_storage"
    version: str = "1.0.0"
    description: str = "Monitor oil storage tank levels via satellite imagery"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the strategy."""
        self.config = StrategyConfig()
        if config:
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        self.oil_tickers = ["USO", "XLE", "XOM", "CVX", "SLB"]
        self.default_horizon_days = 21
        self.default_size_pct = 0.02
        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.10

    def load_inputs(
        self,
        start_date: str,
        end_date: str,
        region: Optional[str] = None,
        output_base: str = "outputs",
    ) -> pd.DataFrame:
        """Load storage data."""
        # Returns DataFrame with: date, region, fill_pct, capacity
        return pd.DataFrame(columns=[
            "date", "region", "fill_pct", "capacity", "baseline_fill"
        ])

    def build_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Compute features from storage data."""
        if raw_df.empty:
            return raw_df

        df = raw_df.copy()

        if "date" not in df.columns and df.index.name == "date":
            df = df.reset_index()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        if "fill_pct" in df.columns:
            df["baseline_fill"] = df["fill_pct"].rolling(window=28, min_periods=5).median()
            df["pct_change"] = df["fill_pct"] - df["baseline_fill"]
            df["zscore"] = (df["fill_pct"] - df["baseline_fill"]) / 10  # 10% = 1 sigma

        df["data_quality"] = df.get("coverage_ratio", 0.5)
        df["sample_count"] = 1

        return df

    def generate_signal(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from storage levels."""
        if feature_df.empty:
            return feature_df

        df = feature_df.copy()
        df["signal_direction"] = "neutral"
        df["signal_strength"] = 0.0
        df["signal"] = "No data"

        if "zscore" in df.columns:
            # High fill = bearish for oil (short)
            df.loc[df["zscore"] >= 1.0, "signal_direction"] = "short"
            df.loc[df["zscore"] <= -1.0, "signal_direction"] = "long"
            df["signal_strength"] = df["zscore"].abs()

            df.loc[df["signal_direction"] == "short", "signal"] = df["zscore"].apply(
                lambda x: f"High storage ({x:.1f}σ) - Oil oversupply"
            )
            df.loc[df["signal_direction"] == "long", "signal"] = df["zscore"].apply(
                lambda x: f"Low storage ({x:.1f}σ) - Oil supply tight"
            )

        return pd.DataFrame([df.iloc[-1].to_dict()]) if len(df) > 0 else df

    def estimate_confidence(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        """Estimate signal confidence."""
        if signal_df.empty:
            return signal_df

        df = signal_df.copy()
        df["confidence"] = df.get("data_quality", 0.5)
        df["confidence_level"] = df["confidence"].apply(self._confidence_to_level)
        df["actionability"] = df["confidence_level"].apply(
            lambda x: "Actionable" if x in ["High", "Medium"] else "Ignore"
        )
        return df

    def map_to_trade(self, signal_df: pd.DataFrame) -> List[TradeCandidate]:
        """Map signal to trade candidates."""
        if signal_df.empty:
            return []

        df = signal_df.iloc[0]

        if df.get("actionability") != "Actionable":
            return []

        direction_str = df.get("signal_direction", "neutral")
        if direction_str == "neutral":
            return []

        direction = Direction.SHORT if direction_str == "short" else Direction.LONG
        region = df.get("region", "unknown")

        candidates = []
        for ticker in self.oil_tickers:
            candidates.append(TradeCandidate(
                strategy=self.name,
                timestamp=datetime.now().isoformat(),
                ticker=ticker,
                asset_type=AssetType.ETF if ticker == "USO" else AssetType.EQUITY,
                direction=direction,
                horizon_days=self.default_horizon_days,
                size_pct=self.default_size_pct,
                stop_loss_pct=self.stop_loss_pct,
                take_profit_pct=self.take_profit_pct,
                rationale=f"Oil storage {direction.value} at {region}",
                source_signals=[f"{region}_{datetime.now().strftime('%Y-%m-%d')}"],
                probability=float(df.get("confidence", 0.5)),
            ))

        return candidates

    def get_config(self) -> Dict[str, Any]:
        """Get strategy configuration."""
        return self.config.to_dict()

    def _confidence_to_level(self, confidence: float) -> str:
        """Convert confidence score to level."""
        if confidence >= 0.7:
            return "High"
        elif confidence >= 0.4:
            return "Medium"
        else:
            return "Low"


def create_oil_storage_signal(
    region_id: str,
    target_date: str,
    fill_pct: float,
    baseline_fill: float,
    coverage_ratio: float = 1.0,
) -> ResearchSignal:
    """
    Create a research signal from oil storage data.

    Args:
        region_id: Region identifier
        target_date: Date of observation
        fill_pct: Current fill percentage
        baseline_fill: Baseline fill percentage
        coverage_ratio: Detection coverage

    Returns:
        ResearchSignal object
    """
    pct_diff = fill_pct - baseline_fill
    zscore = pct_diff / 10  # 10% = 1 sigma

    if zscore >= 1.0:
        direction = Direction.SHORT
        confidence = min(0.9, 0.5 + zscore * 0.1)
        thesis = f"Oil storage {pct_diff:.1f}% above baseline - Oversupply bearish for prices"
    elif zscore <= -1.0:
        direction = Direction.LONG
        confidence = min(0.9, 0.5 + abs(zscore) * 0.1)
        thesis = f"Oil storage {pct_diff:.1f}% below baseline - Supply tightness bullish"
    else:
        direction = Direction.NEUTRAL
        confidence = 0.3
        thesis = f"Oil storage near baseline ({pct_diff:+.1f}% change)"

    return ResearchSignal(
        strategy="oil_storage",
        timestamp=target_date,
        region=region_id,
        direction=direction,
        strength=abs(zscore),
        confidence=confidence,
        data_quality=coverage_ratio,
        sample_count=1,
        coverage_ratio=coverage_ratio,
        thesis=thesis,
        raw_value=fill_pct,
        baseline_value=baseline_fill,
    )


__all__ = [
    "OilStorageStrategy",
    "create_oil_storage_signal",
]
