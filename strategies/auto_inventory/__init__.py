"""
Auto Inventory Strategy.

Monitors vehicle inventory levels at parking lots/dealerships.
High inventory = bearish for auto stocks (demand weakness)
Low inventory = bullish for auto stocks (supply constraint)

Trade mapping:
- High inventory → Short CARZ, F, GM, etc.
- Low inventory → Long auto stocks (or avoid short)
"""
from __future__ import annotations

from typing import Optional, Dict, Any, List
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import numpy as np

from strategies.base import (
    BaseStrategy,
    ResearchSignal,
    TradeCandidate,
    StrategyConfig,
    Direction,
    AssetType,
)


class AutoInventoryStrategy(BaseStrategy):
    """
    Auto inventory monitoring strategy.

    Uses satellite imagery to detect vehicle counts at
    dealerships and parking lots.

    Signal interpretation:
    - Inventory significantly above baseline → BEARISH
    - Inventory significantly below baseline → BULLISH
    """

    # Strategy identity
    name: str = "auto_inventory"
    version: str = "1.0.0"
    description: str = "Monitor vehicle inventory levels via satellite imagery"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy.

        Args:
            config: Optional strategy configuration
        """
        self.config = StrategyConfig()
        if config:
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        # Trading parameters
        self.default_tickers = ["CARZ", "F", "GM", "STLA"]
        self.default_horizon_days = 30
        self.default_size_pct = 0.02
        self.stop_loss_pct = 0.08
        self.take_profit_pct = 0.15

    def load_inputs(
        self,
        start_date: str,
        end_date: str,
        region: Optional[str] = None,
        output_base: str = "outputs",
    ) -> pd.DataFrame:
        """
        Load historical inventory data.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            region: Region identifier
            output_base: Output directory

        Returns:
            DataFrame with columns: date, region, count, coverage_ratio, etc.
        """
        from pipeline.detection_multi import run_detection
        from pipeline.regions import load_registry

        registry = load_registry()
        region_config = registry.get("regions", {}).get(region, {})

        if not region_config:
            # Return empty DataFrame with expected structure
            return pd.DataFrame(columns=[
                "date", "region", "count", "coverage_ratio",
                "detector_confidence", "baseline_count"
            ])

        aoi_file = region_config.get("aoi_file")
        if not aoi_file or not Path(aoi_file).exists():
            return pd.DataFrame(columns=[
                "date", "region", "count", "coverage_ratio",
                "detector_confidence", "baseline_count"
            ])

        # Load detection results
        try:
            detection_result = run_detection(
                monitoring_type="auto_inventory",
                aoi_path=aoi_file,
                target_date=end_date,
                output_base=output_base,
            )

            # Extract data
            if hasattr(detection_result, "to_dict"):
                result_dict = detection_result.to_dict()
            else:
                result_dict = detection_result

            details = result_dict.get("details", [])
            if not details:
                return pd.DataFrame(columns=[
                    "date", "region", "count", "coverage_ratio",
                    "detector_confidence", "baseline_count"
                ])

            df = pd.DataFrame(details)
            df["region"] = region

            return df

        except Exception as e:
            print(f"Error loading data for {region}: {e}")
            return pd.DataFrame(columns=[
                "date", "region", "count", "coverage_ratio",
                "detector_confidence", "baseline_count"
            ])

    def build_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features from raw inventory data.

        Features:
        - pct_change: % change from baseline
        - zscore: Z-score relative to historical distribution
        - coverage_ratio: Spatial coverage of detection
        - days_since: Days since last observation

        Args:
            raw_df: Raw data from load_inputs()

        Returns:
            DataFrame with engineered features
        """
        if raw_df.empty:
            return raw_df

        df = raw_df.copy()

        # Ensure date column
        if "date" not in df.columns and df.index.name == "date":
            df = df.reset_index()

        if "date" not in df.columns:
            return df

        df["date"] = pd.to_datetime(df["date"])

        # Count-based features
        if "count" in df.columns:
            # Compute baseline (rolling median)
            df["baseline_count"] = df["count"].rolling(
                window=28, min_periods=5
            ).median()

            # Percent change from baseline
            df["pct_change"] = (
                (df["count"] - df["baseline_count"]) / df["baseline_count"].replace(0, np.nan)
            ) * 100

            # Z-score
            count_mean = df["count"].rolling(window=28, min_periods=5).mean()
            count_std = df["count"].rolling(window=28, min_periods=5).std()
            df["zscore"] = (df["count"] - count_mean) / count_std.replace(0, np.nan)

        # Quality features
        if "coverage_ratio" in df.columns:
            df["data_quality"] = df["coverage_ratio"]
        else:
            df["data_quality"] = 0.5  # Default

        # Sample count feature
        df["sample_count"] = 1  # Each row is one sample

        return df

    def generate_signal(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from features.

        Signal logic:
        - zscore > short_threshold → SHORT
        - zscore < long_threshold → LONG
        - else → FLAT

        Args:
            feature_df: Feature data from build_features()

        Returns:
            DataFrame with signals
        """
        if feature_df.empty:
            return feature_df

        df = feature_df.copy()

        # Initialize signal columns
        df["signal_direction"] = "neutral"
        df["signal_strength"] = 0.0
        df["signal"] = "No data"

        # Generate signals based on zscore
        if "zscore" in df.columns:
            # Apply thresholds
            df.loc[df["zscore"] >= self.config.short_threshold, "signal_direction"] = "short"
            df.loc[df["zscore"] <= self.config.long_threshold, "signal_direction"] = "long"

            # Signal strength is absolute zscore
            df["signal_strength"] = df["zscore"].abs()

            # Human-readable signals
            df.loc[df["signal_direction"] == "short", "signal"] = (
                df["zscore"].apply(lambda x: f"High inventory (+{x:.1f}σ)")
            )
            df.loc[df["signal_direction"] == "long", "signal"] = (
                df["zscore"].apply(lambda x: f"Low inventory ({x:.1f}σ)")
            )
            df.loc[df["signal_direction"] == "neutral", "signal"] = (
                "Inventory near baseline"
            )

        # Get latest signal
        latest = df.iloc[-1].to_dict()

        # Create single-row signal DataFrame
        signal_df = pd.DataFrame([latest])

        return signal_df

    def estimate_confidence(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate confidence in the signal.

        Confidence factors:
        - Data quality score
        - Signal strength (higher = more confident)
        - Sample count (more history = more confident)

        Args:
            signal_df: Signal data from generate_signal()

        Returns:
            DataFrame with confidence estimates
        """
        if signal_df.empty:
            return signal_df

        df = signal_df.copy()

        # Initialize confidence columns
        df["confidence"] = 0.5
        df["data_quality"] = 0.5

        # Data quality from coverage
        if "data_quality" in df.columns:
            df["data_quality"] = df["data_quality"].clip(0, 1)

        # Confidence based on signal strength
        if "signal_strength" in df.columns:
            strength_conf = df["signal_strength"].clip(0, 3) / 3  # Normalize to 0-1
            df["confidence"] = strength_conf

        # Combine with data quality
        df["confidence"] = df["confidence"] * df["data_quality"]

        # Convert to High/Medium/Low
        df["confidence_level"] = df["confidence"].apply(self._confidence_to_level)
        df["actionability"] = df["confidence_level"].apply(
            lambda x: "Actionable" if x in ["High", "Medium"] else "Ignore"
        )

        return df

    def map_to_trade(self, signal_df: pd.DataFrame) -> List[TradeCandidate]:
        """
        Convert signal to trade candidates.

        Args:
            signal_df: Signal data with confidence estimates

        Returns:
            List of TradeCandidate objects
        """
        if signal_df.empty:
            return []

        df = signal_df.iloc[0]  # Get latest signal

        # Check if signal is actionable
        if df.get("actionability") != "Actionable":
            return []

        direction_str = df.get("signal_direction", "neutral")
        if direction_str == "neutral":
            return []

        # Map direction to enum
        direction = Direction.SHORT if direction_str == "short" else Direction.LONG

        # Get region
        region = df.get("region", "unknown")

        # Get tickers
        tickers = self.default_tickers
        if "tickers" in df:
            tickers = df["tickers"]

        # Create trade candidates
        candidates = []
        for ticker in tickers:
            candidate = TradeCandidate(
                strategy=self.name,
                timestamp=datetime.now().isoformat(),
                ticker=ticker,
                asset_type=AssetType.EQUITY,
                direction=direction,
                horizon_days=self.default_horizon_days,
                size_pct=self.default_size_pct,
                stop_loss_pct=self.stop_loss_pct,
                take_profit_pct=self.take_profit_pct,
                rationale=self._build_rationale(df),
                source_signals=[f"{region}_{datetime.now().strftime('%Y-%m-%d')}"],
                probability=float(df.get("confidence", 0.5)),
            )
            candidates.append(candidate)

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

    def _build_rationale(self, signal_row: pd.Series) -> str:
        """Build human-readable rationale for the trade."""
        direction = signal_row.get("signal_direction", "neutral")
        strength = signal_row.get("signal_strength", 0)

        if direction == "short":
            return (
                f"Auto inventory elevated {strength:.1f}σ above baseline. "
                f"High inventory suggests potential demand weakness for auto manufacturers."
            )
        elif direction == "long":
            return (
                f"Auto inventory depressed {strength:.1f}σ below baseline. "
                f"Low inventory may indicate strong demand or supply constraints."
            )
        else:
            return "Auto inventory near baseline levels."


# Convenience function
def create_auto_inventory_signal(
    region_id: str,
    target_date: str,
    inventory_count: int,
    baseline_count: float,
    coverage_ratio: float = 1.0,
) -> ResearchSignal:
    """
    Create a research signal from auto inventory data.

    Args:
        region_id: Region identifier
        target_date: Date of observation (YYYY-MM-DD)
        inventory_count: Observed vehicle count
        baseline_count: Baseline count for comparison
        coverage_ratio: Detection coverage ratio

    Returns:
        ResearchSignal object
    """
    # Compute zscore
    pct_change = (inventory_count - baseline_count) / baseline_count * 100
    zscore = pct_change / 20  # Assume 20% = 1 sigma

    # Determine direction
    if zscore >= 1.5:
        direction = Direction.SHORT
        confidence = min(0.9, 0.5 + zscore * 0.1)
        thesis = f"Vehicle inventory {pct_change:.1f}% above baseline - potential demand weakness"
    elif zscore <= -1.5:
        direction = Direction.LONG
        confidence = min(0.9, 0.5 + abs(zscore) * 0.1)
        thesis = f"Vehicle inventory {pct_change:.1f}% below baseline - potential supply constraint"
    else:
        direction = Direction.NEUTRAL
        confidence = 0.3
        thesis = f"Vehicle inventory near baseline ({pct_change:.1f}% change)"

    return ResearchSignal(
        strategy="auto_inventory",
        timestamp=target_date,
        region=region_id,
        direction=direction,
        strength=abs(zscore),
        confidence=confidence,
        data_quality=coverage_ratio,
        sample_count=1,
        coverage_ratio=coverage_ratio,
        thesis=thesis,
        raw_value=float(inventory_count),
        baseline_value=float(baseline_count),
        percentile_rank=0.5 + zscore * 0.1,  # Approximate
    )


__all__ = [
    "AutoInventoryStrategy",
    "create_auto_inventory_signal",
]
