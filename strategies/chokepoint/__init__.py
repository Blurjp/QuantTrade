"""
Chokepoint Strategy.

Monitors maritime chokepoint throughput (Hormuz, Panama, Suez).

High throughput = bullish for shipping/oil demand
Low throughput = bearish (disruption risk)

Trade mapping:
- High throughput → Long shipping stocks
- Low throughput → Short shippers, Long oil (supply risk)
"""
from __future__ import annotations

from typing import Optional, Dict, Any, List
from datetime import date, datetime

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


class ChokepointStrategy(BaseStrategy):
    """
    Maritime chokepoint monitoring strategy.

    Uses SAR imagery to count ships transiting critical chokepoints.

    Signal interpretation:
    - High throughput → BULLISH for global trade/shipping
    - Low throughput → BEARISH for shipping, potential supply disruption
    """

    # Strategy identity
    name: str = "chokepoint"
    version: str = "1.0.0"
    description: str = "Monitor maritime chokepoint throughput via SAR imagery"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the strategy."""
        self.config = StrategyConfig()
        if config:
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        # Trading parameters
        self.shipping_tickers = ["DRYS", "SBLK", "TNK", "NAT"]  # Shipping stocks
        self.oil_tickers = ["USO", "XLE", "XOM", "CVX"]  # Oil/energy
        self.default_horizon_days = 14
        self.default_size_pct = 0.02
        self.stop_loss_pct = 0.06
        self.take_profit_pct = 0.12

    def load_inputs(
        self,
        start_date: str,
        end_date: str,
        region: Optional[str] = None,
        output_base: str = "outputs",
    ) -> pd.DataFrame:
        """
        Load historical chokepoint throughput data.

        Args:
            start_date: Start date
            end_date: End date
            region: Region identifier (e.g., "hormuz", "panama_canal")
            output_base: Output directory

        Returns:
            DataFrame with throughput data
        """
        from pipeline.detection_multi import run_detection
        from pipeline.regions import load_registry
        from pathlib import Path

        registry = load_registry()
        region_config = registry.get("regions", {}).get(region, {})

        if not region_config:
            return pd.DataFrame(columns=[
                "date", "region", "detections", "throughput_estimate",
                "coverage_ratio", "baseline_throughput"
            ])

        aoi_file = region_config.get("aoi_file")
        if not aoi_file or not Path(aoi_file).exists():
            return pd.DataFrame(columns=[
                "date", "region", "detections", "throughput_estimate",
                "coverage_ratio", "baseline_throughput"
            ])

        try:
            detection_result = run_detection(
                monitoring_type="chokepoint",
                aoi_path=aoi_file,
                target_date=end_date,
                output_base=output_base,
            )

            if hasattr(detection_result, "to_dict"):
                result_dict = detection_result.to_dict()
            else:
                result_dict = detection_result

            details = result_dict.get("details", [])
            if not details:
                return pd.DataFrame(columns=[
                    "date", "region", "detections", "throughput_estimate",
                    "coverage_ratio", "baseline_throughput"
                ])

            df = pd.DataFrame(details)
            df["region"] = region

            # Detect column is actually detections count
            if "detect" in df.columns:
                df["detections"] = df["detect"]

            return df

        except Exception as e:
            print(f"Error loading chokepoint data for {region}: {e}")
            return pd.DataFrame(columns=[
                "date", "region", "detections", "throughput_estimate",
                "coverage_ratio", "baseline_throughput"
            ])

    def build_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features from throughput data.

        Features:
        - throughput_ma: Moving average of throughput
        - pct_change: % change from baseline
        - zscore: Z-score of throughput
        - trend: Trend direction (increasing/decreasing)

        Args:
            raw_df: Raw detection data

        Returns:
            DataFrame with features
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

        # Use detections as throughput proxy
        if "detections" in df.columns:
            throughput = df["detections"]

            # Moving average baseline
            df["baseline_throughput"] = throughput.rolling(window=14, min_periods=3).median()

            # Percent change
            df["pct_change"] = (
                (throughput - df["baseline_throughput"]) / df["baseline_throughput"].replace(0, np.nan)
            ) * 100

            # Z-score
            throughput_mean = throughput.rolling(window=14, min_periods=3).mean()
            throughput_std = throughput.rolling(window=14, min_periods=3).std()
            df["zscore"] = (throughput - throughput_mean) / throughput_std.replace(0, np.nan)

        # Quality features
        if "coverage_ratio" in df.columns:
            df["data_quality"] = df["coverage_ratio"]
        else:
            df["data_quality"] = 0.5

        df["sample_count"] = 1

        return df

    def generate_signal(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from throughput features.

        Signal logic:
        - zscore > 1.5 → High throughput → BULLISH shipping
        - zscore < -1.5 → Low throughput → BEARISH shipping

        Args:
            feature_df: Feature data

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

        if "zscore" in df.columns:
            # High throughput = bullish
            df.loc[df["zscore"] >= 1.5, "signal_direction"] = "long"
            df.loc[df["zscore"] <= -1.5, "signal_direction"] = "short"

            df["signal_strength"] = df["zscore"].abs()

            df.loc[df["signal_direction"] == "long", "signal"] = df["zscore"].apply(
                lambda x: f"High throughput (+{x:.1f}σ) - Strong shipping demand"
            )
            df.loc[df["signal_direction"] == "short", "signal"] = df["zscore"].apply(
                lambda x: f"Low throughput ({x:.1f}σ) - Shipping disruption risk"
            )
            df.loc[df["signal_direction"] == "neutral", "signal"] = "Throughput near baseline"

        # Get latest signal
        latest = df.iloc[-1].to_dict()
        return pd.DataFrame([latest])

    def estimate_confidence(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        """Estimate signal confidence."""
        if signal_df.empty:
            return signal_df

        df = signal_df.copy()

        df["confidence"] = 0.5
        df["data_quality"] = df.get("data_quality", 0.5)

        if "signal_strength" in df.columns:
            df["confidence"] = (df["signal_strength"].clip(0, 2) / 2) * df["data_quality"]

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

        # For high throughput (long): Buy shipping stocks
        if direction == Direction.LONG:
            for ticker in self.shipping_tickers:
                candidates.append(TradeCandidate(
                    strategy=self.name,
                    timestamp=datetime.now().isoformat(),
                    ticker=ticker,
                    asset_type=AssetType.EQUITY,
                    direction=direction,
                    horizon_days=self.default_horizon_days,
                    size_pct=self.default_size_pct,
                    stop_loss_pct=self.stop_loss_pct,
                    take_profit_pct=self.take_profit_pct,
                    rationale=f"Strong throughput at {region} indicates healthy shipping demand",
                    source_signals=[f"{region}_{datetime.now().strftime('%Y-%m-%d')}"],
                    probability=float(df.get("confidence", 0.5)),
                ))

        # For low throughput (short): Short shipping, consider oil long
        elif direction == Direction.SHORT:
            for ticker in self.shipping_tickers:
                candidates.append(TradeCandidate(
                    strategy=self.name,
                    timestamp=datetime.now().isoformat(),
                    ticker=ticker,
                    asset_type=AssetType.EQUITY,
                    direction=direction,
                    horizon_days=self.default_horizon_days,
                    size_pct=self.default_size_pct,
                    stop_loss_pct=self.stop_loss_pct,
                    take_profit_pct=self.take_profit_pct,
                    rationale=f"Low throughput at {region} suggests shipping weakness",
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


def create_chokepoint_signal(
    region_id: str,
    target_date: str,
    detections: int,
    baseline_throughput: float,
    coverage_ratio: float = 1.0,
) -> ResearchSignal:
    """
    Create a research signal from chokepoint data.

    Args:
        region_id: Region identifier
        target_date: Date of observation
        detections: Ship detections count
        baseline_throughput: Baseline throughput
        coverage_ratio: Detection coverage

    Returns:
        ResearchSignal object
    """
    pct_change = (detections - baseline_throughput) / baseline_throughput * 100
    zscore = pct_change / 25  # Assume 25% = 1 sigma

    if zscore >= 1.5:
        direction = Direction.LONG
        confidence = min(0.9, 0.5 + zscore * 0.1)
        thesis = f"Ship throughput {pct_change:.1f}% above baseline - Strong shipping demand"
    elif zscore <= -1.5:
        direction = Direction.SHORT
        confidence = min(0.9, 0.5 + abs(zscore) * 0.1)
        thesis = f"Ship throughput {pct_change:.1f}% below baseline - Shipping disruption risk"
    else:
        direction = Direction.NEUTRAL
        confidence = 0.3
        thesis = f"Ship throughput near baseline ({pct_change:.1f}% change)"

    return ResearchSignal(
        strategy="chokepoint",
        timestamp=target_date,
        region=region_id,
        direction=direction,
        strength=abs(zscore),
        confidence=confidence,
        data_quality=coverage_ratio,
        sample_count=1,
        coverage_ratio=coverage_ratio,
        thesis=thesis,
        raw_value=float(detections),
        baseline_value=float(baseline_throughput),
    )


__all__ = [
    "ChokepointStrategy",
    "create_chokepoint_signal",
]
