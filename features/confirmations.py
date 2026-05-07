"""
Confirmation features.

Multi-source validation to upgrade signal confidence.

A signal is more reliable when confirmed by independent sources:
- Price confirmation: Price moving in expected direction
- Volume confirmation: Volume supporting the move
- Macro confirmation: Macro data aligning
- Alt-data confirmation: Other alternative data agreeing

Example:
    Satellite detects low inventory → Price should rise → If price rising, CONFIRMED
"""
from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

from features.base import BaseFeature, FeatureOutput, validate_feature_input


class ConfirmationType(Enum):
    """Types of confirmations."""
    PRICE = "price"
    VOLUME = "volume"
    MACRO = "macro"
    ALT_DATA = "alt_data"
    NEWS = "news"
    TECHNICAL = "technical"


@dataclass
class ConfirmationResult:
    """Result of a confirmation check."""
    type: ConfirmationType
    confirmed: bool
    strength: float  # 0-1, how strong the confirmation is
    description: str
    raw_value: Optional[float] = None


class ConfirmationFeature(BaseFeature):
    """
    Compute confirmation scores for signals.

    Confirmations increase confidence that a signal will translate
    into a profitable trade.

    Example:
        >>> from features.confirmations import ConfirmationFeature
        >>> confirm = ConfirmationFeature()
        >>> output = confirm.compute(
        ...     signal_df,
        ...     price_df,
        ...     signal_direction="long"
        ... )
        >>> confirmation_score = output.features["confirmation_score"]
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ConfirmationFeature.

        Args:
            config: Optional configuration with thresholds
        """
        super().__init__(config)

        # Default thresholds
        self.price_change_threshold = self.config.get("price_change_threshold", 0.02)  # 2%
        self.volume_increase_threshold = self.config.get("volume_increase_threshold", 1.2)  # 20%
        self.confirmation_window = self.config.get("confirmation_window", 5)  # days

    def fit(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> "ConfirmationFeature":
        """
        No fitting needed for confirmation.

        Args:
            data: Input DataFrame
            target_column: Not used

        Returns:
            Self
        """
        return self

    def transform(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> FeatureOutput:
        """
        Compute confirmation score.

        Args:
            data: DataFrame with signal data
            target_column: Not used (use check_price_confirmation etc.)

        Returns:
            FeatureOutput with confirmation metrics
        """
        validate_feature_input(data, min_rows=1)

        # Create output DataFrame
        output_df = pd.DataFrame(index=data.index)

        # Default columns (will be filled by specific confirmation methods)
        output_df["confirmation_count"] = 0
        output_df["confirmation_score"] = 0.0
        output_df["price_confirmation"] = False
        output_df["volume_confirmation"] = False

        return FeatureOutput(
            features=output_df,
            feature_name="confirmation",
            config=self.get_config(),
        )

    def check_price_confirmation(
        self,
        signal_df: pd.DataFrame,
        price_df: pd.DataFrame,
        signal_column: str = "direction",
        expected_direction: str = "long",
    ) -> FeatureOutput:
        """
        Check if price is moving in expected direction.

        Args:
            signal_df: DataFrame with signals
            price_df: DataFrame with price data
            signal_column: Column with signal direction
            expected_direction: Expected price direction ("long" or "short")

        Returns:
            FeatureOutput with price confirmation
        """
        validate_feature_input(signal_df, min_rows=1)
        validate_feature_input(price_df, min_rows=1)

        # Merge on date
        merged = signal_df.merge(price_df, on="date", how="left")

        # Compute price change over confirmation window
        if "close" in merged.columns:
            merged["price_change"] = merged["close"].pct_change()
        elif "price_close" in merged.columns:
            merged["price_change"] = merged["price_close"].pct_change()
        else:
            # No price data available
            output_df = pd.DataFrame(index=signal_df.index)
            output_df["price_confirmation"] = False
            output_df["price_change"] = np.nan
            return FeatureOutput(
                features=output_df,
                feature_name="price_confirmation",
                config=self.get_config(),
            )

        # Check if price moving in expected direction
        if expected_direction.lower() == "long":
            confirmed = merged["price_change"] > self.price_change_threshold
            strength = merged["price_change"].clip(lower=0) / self.price_change_threshold
        else:  # short
            confirmed = merged["price_change"] < -self.price_change_threshold
            strength = (-merged["price_change"]).clip(lower=0) / self.price_change_threshold

        # Create output
        output_df = pd.DataFrame({
            "price_confirmation": confirmed.values,
            "price_change": merged["price_change"].values,
            "price_confirmation_strength": strength.clip(upper=1).values,
        }, index=merged.index)

        return FeatureOutput(
            features=output_df,
            feature_name="price_confirmation",
            data_quality_score=confirmed.sum() / len(confirmed) if len(confirmed) > 0 else 0,
            config=self.get_config(),
        )

    def check_volume_confirmation(
        self,
        signal_df: pd.DataFrame,
        price_df: pd.DataFrame,
        expected_direction: str = "long",
    ) -> FeatureOutput:
        """
        Check if volume supports the move.

        For long signals: Expect high volume on up days
        For short signals: Expect high volume on down days

        Args:
            signal_df: DataFrame with signals
            price_df: DataFrame with OHLCV data
            expected_direction: Expected direction

        Returns:
            FeatureOutput with volume confirmation
        """
        validate_feature_input(signal_df, min_rows=1)
        validate_feature_input(price_df, min_rows=1)

        # Merge on date
        merged = signal_df.merge(price_df, on="date", how="left")

        # Check volume column
        if "volume" not in merged.columns:
            output_df = pd.DataFrame(index=signal_df.index)
            output_df["volume_confirmation"] = False
            return FeatureOutput(
                features=output_df,
                feature_name="volume_confirmation",
                config=self.get_config(),
            )

        # Compute volume average
        avg_volume = merged["volume"].rolling(window=20, min_periods=5).mean()

        # Check for above-average volume
        high_volume = merged["volume"] > avg_volume * self.volume_increase_threshold

        output_df = pd.DataFrame({
            "volume_confirmation": high_volume.values,
            "volume_ratio": (merged["volume"] / avg_volume).values,
        }, index=merged.index)

        return FeatureOutput(
            features=output_df,
            feature_name="volume_confirmation",
            data_quality_score=high_volume.sum() / len(high_volume) if len(high_volume) > 0 else 0,
            config=self.get_config(),
        )

    def combine_confirmations(
        self,
        confirmations: List[FeatureOutput],
        weights: Optional[Dict[str, float]] = None,
    ) -> FeatureOutput:
        """
        Combine multiple confirmation sources into a single score.

        Args:
            confirmations: List of FeatureOutput with confirmation results
            weights: Optional weights for each confirmation type

        Returns:
            FeatureOutput with combined confirmation score
        """
        if not confirmations:
            return FeatureOutput(
                features=pd.DataFrame({"confirmation_count": 0}, index=[0]),
                feature_name="combined_confirmation",
            )

        # Merge all confirmation DataFrames
        merged = confirmations[0].features
        for conf in confirmations[1:]:
            merged = merged.merge(conf.features, left_index=True, right_index=True, how="outer")

        # Count confirmations
        confirmation_cols = [col for col in merged.columns if col.endswith("_confirmation")]
        merged["confirmation_count"] = merged[confirmation_cols].fillna(False).sum(axis=1)

        # Compute weighted score
        if weights:
            score = 0.0
            for conf_type, weight in weights.items():
                col = f"{conf_type}_confirmation"
                if col in merged.columns:
                    strength_col = f"{conf_type}_confirmation_strength"
                    if strength_col in merged.columns:
                        score += merged[col].fillna(False) * merged[strength_col].fillna(0) * weight
                    else:
                        score += merged[col].fillna(False) * weight
            merged["confirmation_score"] = score / sum(weights.values())
        else:
            # Equal weights
            merged["confirmation_score"] = merged["confirmation_count"] / len(confirmation_cols)

        return FeatureOutput(
            features=merged,
            feature_name="combined_confirmation",
            data_quality_score=merged["confirmation_count"].max() / len(confirmation_cols),
            config=self.get_config(),
        )

    def get_confirmation_summary(
        self,
        confirmations: List[ConfirmationResult],
    ) -> Dict[str, Any]:
        """
        Get a summary of confirmation results.

        Args:
            confirmations: List of ConfirmationResult objects

        Returns:
            Summary dict
        """
        confirmed = [c for c in confirmations if c.confirmed]
        total_strength = sum(c.strength for c in confirmations)
        confirmed_strength = sum(c.strength for c in confirmed)

        return {
            "total_confirmations": len(confirmations),
            "confirmed_count": len(confirmed),
            "confirmation_rate": len(confirmed) / len(confirmations) if confirmations else 0,
            "total_strength": total_strength,
            "confirmed_strength": confirmed_strength,
            "avg_strength": total_strength / len(confirmations) if confirmations else 0,
            "confirmations": [
                {
                    "type": c.type.value,
                    "confirmed": c.confirmed,
                    "strength": c.strength,
                    "description": c.description,
                }
                for c in confirmations
            ],
        }


def compute_price_confirmation(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    signal_direction: str = "long",
    threshold: float = 0.02,
) -> pd.Series:
    """
    Convenience function for price confirmation.

    Args:
        signal_df: DataFrame with signals
        price_df: DataFrame with price data
        signal_direction: Expected direction ("long" or "short")
        threshold: Price change threshold for confirmation

    Returns:
        Boolean series indicating price confirmation
    """
    confirm = ConfirmationFeature(config={"price_change_threshold": threshold})
    output = confirm.check_price_confirmation(signal_df, price_df, expected_direction=signal_direction)
    return output.features["price_confirmation"]


__all__ = [
    "ConfirmationFeature",
    "ConfirmationType",
    "ConfirmationResult",
    "compute_price_confirmation",
]
