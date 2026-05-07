"""
Base feature engineering classes.

All feature modules should inherit from BaseFeature to ensure
consistent interface and behavior.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
from enum import Enum

import pandas as pd
import numpy as np


class FeatureTransform(Enum):
    """Types of feature transformations."""
    NONE = "none"
    LOG = "log"
    DIFF = "diff"
    PCT_CHANGE = "pct_change"
    ZSCORE = "zscore"
    ROBUST_ZSCORE = "robust_zscore"
    RANK = "rank"
    WINSORIZE = "winsorize"
    MIN_MAX = "min_max"


@dataclass
class FeatureOutput:
    """
    Standard output structure for features.

    All feature modules should return data in this format
    for consistency across the system.
    """
    # Core feature data
    features: pd.DataFrame
    feature_name: str

    # Metadata
    computed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    transform: FeatureTransform = FeatureTransform.NONE

    # Quality metrics
    data_quality_score: float = 1.0  # 0-1
    coverage_ratio: float = 1.0  # 0-1
    sample_count: int = 0

    # Configuration used
    config: Dict[str, Any] = field(default_factory=dict)

    # Warnings/errors
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "computed_at": self.computed_at,
            "transform": self.transform.value,
            "data_quality_score": self.data_quality_score,
            "coverage_ratio": self.coverage_ratio,
            "sample_count": self.sample_count,
            "config": self.config,
            "warnings": self.warnings,
            "errors": self.errors,
            "shape": self.features.shape if self.features is not None else (0, 0),
        }

    def merge(self, other: "FeatureOutput") -> "FeatureOutput":
        """
        Merge with another FeatureOutput.

        Combines feature DataFrames and aggregates metadata.
        """
        if self.features is None:
            features = other.features
        elif other.features is None:
            features = self.features
        else:
            # Merge on index (date)
            features = self.features.merge(
                other.features,
                left_index=True,
                right_index=True,
                how="outer"
            )

        # Aggregate quality scores (weighted average by sample count)
        total_samples = self.sample_count + other.sample_count
        if total_samples > 0:
            quality_score = (
                self.data_quality_score * self.sample_count +
                other.data_quality_score * other.sample_count
            ) / total_samples
        else:
            quality_score = (self.data_quality_score + other.data_quality_score) / 2

        return FeatureOutput(
            features=features,
            feature_name=f"{self.feature_name}_{other.feature_name}",
            computed_at=datetime.now().isoformat(),
            data_quality_score=quality_score,
            coverage_ratio=(self.coverage_ratio + other.coverage_ratio) / 2,
            sample_count=total_samples,
            warnings=self.warnings + other.warnings,
            errors=self.errors + other.errors,
        )


class BaseFeature(ABC):
    """
    Abstract base class for all feature modules.

    All feature implementations should:
    1. Inherit from BaseFeature
    2. Implement fit(), transform(), and compute() methods
    3. Return FeatureOutput with proper metadata
    4. Be stateless (don't store data between calls)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature module.

        Args:
            config: Optional configuration dict
        """
        self.config = config or {}
        self._fitted_params: Dict[str, Any] = {}

    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> "BaseFeature":
        """
        Fit the feature transformer to data.

        Computes any statistics needed for transformation
        (e.g., mean, std for zscore, percentiles for seasonal baselines).

        Args:
            data: Input DataFrame
            target_column: Column to use for fitting (optional)

        Returns:
            Self (for method chaining)
        """
        pass

    @abstractmethod
    def transform(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> FeatureOutput:
        """
        Transform data and compute features.

        Args:
            data: Input DataFrame
            target_column: Column to transform (optional)

        Returns:
            FeatureOutput with computed features
        """
        pass

    def compute(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> FeatureOutput:
        """
        Fit and transform in one step.

        Args:
            data: Input DataFrame
            target_column: Column to compute features from

        Returns:
            FeatureOutput with computed features
        """
        self.fit(data, target_column)
        return self.transform(data, target_column)

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration."""
        return self.config.copy()

    def set_config(self, **kwargs) -> "BaseFeature":
        """Update configuration."""
        self.config.update(kwargs)
        return self


@dataclass
class FeatureConfig:
    """
    Default configuration for feature computation.

    This can be extended by specific feature modules.
    """
    # Transform options
    transform: FeatureTransform = FeatureTransform.NONE
    log_epsilon: float = 1e-6  # Small value to avoid log(0)

    # Z-score options
    zscore_window: int = 20  # Rolling window for z-score
    zscore_min_periods: int = 5

    # Robust z-score (using median and MAD)
    robust_zscore_mad_multiplier: float = 1.4826  # MAD to std conversion

    # Rank options
    rank_method: str = "average"  # average, min, max, dense, ordinal
    ascending: bool = True
    pct: bool = True  # Return percentile ranks (0-1)

    # Winsorize options
    winsorize_limits: tuple = (0.05, 0.05)  # Lower and upper percentiles
    winsorize_inclusive: tuple = (True, True)

    # Min-max scaling
    min_max_range: tuple = (0, 1)  # Target range

    # Seasonality options
    seasonal_periods: List[int] = None  # Periods to check (e.g., [7, 30, 365])
    seasonal_method: str = "rolling"  # rolling, decomposition

    # Quality scoring
    quality_weights: Dict[str, float] = None  # Weights for quality components
    min_quality_score: float = 0.0  # Minimum acceptable quality

    def __post_init__(self):
        if self.seasonal_periods is None:
            self.seasonal_periods = [7, 30, 90, 365]  # Default: weekly, monthly, quarterly, yearly

        if self.quality_weights is None:
            self.quality_weights = {
                "coverage": 0.3,
                "recency": 0.2,
                "sample_count": 0.2,
                "variance": 0.15,
                "outlier_ratio": 0.15,
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "transform": self.transform.value if isinstance(self.transform, FeatureTransform) else self.transform,
            "log_epsilon": self.log_epsilon,
            "zscore_window": self.zscore_window,
            "zscore_min_periods": self.zscore_min_periods,
            "robust_zscore_mad_multiplier": self.robust_zscore_mad_multiplier,
            "rank_method": self.rank_method,
            "ascending": self.ascending,
            "pct": self.pct,
            "winsorize_limits": self.winsorize_limits,
            "winsorize_inclusive": self.winsorize_inclusive,
            "min_max_range": self.min_max_range,
            "seasonal_periods": self.seasonal_periods,
            "seasonal_method": self.seasonal_method,
            "quality_weights": self.quality_weights,
            "min_quality_score": self.min_quality_score,
        }


def validate_feature_input(
    data: pd.DataFrame,
    require_columns: Optional[List[str]] = None,
    min_rows: int = 1,
) -> None:
    """
    Validate input data for feature computation.

    Args:
        data: Input DataFrame
        require_columns: Columns that must be present
        min_rows: Minimum number of rows required

    Raises:
        ValueError: If validation fails
    """
    if data is None or data.empty:
        raise ValueError("Input data cannot be None or empty")

    if len(data) < min_rows:
        raise ValueError(f"Input data must have at least {min_rows} rows, got {len(data)}")

    if require_columns:
        missing = [col for col in require_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


def ensure_date_index(
    data: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:
    """
    Ensure DataFrame has a DatetimeIndex.

    Args:
        data: Input DataFrame
        date_column: Column to use as index

    Returns:
        DataFrame with DatetimeIndex
    """
    df = data.copy()

    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df


__all__ = [
    "BaseFeature",
    "FeatureOutput",
    "FeatureConfig",
    "FeatureTransform",
    "validate_feature_input",
    "ensure_date_index",
]
