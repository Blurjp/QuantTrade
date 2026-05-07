"""
Data quality scoring features.

Quantifies the quality of input data to ensure signals are
generated from reliable data sources.

Quality components:
- Coverage: Spatial/temporal coverage ratio
- Recency: How recent the data is
- Sample count: Number of observations
- Variance: Data variability (flat data = low quality)
- Outlier ratio: Proportion of extreme values
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from features.base import BaseFeature, FeatureOutput, FeatureConfig, validate_feature_input


@dataclass
class QualityMetrics:
    """
    Individual quality component scores.
    """
    coverage_score: float = 0.0  # 0-1
    recency_score: float = 0.0  # 0-1
    sample_count_score: float = 0.0  # 0-1
    variance_score: float = 0.0  # 0-1
    outlier_score: float = 0.0  # 0-1
    overall_quality: float = 0.0  # 0-1 (weighted average)

    # Raw values for transparency
    coverage_ratio: float = 0.0
    days_since_last: int = 0
    sample_count: int = 0
    variance: float = 0.0
    outlier_ratio: float = 0.0

    # Metadata
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_quality": self.overall_quality,
            "coverage_score": self.coverage_score,
            "recency_score": self.recency_score,
            "sample_count_score": self.sample_count_score,
            "variance_score": self.variance_score,
            "outlier_score": self.outlier_score,
            "coverage_ratio": self.coverage_ratio,
            "days_since_last": self.days_since_last,
            "sample_count": self.sample_count,
            "variance": self.variance,
            "outlier_ratio": self.outlier_ratio,
            "warnings": self.warnings,
        }


class QualityFeature(BaseFeature):
    """
    Compute data quality scores.

    Quality is a weighted combination of:
    1. Coverage: How much of the expected data is present
    2. Recency: How recent the last observation is
    3. Sample count: Number of observations (more is better)
    4. Variance: Data variability (some variance is good)
    5. Outlier ratio: Proportion of extreme values

    Example:
        >>> from features.quality import QualityFeature
        >>> quality = QualityFeature()
        >>> output = quality.compute(df, target_column="value")
        >>> print(output.to_dict())
        {'feature_name': 'quality_score', 'data_quality_score': 0.85, ...}
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize QualityFeature.

        Args:
            config: Optional configuration
            weights: Quality component weights (overrides config)
        """
        super().__init__(config)
        self.weights = weights or self.config.get("quality_weights", {})

        # Default weights if not specified
        if not self.weights:
            self.weights = {
                "coverage": 0.3,
                "recency": 0.2,
                "sample_count": 0.2,
                "variance": 0.15,
                "outlier_ratio": 0.15,
            }

    def fit(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> "QualityFeature":
        """
        No fitting needed for quality scoring.

        Args:
            data: Input DataFrame
            target_column: Column to analyze

        Returns:
            Self
        """
        validate_feature_input(data, min_rows=1)
        return self

    def transform(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> FeatureOutput:
        """
        Compute quality scores for input data.

        Args:
            data: Input DataFrame with 'date' column and target column
            target_column: Column to analyze (uses first numeric if None)

        Returns:
            FeatureOutput with quality_score column
        """
        validate_feature_input(data, min_rows=1)

        df = data.copy()

        # Ensure date column exists
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # Determine target column
        if target_column is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric columns found in data")
            target_column = numeric_cols[0]

        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in data")

        # Get date column
        if "date" in df.columns:
            date_col = "date"
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            date_col = df.columns[0]
        else:
            date_col = None

        # Compute quality metrics
        metrics = self._compute_quality_metrics(
            df,
            target_column,
            date_col,
        )

        # Create output DataFrame
        output_df = pd.DataFrame({
            "quality_score": [metrics.overall_quality],
            "coverage_ratio": [metrics.coverage_ratio],
            "sample_count": [metrics.sample_count],
            "recency_days": [metrics.days_since_last],
        })

        if date_col and date_col in df.columns:
            output_df["date"] = df[date_col].max()

        return FeatureOutput(
            features=output_df,
            feature_name="quality_score",
            data_quality_score=metrics.overall_quality,
            coverage_ratio=metrics.coverage_ratio,
            sample_count=metrics.sample_count,
            config=self.get_config(),
            warnings=metrics.warnings,
        )

    def _compute_quality_metrics(
        self,
        df: pd.DataFrame,
        target_column: str,
        date_column: Optional[str],
    ) -> QualityMetrics:
        """Compute all quality components."""
        metrics = QualityMetrics()

        # Sample count
        metrics.sample_count = len(df)
        metrics.sample_count_score = self._score_sample_count(len(df))

        # Coverage (non-null ratio)
        non_null = df[target_column].notna().sum()
        metrics.coverage_ratio = non_null / len(df) if len(df) > 0 else 0
        metrics.coverage_score = metrics.coverage_ratio  # Direct mapping

        # Recency
        if date_column:
            last_date = pd.to_datetime(df[date_column]).max()
            days_ago = (datetime.now() - last_date).days
            metrics.days_since_last = max(0, days_ago)
            metrics.recency_score = self._score_recency(metrics.days_since_last)
        else:
            metrics.recency_score = 0.5  # Neutral if no date info

        # Variance (some variance is good, flat data is bad)
        values = df[target_column].dropna()
        if len(values) > 1:
            metrics.variance = float(values.var())
            metrics.variance_score = self._score_variance(metrics.variance, values.mean())
        else:
            metrics.variance_score = 0.0

        # Outlier ratio
        metrics.outlier_ratio = self._compute_outlier_ratio(values)
        metrics.outlier_score = 1.0 - metrics.outlier_ratio  # Fewer outliers is better

        # Weighted overall quality
        metrics.overall_quality = (
            metrics.coverage_score * self.weights.get("coverage", 0.3) +
            metrics.recency_score * self.weights.get("recency", 0.2) +
            metrics.sample_count_score * self.weights.get("sample_count", 0.2) +
            metrics.variance_score * self.weights.get("variance", 0.15) +
            metrics.outlier_score * self.weights.get("outlier_ratio", 0.15)
        )

        # Generate warnings
        if metrics.coverage_ratio < 0.5:
            metrics.warnings.append(f"Low coverage: {metrics.coverage_ratio:.1%}")
        if metrics.days_since_last > 30:
            metrics.warnings.append(f"Stale data: {metrics.days_since_last} days old")
        if metrics.sample_count < 5:
            metrics.warnings.append(f"Low sample count: {metrics.sample_count}")
        if metrics.outlier_ratio > 0.2:
            metrics.warnings.append(f"High outlier ratio: {metrics.outlier_ratio:.1%}")

        return metrics

    def _score_sample_count(self, count: int) -> float:
        """Score sample count (0-1)."""
        # Logarithmic scaling: 1=0, 5=0.7, 20+=1.0
        if count <= 1:
            return 0.0
        elif count >= 20:
            return 1.0
        else:
            return np.log(count) / np.log(20)

    def _score_recency(self, days: int) -> float:
        """Score recency (0-1)."""
        # Linear decay: 0 days=1.0, 30+ days=0.0
        if days <= 0:
            return 1.0
        elif days >= 30:
            return 0.0
        else:
            return 1.0 - (days / 30)

    def _score_variance(self, variance: float, mean: float) -> float:
        """Score variance (0-1)."""
        if mean == 0 or variance == 0:
            return 0.0

        # Coefficient of variation
        cv = np.sqrt(variance) / abs(mean)

        # CV of 0.1 to 1.0 is good
        if cv < 0.01:
            return 0.0  # Too flat
        elif 0.01 <= cv <= 1.0:
            return cv  # Linear scaling
        else:
            return 1.0  # High variance is OK

    def _compute_outlier_ratio(self, values: pd.Series) -> float:
        """Compute ratio of outliers using IQR method."""
        if len(values) < 4:
            return 0.0

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            return 0.0

        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr

        outliers = ((values < lower_bound) | (values > upper_bound)).sum()
        return outliers / len(values)


def compute_quality_score(
    data: pd.DataFrame,
    target_column: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Convenience function to compute overall quality score.

    Args:
        data: Input DataFrame
        target_column: Column to analyze
        weights: Optional quality weights

    Returns:
        Overall quality score (0-1)
    """
    quality = QualityFeature(weights=weights)
    output = quality.compute(data, target_column)
    return output.data_quality_score


__all__ = [
    "QualityFeature",
    "QualityMetrics",
    "compute_quality_score",
]
