"""
Normalization features.

Provides various normalization methods to make signals comparable
across regions, assets, and time periods.

Normalization methods:
- Percent change
- Z-score (standard score)
- Robust z-score (median-based)
- Cross-sectional rank
- Winsorized values
- Min-max scaling
"""
from __future__ import annotations

from typing import Optional, Dict, Any, List, Union
from enum import Enum

import pandas as pd
import numpy as np

from features.base import BaseFeature, FeatureOutput, FeatureTransform, validate_feature_input


class NormalizationMethod(Enum):
    """Available normalization methods."""
    PCT_CHANGE = "pct_change"
    ZSCORE = "zscore"
    ROBUST_ZSCORE = "robust_zscore"
    RANK = "rank"
    WINSORIZE = "winsorize"
    MIN_MAX = "min_max"
    LOG = "log"
    DIFF = "diff"


class NormalizationFeature(BaseFeature):
    """
    Normalize features for comparability.

    Different regions have different scales:
    - Texas auto inventory: +130% (large numbers)
    - Detroit auto inventory: +7% (smaller base)
    - Chokepoint throughput: -40% (negative direction)

    Normalization puts everything on a common scale.

    Example:
        >>> from features.normalization import NormalizationFeature
        >>> norm = NormalizationFeature(method="zscore")
        >>> output = norm.compute(df, target_column="value")
        >>> normalized = output.features["value_normalized"]
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        method: Union[str, NormalizationMethod] = NormalizationMethod.ZSCORE,
        window: Optional[int] = None,
        min_periods: int = 1,
    ):
        """
        Initialize NormalizationFeature.

        Args:
            config: Optional configuration
            method: Normalization method to use
            window: Rolling window for statistics (None = use full history)
            min_periods: Minimum periods for rolling computation
        """
        super().__init__(config)
        self.method = method if isinstance(method, NormalizationMethod) else NormalizationMethod(method)
        self.window = window
        self.min_periods = min_periods

        # Store fitted parameters
        self._fitted_params: Dict[str, Any] = {}

    def fit(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> "NormalizationFeature":
        """
        Fit normalization parameters to data.

        Computes statistics (mean, std, median, quantiles) needed
        for normalization.

        Args:
            data: Input DataFrame
            target_column: Column to fit on

        Returns:
            Self
        """
        validate_feature_input(data, min_rows=1)

        if target_column is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            target_column = numeric_cols[0] if numeric_cols else None

        if target_column is None or target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        series = data[target_column].dropna()
        if len(series) == 0:
            return self

        # Compute statistics based on method
        if self.method == NormalizationMethod.ZSCORE:
            self._fitted_params["mean"] = series.mean()
            self._fitted_params["std"] = series.std()

        elif self.method == NormalizationMethod.ROBUST_ZSCORE:
            self._fitted_params["median"] = series.median()
            self._fitted_params["mad"] = (series - series.median()).abs().median()

        elif self.method == NormalizationMethod.MIN_MAX:
            self._fitted_params["min"] = series.min()
            self._fitted_params["max"] = series.max()

        elif self.method == NormalizationMethod.WINSORIZE:
            self._fitted_params["q_low"] = series.quantile(0.05)
            self._fitted_params["q_high"] = series.quantile(0.95)

        elif self.method == NormalizationMethod.RANK:
            self._fitted_params["rank_method"] = "average"

        return self

    def transform(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> FeatureOutput:
        """
        Apply normalization to data.

        Args:
            data: Input DataFrame
            target_column: Column to normalize

        Returns:
            FeatureOutput with normalized values
        """
        validate_feature_input(data, min_rows=1)

        if target_column is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            target_column = numeric_cols[0] if numeric_cols else None

        if target_column is None or target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        df = data.copy()
        series = df[target_column]

        # Apply normalization
        if self.method == NormalizationMethod.PCT_CHANGE:
            normalized = self._pct_change(series)

        elif self.method == NormalizationMethod.ZSCORE:
            normalized = self._zscore(series)

        elif self.method == NormalizationMethod.ROBUST_ZSCORE:
            normalized = self._robust_zscore(series)

        elif self.method == NormalizationMethod.RANK:
            normalized = self._rank(series)

        elif self.method == NormalizationMethod.WINSORIZE:
            normalized = self._winsorize(series)

        elif self.method == NormalizationMethod.MIN_MAX:
            normalized = self._min_max(series)

        elif self.method == NormalizationMethod.LOG:
            normalized = self._log(series)

        elif self.method == NormalizationMethod.DIFF:
            normalized = self._diff(series)

        else:
            normalized = series.copy()

        # Create output DataFrame
        output_df = pd.DataFrame(index=df.index)
        output_df[f"{target_column}_normalized"] = normalized
        if "date" in df.columns:
            output_df["date"] = df["date"]

        return FeatureOutput(
            features=output_df,
            feature_name=f"normalized_{self.method.value}",
            transform=FeatureTransform(self.method.value),
            data_quality_score=normalized.notna().sum() / len(normalized),
            coverage_ratio=normalized.notna().sum() / len(normalized),
            sample_count=len(normalized.dropna()),
            config=self.get_config(),
        )

    def _pct_change(self, series: pd.Series) -> pd.Series:
        """Compute percent change."""
        return series.pct_change()

    def _zscore(self, series: pd.Series) -> pd.Series:
        """Compute z-score normalization."""
        if self.window is None:
            # Use global parameters from fit
            mean = self._fitted_params.get("mean", series.mean())
            std = self._fitted_params.get("std", series.std())
        else:
            # Use rolling window
            mean = series.rolling(window=self.window, min_periods=self.min_periods).mean()
            std = series.rolling(window=self.window, min_periods=self.min_periods).std()

        return (series - mean) / std.replace(0, np.nan)

    def _robust_zscore(self, series: pd.Series) -> pd.Series:
        """Compute robust z-score using median and MAD."""
        mad_multiplier = self.config.get("robust_zscore_mad_multiplier", 1.4826)

        if self.window is None:
            median = self._fitted_params.get("median", series.median())
            mad = self._fitted_params.get("mad", (series - series.median()).abs().median())
        else:
            median = series.rolling(window=self.window, min_periods=self.min_periods).median()
            mad = (series - median).abs().rolling(window=self.window, min_periods=self.min_periods).median()

        std_estimate = mad * mad_multiplier
        return (series - median) / std_estimate.replace(0, np.nan)

    def _rank(self, series: pd.Series) -> pd.Series:
        """Compute cross-sectional rank (0-1)."""
        pct = self.config.get("pct", True)
        method = self.config.get("rank_method", "average")

        if pct:
            return series.rank(method=method) / len(series)
        else:
            return series.rank(method=method)

    def _winsorize(self, series: pd.Series) -> pd.Series:
        """Winsorize values (clip outliers)."""
        limits = self.config.get("winsorize_limits", (0.05, 0.05))

        if self.window is None:
            q_low = self._fitted_params.get("q_low", series.quantile(limits[0]))
            q_high = self._fitted_params.get("q_high", series.quantile(1 - limits[1]))
        else:
            q_low = series.rolling(window=self.window, min_periods=self.min_periods).quantile(limits[0])
            q_high = series.rolling(window=self.window, min_periods=self.min_periods).quantile(1 - limits[1])

        return series.clip(lower=q_low, upper=q_high)

    def _min_max(self, series: pd.Series) -> pd.Series:
        """Min-max scaling to [0, 1] range."""
        target_range = self.config.get("min_max_range", (0, 1))

        if self.window is None:
            min_val = self._fitted_params.get("min", series.min())
            max_val = self._fitted_params.get("max", series.max())
        else:
            min_val = series.rolling(window=self.window, min_periods=self.min_periods).min()
            max_val = series.rolling(window=self.window, min_periods=self.min_periods).max()

        scaled = (series - min_val) / (max_val - min_val).replace(0, np.nan)
        range_size = target_range[1] - target_range[0]
        return scaled * range_size + target_range[0]

    def _log(self, series: pd.Series) -> pd.Series:
        """Log transform."""
        epsilon = self.config.get("log_epsilon", 1e-6)
        return np.log(series.clip(lower=epsilon))

    def _diff(self, series: pd.Series) -> pd.Series:
        """First difference."""
        return series.diff()


def normalize(
    data: pd.DataFrame,
    target_column: str,
    method: Union[str, NormalizationMethod] = "zscore",
) -> pd.Series:
    """
    Convenience function to normalize a column.

    Args:
        data: Input DataFrame
        target_column: Column to normalize
        method: Normalization method

    Returns:
        Normalized series
    """
    norm = NormalizationFeature(method=method)
    output = norm.compute(data, target_column)
    return output.features[f"{target_column}_normalized"]


def cross_sectional_normalize(
    data: pd.DataFrame,
    target_column: str,
    group_column: Optional[str] = None,
) -> pd.Series:
    """
    Normalize within groups (cross-sectional).

    Useful for comparing values across regions at the same point in time.

    Args:
        data: Input DataFrame
        target_column: Column to normalize
        group_column: Column to group by (default: use date)

    Returns:
        Cross-sectionally normalized series
    """
    if group_column is None and "date" in data.columns:
        group_column = "date"

    if group_column is None:
        return normalize(data, target_column, method="rank")

    def normalize_group(group):
        norm = NormalizationFeature(method=NormalizationMethod.ZSCORE)
        output = norm.compute(group, target_column)
        return output.features[f"{target_column}_normalized"]

    return data.groupby(group_column, group_keys=False).apply(
        lambda g: normalize_group(g)
    ).reset_index(level=0, drop=True)


__all__ = [
    "NormalizationFeature",
    "NormalizationMethod",
    "normalize",
    "cross_sectional_normalize",
]
