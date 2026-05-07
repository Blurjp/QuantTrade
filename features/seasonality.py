"""
Seasonality features.

Computes seasonal baselines and adjustments for time series data.

Seasonal patterns are common in:
- Agriculture (planting/harvest cycles)
- Energy (heating/cooling demand)
- Retail (holiday seasons)
- Shipping (weather patterns)

Features:
- Seasonal baseline (rolling window by day-of-week, week-of-year)
- Seasonal deviation from baseline
- Seasonal z-score
- Seasonally-adjusted values
"""
from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from features.base import BaseFeature, FeatureOutput, validate_feature_input, ensure_date_index


class SeasonalityFeature(BaseFeature):
    """
    Compute seasonality features for time series data.

    Supports multiple seasonal periods:
    - Weekly (day-of-week patterns)
    - Monthly (day-of-month patterns)
    - Annual (day-of-year patterns)

    Example:
        >>> from features.seasonality import SeasonalityFeature
        >>> seasonality = SeasonalityFeature(periods=[7, 30, 365])
        >>> output = seasonality.compute(df, target_column="value")
        >>> df_with_seasonal = output.features
        >>> df_with_seasonal[["baseline", "seasonal_deviation", "zscore_seasonal"]]
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        periods: Optional[List[int]] = None,
        method: str = "rolling",
    ):
        """
        Initialize SeasonalityFeature.

        Args:
            config: Optional configuration
            periods: Seasonal periods to detect (default: [7, 30, 365])
            method: Method for computing baseline ('rolling', 'decomposition')
        """
        super().__init__(config)
        self.periods = periods or self.config.get("seasonal_periods", [7, 30, 365])
        self.method = method or self.config.get("seasonal_method", "rolling")

        # Store fitted parameters
        self._fitted_params: Dict[str, Any] = {}

    def fit(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> "SeasonalityFeature":
        """
        Fit seasonal baselines to historical data.

        Computes seasonal statistics for each period.

        Args:
            data: Input DataFrame with date index or column
            target_column: Column to analyze

        Returns:
            Self
        """
        validate_feature_input(data, min_rows=1)

        df = ensure_date_index(data)
        if target_column is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target_column = numeric_cols[0] if numeric_cols else None

        if target_column is None or target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        series = df[target_column].dropna()
        if len(series) == 0:
            return self

        # Compute seasonal baselines for each period
        for period in self.periods:
            if self.method == "rolling":
                baseline = self._compute_rolling_baseline(series, period)
                self._fitted_params[f"baseline_{period}"] = baseline
            else:
                # decomposition method
                trend, seasonal, residual = self._decompose_series(series, period)
                self._fitted_params[f"trend_{period}"] = trend
                self._fitted_params[f"seasonal_{period}"] = seasonal
                self._fitted_params[f"residual_{period}"] = residual

        return self

    def transform(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> FeatureOutput:
        """
        Apply seasonal adjustment to data.

        Args:
            data: Input DataFrame
            target_column: Column to adjust

        Returns:
            FeatureOutput with seasonal features
        """
        validate_feature_input(data, min_rows=1)

        df = ensure_date_index(data)
        if target_column is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target_column = numeric_cols[0] if numeric_cols else None

        if target_column is None or target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        series = df[target_column].copy()

        # Initialize output DataFrame
        output_df = pd.DataFrame(index=df.index)
        output_df["date"] = df.index

        # Use the largest period as the primary seasonal baseline
        primary_period = max(self.periods)
        output_df["baseline"] = np.nan
        output_df["seasonal_deviation"] = np.nan
        output_df["zscore_seasonal"] = np.nan
        output_df["pct_of_baseline"] = np.nan

        if len(series) == 0:
            return FeatureOutput(
                features=output_df,
                feature_name="seasonality",
                config=self.get_config(),
            )

        # Get baseline
        if self.method == "rolling" and f"baseline_{primary_period}" in self._fitted_params:
            baseline = self._fitted_params[f"baseline_{primary_period}"]

            # Align baseline with input data
            common_index = series.index.intersection(baseline.index)
            if len(common_index) > 0:
                baseline_aligned = baseline.loc[common_index]
                series_aligned = series.loc[common_index]

                # Compute seasonal features
                output_df.loc[common_index, "baseline"] = baseline_aligned.values
                output_df.loc[common_index, "seasonal_deviation"] = (
                    series_aligned.values - baseline_aligned.values
                )

                # Z-score relative to seasonal distribution
                if len(baseline_aligned) > 1:
                    std = baseline_aligned.std()
                    if std > 0:
                        output_df.loc[common_index, "zscore_seasonal"] = (
                            output_df.loc[common_index, "seasonal_deviation"] / std
                        )

                # Percent of baseline
                nonzero_baseline = baseline_aligned != 0
                output_df.loc[common_index[nonzero_baseline], "pct_of_baseline"] = (
                    series_aligned.loc[nonzero_baseline].values /
                    baseline_aligned.loc[nonzero_baseline].values
                )

        elif self.method == "decomposition" and f"seasonal_{primary_period}" in self._fitted_params:
            # Use decomposition-based adjustment
            seasonal = self._fitted_params[f"seasonal_{primary_period}"]
            trend = self._fitted_params[f"trend_{primary_period}"]
            residual = self._fitted_params[f"residual_{primary_period}"]

            # Seasonal baseline = trend + seasonal
            baseline = trend + seasonal

            common_index = series.index.intersection(baseline.index)
            if len(common_index) > 0:
                output_df.loc[common_index, "baseline"] = baseline.loc[common_index].values
                output_df.loc[common_index, "seasonal_deviation"] = (
                    residual.loc[common_index].values
                )

                if len(residual) > 1:
                    std = residual.std()
                    if std > 0:
                        output_df.loc[common_index, "zscore_seasonal"] = (
                            residual.loc[common_index].values / std
                        )

        # Add data quality info
        coverage = output_df["baseline"].notna().sum() / len(output_df)

        return FeatureOutput(
            features=output_df,
            feature_name="seasonality",
            data_quality_score=coverage,
            coverage_ratio=coverage,
            sample_count=len(output_df["baseline"].dropna()),
            config=self.get_config(),
        )

    def _compute_rolling_baseline(
        self,
        series: pd.Series,
        period: int,
        window: Optional[int] = None,
    ) -> pd.Series:
        """
        Compute rolling seasonal baseline.

        Uses centered rolling window to compute average for each
        seasonal position (day of week, day of year, etc.).

        Args:
            series: Time series data
            period: Seasonal period (7=daily/weekly, 365=daily/yearly)
            window: Rolling window size (default=period)

        Returns:
            Series with baseline values
        """
        if window is None:
            window = min(period * 3, len(series))

        # For annual seasonality on daily data, use a different approach
        if period >= 30 and len(series) < period * 2:
            # Not enough data for proper annual seasonality
            # Fall back to simple moving average
            return series.rolling(window=window, min_periods=1).mean()

        # Create seasonal position grouping
        if period == 7:
            # Day of week seasonality
            group_key = series.index.dayofweek
        elif period == 30:
            # Day of month seasonality
            group_key = series.index.day
        elif period == 365:
            # Day of year seasonality
            group_key = series.index.dayofyear
        else:
            # Generic period
            group_key = (series.index.astype(np.int64) // (86400 * 10**9 * period)) % period

        # Compute baseline by seasonal position using rolling mean
        baseline = series.groupby(group_key).transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

        return baseline

    def _decompose_series(
        self,
        series: pd.Series,
        period: int,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Decompose series into trend, seasonal, and residual components.

        Simple decomposition using moving averages.

        Args:
            series: Time series data
            period: Seasonal period

        Returns:
            Tuple of (trend, seasonal, residual)
        """
        # Trend: centered moving average
        trend = series.rolling(window=period, center=True, min_periods=1).mean()

        # Detrended series
        detrended = series - trend

        # Seasonal: average detrended value by seasonal position
        if period == 7:
            group_key = detrended.index.dayofweek
        elif period == 365:
            group_key = detrended.index.dayofyear
        else:
            group_key = (detrended.index.astype(np.int64) // (86400 * 10**9 * period)) % period

        seasonal_avg = detrended.groupby(group_key).mean()
        seasonal = group_key.map(seasonal_avg)

        # Residual
        residual = series - trend - seasonal

        return trend, seasonal, residual


def compute_seasonal_adjustment(
    data: pd.DataFrame,
    target_column: str,
    period: int = 365,
) -> pd.DataFrame:
    """
    Convenience function to compute seasonal adjustments.

    Args:
        data: Input DataFrame with date index
        target_column: Column to adjust
        period: Seasonal period

    Returns:
        DataFrame with seasonal features added
    """
    seasonality = SeasonalityFeature(periods=[period])
    output = seasonality.compute(data, target_column)
    return output.features


__all__ = [
    "SeasonalityFeature",
    "compute_seasonal_adjustment",
]
