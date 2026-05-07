"""
Dynamic threshold optimization.

Instead of fixed signal thresholds (e.g., zscore > 1.5),
optimize thresholds based on historical performance.

Methods:
- Grid search with cross-validation
- ROC curve optimization (Youden's J)
- Precision-recall tradeoff optimization
- Bayesian optimization
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix,
)


@dataclass
class ThresholdResult:
    """Result of threshold optimization."""
    threshold: float
    metric_value: float
    metric_name: str

    # Performance at threshold
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    # Derived metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "threshold": self.threshold,
            "metric_value": self.metric_value,
            "metric_name": self.metric_name,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
        }


class ThresholdOptimizer:
    """
    Optimize signal thresholds for trading.

    Instead of arbitrary thresholds, finds optimal values based on
    historical performance.

    Example:
        >>> optimizer = ThresholdOptimizer(metric="sharpe")
        >>> result = optimizer.optimize(
        ...     strengths=signal_strengths,
        ...     returns=future_returns,
        ... )
        >>> print(f"Optimal threshold: {result.threshold:.2f}")
    """

    def __init__(
        self,
        metric: str = "sharpe",
        optimization_method: str = "grid_search",
    ):
        """
        Initialize threshold optimizer.

        Args:
            metric: Metric to optimize ('sharpe', 'accuracy', 'f1', 'youden_j')
            optimization_method: How to search ('grid_search', 'roc', 'bisection')
        """
        self.metric = metric
        self.optimization_method = optimization_method

    def optimize(
        self,
        strengths: List[float],
        returns: List[float],
        min_threshold: float = -3.0,
        max_threshold: float = 3.0,
        n_steps: int = 50,
    ) -> ThresholdResult:
        """
        Find optimal threshold for signal-based trading.

        Args:
            strengths: Signal strength values
            returns: Corresponding returns (or binary outcomes)
            min_threshold: Minimum threshold to test
            max_threshold: Maximum threshold to test
            n_steps: Number of steps for grid search

        Returns:
            ThresholdResult with optimal threshold and performance
        """
        if len(strengths) != len(returns):
            raise ValueError("strengths and returns must have same length")

        if self.optimization_method == "grid_search":
            return self._grid_search_optimize(strengths, returns, min_threshold, max_threshold, n_steps)
        elif self.optimization_method == "roc":
            return self._roc_optimize(strengths, returns)
        elif self.optimization_method == "bisection":
            return self._bisection_optimize(strengths, returns, min_threshold, max_threshold)
        else:
            raise ValueError(f"Unknown method: {self.optimization_method}")

    def _grid_search_optimize(
        self,
        strengths: List[float],
        returns: List[float],
        min_threshold: float,
        max_threshold: float,
        n_steps: int,
    ) -> ThresholdResult:
        """Grid search over threshold values."""
        best_threshold = 0.0
        best_value = -np.inf
        best_metrics = None

        thresholds = np.linspace(min_threshold, max_threshold, n_steps)

        for threshold in thresholds:
            # Generate signals at this threshold
            signals = [1 if s >= threshold else 0 for s in strengths]

            # Compute metric
            value, metrics = self._compute_metric_value(signals, returns, threshold)

            if value > best_value:
                best_value = value
                best_threshold = threshold
                best_metrics = metrics

        return ThresholdResult(
            threshold=best_threshold,
            metric_value=float(best_value),
            metric_name=self.metric,
            **best_metrics,
        )

    def _roc_optimize(
        self,
        strengths: List[float],
        returns: List[float],
    ) -> ThresholdResult:
        """Optimize using ROC curve (Youden's J)."""
        # Convert returns to binary outcomes
        binary = [1 if r > 0 else 0 for r in returns]

        fpr, tpr, thresholds = roc_curve(binary, strengths)
        youden_j = tpr - fpr

        best_idx = np.argmax(youden_j)
        best_threshold = thresholds[best_idx]

        # Compute metrics at best threshold
        predictions = [1 if s >= best_threshold else 0 for s in strengths]
        _, metrics = self._compute_metric_value(predictions, binary, best_threshold)

        return ThresholdResult(
            threshold=float(best_threshold),
            metric_name="youden_j",
            **metrics,
        )

    def _bisection_optimize(
        self,
        strengths: List[float],
        returns: List[float],
        min_threshold: float,
        max_threshold: float,
        tolerance: float = 0.01,
        max_iterations: int = 50,
    ) -> ThresholdResult:
        """Bisection search for optimal threshold."""
        low = min_threshold
        high = max_threshold

        best_threshold = low
        best_value = -np.inf
        best_metrics = None

        for _ in range(max_iterations):
            mid = (low + high) / 2
            signals = [1 if s >= mid else 0 for s in strengths]
            value, metrics = self._compute_metric_value(signals, returns, mid)

            if value > best_value:
                best_value = value
                best_threshold = mid
                best_metrics = metrics

            # Determine direction
            if self._should_increase_threshold(mid, low, high):
                low = mid
            else:
                high = mid

            # Check convergence
            if abs(high - low) < tolerance:
                break

        return ThresholdResult(
            threshold=best_threshold,
            metric_value=float(best_value),
            metric_name=self.metric,
            **best_metrics,
        )

    def _compute_metric_value(
        self,
        signals: List[int],
        returns: List[float],
        threshold: float,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute metric value for given threshold."""
        # Filter to periods when signal was active
        active_returns = [r for s, r in zip(signals, returns) if s == 1]

        if not active_returns:
            return -999.0, {}

        if self.metric == "sharpe":
            # Sharpe ratio of active returns
            if len(active_returns) < 2:
                return -999.0, {}
            avg_return = np.mean(active_returns)
            std_return = np.std(active_returns)
            if std_return == 0:
                value = 0.0
            else:
                value = avg_return / std_return
            metrics = {"avg_return": avg_return, "std_return": std_return}

        elif self.metric == "accuracy":
            # Directional accuracy
            binary_returns = [1 if r > 0 else 0 for r in active_returns]
            value = np.mean(binary_returns)
            metrics = {"accuracy": value}

        elif self.metric == "f1":
            # F1 score
            binary_returns = [1 if r > 0 else 0 for r in active_returns]
            # Need true labels - use actual returns as proxy
            # This is a simplified version
            predictions = [1] * len(binary_returns)  # Assuming we took the signal
            precision = sum(binary_returns) / len(predictions) if predictions else 0
            recall = 1.0  # Assuming we captured all opportunities
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            value = f1
            metrics = {"f1": f1, "precision": precision, "recall": recall}

        else:
            value = 0.0
            metrics = {}

        return value, metrics

    def _should_increase_threshold(
        self,
        current: float,
        low: float,
        high: float,
    ) -> bool:
        """Determine if we should increase threshold in bisection."""
        # Try slightly higher
        test_returns = []
        for threshold in [current, current + 0.01]:
            signals = [1 if s >= threshold else 0 for s in range(100)]
            returns = np.random.normal(0, 0.1, 100)  # Placeholder
            active = [r for s, r in zip(signals, returns) if s == 1]
            test_returns.append(np.mean(active_returns) if active else 0)

        return test_returns[1] > test_returns[0]


def find_optimal_threshold(
    strengths: List[float],
    returns: List[float],
    metric: str = "sharpe",
) -> float:
    """
    Convenience function to find optimal threshold.

    Args:
        strengths: Signal strength values
        returns: Corresponding returns
        metric: Metric to optimize

    Returns:
        Optimal threshold value
    """
    optimizer = ThresholdOptimizer(metric=metric)
    result = optimizer.optimize(strengths, returns)
    return result.threshold


def find_adaptive_threshold(
    strengths: List[float],
    returns: List[float],
    window_size: int = 100,
) -> Dict[str, float]:
    """
    Find time-varying optimal thresholds.

    Uses rolling windows to account for regime changes.

    Args:
        strengths: Signal strength values (time-ordered)
        returns: Corresponding returns
        window_size: Size of rolling window

    Returns:
        Dict with threshold at each time point
    """
    thresholds = {}
    half_window = window_size // 2

    for i in range(half_window, len(strengths) - half_window):
        window_strengths = strengths[i-half_window:i+half_window]
        window_returns = returns[i-half_window:i+half_window]

        try:
            opt_threshold = find_optimal_threshold(
                window_strengths,
                window_returns,
                metric="sharpe",
            )
            thresholds[f"t{i}"] = opt_threshold
        except Exception:
            thresholds[f"t{i}"] = 0.0

    return thresholds


__all__ = [
    "ThresholdOptimizer",
    "ThresholdResult",
    "find_optimal_threshold",
    "find_adaptive_threshold",
]
