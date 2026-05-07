"""
Signal probability estimation.

Converts raw signal strength into success probability.

Methods:
- Logistic regression on historical outcomes
- Platt scaling
- Isotonic regression
- Empirical Bayes
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


@dataclass
class ProbabilityModel:
    """
    Model for converting signal strength to success probability.

    Attributes:
        method: Calibration method used
        calibration_data: Historical data used for calibration
        last_calibrated: Timestamp of last calibration
        is_fit: Whether the model has been fitted
    """
    method: str = "logistic"
    calibration_data: pd.DataFrame = None
    last_calibrated: str = ""
    is_fit: bool = False

    # Model parameters
    model: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "last_calibrated": self.last_calibrated,
            "is_fit": self.is_fit,
        }


class ProbabilityEstimator:
    """
    Estimate probability of signal success from historical outcomes.

    Instead of just A/B/C grades, outputs actual probabilities
    that can be used for position sizing and risk management.

    Example:
        >>> estimator = ProbabilityEstimator(method="logistic")
        >>> estimator.fit(
        ...     strengths=[1.0, 2.0, 3.0],
        ...     outcomes=[1, 0, 1]  # 1=success, 0=failure
        ... )
        >>> prob = estimator.predict_proba(2.5)
        >>> print(f"Success probability: {prob:.1%}")
    """

    def __init__(
        self,
        method: str = "logistic",
        min_samples: int = 30,
    ):
        """
        Initialize probability estimator.

        Args:
            method: Calibration method ('logistic', 'platt', 'isotonic', 'empirical')
            min_samples: Minimum samples required for calibration
        """
        self.method = method
        self.min_samples = min_samples
        self.model = ProbabilityModel(method=method)

    def fit(
        self,
        strengths: List[float],
        outcomes: List[int],
        weights: Optional[List[float]] = None,
    ) -> ProbabilityModel:
        """
        Fit probability model to historical data.

        Args:
            strengths: Signal strength values (z-scores, etc.)
            outcomes: Binary outcomes (1=success, 0=failure)
            weights: Optional sample weights

        Returns:
            Fitted ProbabilityModel
        """
        if len(strengths) < self.min_samples:
            raise ValueError(f"Need at least {self.min_samples} samples, got {len(strengths)}")

        X = np.array(strengths).reshape(-1, 1)
        y = np.array(outcomes)

        if self.method == "logistic":
            model = LogisticRegression()
            model.fit(X, y, sample_weight=weights)
            self.model.model = model

        elif self.method == "platt":
            # Platt scaling: logistic calibration on raw scores
            self.model.model = self._fit_platt(X, y)

        elif self.method == "isotonic":
            # Isotonic regression: non-parametric monotonic calibration
            iso_model = IsotonicRegression(out_of_bounds='clip')
            iso_model.fit(X.flatten(), y)
            self.model.model = iso_model

        elif self.method == "empirical":
            # Empirical: use historical win rate by bucket
            self.model.model = self._fit_empirical(X, y)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.model.calibration_data = pd.DataFrame({
            "strength": strengths,
            "outcome": outcomes,
        })
        self.model.last_calibrated = datetime.now().isoformat()
        self.model.is_fit = True

        return self.model

    def predict_proba(
        self,
        strength: float,
    ) -> float:
        """
        Predict success probability for a signal.

        Args:
            strength: Signal strength value

        Returns:
            Probability of success (0-1)
        """
        if not self.model.is_fit:
            return 0.5  # Default to 50%

        X = np.array([[strength]])

        if self.method == "logistic":
            prob = self.model.model.predict_proba(X)[0, 1]

        elif self.method == "platt":
            prob = self._predict_platt(X)

        elif self.method == "isotonic":
            prob = self.model.model.predict(X.flatten())
            prob = np.clip(prob, 0.01, 0.99)  # Avoid extremes

        elif self.method == "empirical":
            prob = self._predict_empirical(X)

        else:
            prob = 0.5

        return float(prob)

    def predict_proba_batch(
        self,
        strengths: List[float],
    ) -> List[float]:
        """Predict probabilities for multiple values."""
        return [self.predict_proba(s) for s in strengths]

    def _fit_platt(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, float]:
        """Fit Platt scaling model."""
        # Platt scaling: fit logistic regression to scores
        model = LogisticRegression(C=1000)  # High C to prevent regularization
        model.fit(X, y)
        return model

    def _predict_platt(self, X: np.ndarray) -> float:
        """Predict using Platt scaling."""
        return self.model.model.predict_proba(X)[0, 1]

    def _fit_empirical(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_buckets: int = 10,
    ) -> Dict[str, float]:
        """Fit empirical model (binning by strength)."""
        df = pd.DataFrame({"strength": X.flatten(), "outcome": y})

        # Create quantile-based buckets
        df["bucket"] = pd.qcut(df["strength"], q=n_buckets, duplicates="drop")

        # Calculate win rate per bucket
        bucket_stats = df.groupby("bucket")["outcome"].agg(["mean", "count"])

        # Store bucket boundaries and rates
        boundaries = []
        rates = []

        for interval in bucket_stats.index:
            boundaries.append(interval)
            rates.append(bucket_stats.loc[interval, "mean"])

        return {"boundaries": boundaries, "rates": rates}

    def _predict_empirical(self, X: np.ndarray) -> float:
        """Predict using empirical model."""
        strength = X.flatten()[0]

        # Find the right bucket
        for interval, rate in zip(
            self.model.model["boundaries"],
            self.model.model["rates"]
        ):
            if strength in interval:
                return float(rate)

        # If outside buckets, use nearest
        if strength < self.model.model["boundaries"][0].left:
            return float(self.model.model["rates"][0])
        else:
            return float(self.model.model["rates"][-1])

    def get_calibration_curve(
        self,
        n_points: int = 20,
    ) -> Tuple[List[float], List[float]]:
        """
        Get calibration curve for visualization.

        Returns:
            Tuple of (strengths, predicted_probabilities)
        """
        if not self.model.is_fit:
            return [], []

        strengths = np.linspace(-3, 3, n_points)
        probs = [self.predict_proba(s) for s in strengths]

        return strengths.tolist(), probs


def estimate_signal_probability(
    strength: float,
    historical_strengths: List[float],
    historical_outcomes: List[int],
    method: str = "logistic",
) -> float:
    """
    Convenience function for probability estimation.

    Args:
        strength: Signal strength value
        historical_strengths: Historical strength values
        historical_outcomes: Historical binary outcomes
        method: Calibration method

    Returns:
        Success probability (0-1)
    """
    estimator = ProbabilityEstimator(method=method)
    estimator.fit(historical_strengths, historical_outcomes)
    return estimator.predict_proba(strength)


def calibrate_signal_grades(
    probabilities: List[float],
) -> Dict[str, str]:
    """
    Convert probabilities to grade thresholds.

    Args:
        probabilities: List of probability values

    Returns:
        Dict mapping grades to probability ranges
    """
    sorted_probs = sorted(probabilities)

    if len(sorted_probs) < 3:
        return {"A": "N/A", "B": "N/A", "C": "N/A"}

    # Use tertiles
    p33 = np.percentile(sorted_probs, 33)
    p67 = np.percentile(sorted_probs, 67)

    return {
        "A": f"High (p > {p67:.2f})",
        "B": f"Medium ({p33:.2f} <= p <= {p67:.2f})",
        "C": f"Low (p < {p33:.2f})",
    }


__all__ = [
    "ProbabilityEstimator",
    "ProbabilityModel",
    "estimate_signal_probability",
    "calibrate_signal_grades",
]
