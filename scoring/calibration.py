"""
Signal calibration.

Adjusts signal scores based on historical performance.

Methods:
- Platt scaling calibration
- Temperature scaling
- Ensemble calibration
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


@dataclass
class CalibrationResult:
    """Result of signal calibration."""
    method: str
    calibrated_at: str
    num_samples: int
    calibration_score: float  # How well-calibrated (Brier score)

    # Parameters
    temperature: float = 1.0
    intercept: float = 0.0
    slope: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "calibrated_at": self.calibrated_at,
            "num_samples": self.num_samples,
            "calibration_score": self.calibration_score,
            "temperature": self.temperature,
            "intercept": self.intercept,
            "slope": self.slope,
        }


class SignalCalibrator:
    """
    Calibrate signal scores based on historical performance.

    Example:
        >>> calibrator = SignalCalibrator()
        >>> calibrator.fit(
        ...     raw_scores=[0.5, 0.7, 0.9],
        ...     actual_outcomes=[1, 1, 0],
        ... )
        >>> calibrated = calibrator.calibrate([0.6, 0.8])
    """

    def __init__(self, method: str = "platt"):
        """
        Initialize calibrator.

        Args:
            method: Calibration method ('platt', 'temperature', 'isotonic')
        """
        self.method = method
        self.is_fit = False
        self._model = None

    def fit(
        self,
        raw_scores: List[float],
        actual_outcomes: List[int],
    ) -> CalibrationResult:
        """
        Fit calibration model to historical data.

        Args:
            raw_scores: Raw signal scores (z-scores, etc.)
            actual_outcomes: Binary outcomes (1=success, 0=failure)

        Returns:
            CalibrationResult
        """
        X = np.array(raw_scores).reshape(-1, 1)
        y = np.array(actual_outcomes)

        if self.method == "platt":
            # Platt scaling
            self._model = LogisticRegression(C=1000)
            self._model.fit(X, y)

            result = CalibrationResult(
                method="platt",
                calibrated_at=datetime.now().isoformat(),
                num_samples=len(raw_scores),
                calibration_score=self._brier_score(X, y),
                intercept=float(self._model.intercept_[0]),
                slope=float(self._model.coef_[0]),
            )

        elif self.method == "temperature":
            # Temperature scaling for neural network style outputs
            best_temp = self._find_optimal_temperature(X, y)
            self._model = {"temperature": best_temp}

            result = CalibrationResult(
                method="temperature",
                calibrated_at=datetime.now().isoformat(),
                num_samples=len(raw_scores),
                calibration_score=self._brier_score(X, y, best_temp),
                temperature=best_temp,
            )

        elif self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self._model = IsotonicRegression(out_of_bounds='clip')
            self._model.fit(X.flatten(), y)

            result = CalibrationResult(
                method="isotonic",
                calibrated_at=datetime.now().isoformat(),
                num_samples=len(raw_scores),
                calibration_score=self._brier_score(X, y),
            )

        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.is_fit = True
        return result

    def calibrate(
        self,
        raw_scores: List[float],
    ) -> List[float]:
        """
        Calibrate raw signal scores.

        Args:
            raw_scores: Raw signal scores

        Returns:
            Calibrated probabilities
        """
        if not self.is_fit:
            return raw_scores  # Return as-is if not calibrated

        X = np.array(raw_scores).reshape(-1, 1)

        if self.method == "platt":
            probs = self._model.predict_proba(X)[:, 1]

        elif self.method == "temperature":
            temp = self._model["temperature"]
            probs = 1 / (1 + np.exp(-X.flatten() / temp))

        elif self.method == "isotonic":
            probs = self._model.predict(X.flatten())

        else:
            probs = np.array(raw_scores)

        return probs.tolist()

    def _find_optimal_temperature(
        self,
        X: np.ndarray,
        y: np.ndarray,
        temps: List[float] = None,
    ) -> float:
        """Find optimal temperature for temperature scaling."""
        if temps is None:
            temps = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

        best_temp = 1.0
        best_score = float('inf')

        for temp in temps:
            # Apply temperature scaling
            scaled = 1 / (1 + np.exp(-X.flatten() / temp))
            score = self._brier_score(X, y, temperature=temp)

            if score < best_score:
                best_score = score
                best_temp = temp

        return best_temp

    def _brier_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        temperature: Optional[float] = None,
    ) -> float:
        """Calculate Brier score (lower is better)."""
        if temperature is not None:
            probs = 1 / (1 + np.exp(-X.flatten() / temperature))
        else:
            probs = self._model.predict_proba(X)[:, 1]

        return np.mean((probs - y) ** 2)


def calibrate_signals(
    raw_scores: List[float],
    actual_returns: List[float],
    method: str = "platt",
) -> Tuple[List[float], CalibrationResult]:
    """
    Convenience function for signal calibration.

    Args:
        raw_scores: Raw signal scores
        actual_returns: Actual returns (converted to binary)
        method: Calibration method

    Returns:
        Tuple of (calibrated_scores, calibration_result)
    """
    # Convert returns to binary outcomes
    binary_outcomes = [1 if r > 0 else 0 for r in actual_returns]

    calibrator = SignalCalibrator(method=method)
    result = calibrator.fit(raw_scores, binary_outcomes)
    calibrated = calibrator.calibrate(raw_scores)

    return calibrated, result


def cross_validate_calibration(
    scores: List[float],
    outcomes: List[int],
    n_folds: int = 5,
) -> Dict[str, float]:
    """
    Cross-validate calibration to check for overfitting.

    Args:
        scores: Signal scores
        outcomes: Binary outcomes
        n_folds: Number of CV folds

    Returns:
        Dict with CV metrics
    """
    from sklearn.model_selection import KFold

    scores_arr = np.array(scores)
    outcomes_arr = np.array(outcomes)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    brier_scores = []
    accuracies = []

    for train_idx, test_idx in kf.split(scores_arr):
        calibrator = SignalCalibrator(method="platt")

        # Fit on train
        calibrator.fit(
            scores_arr[train_idx].tolist(),
            outcomes_arr[train_idx].tolist(),
        )

        # Evaluate on test
        test_probs = calibrator.calibrate(scores_arr[test_idx].tolist())
        test_outcomes = outcomes_arr[test_idx]

        # Brier score
        brier = np.mean((test_probs - test_outcomes) ** 2)
        brier_scores.append(brier)

        # Accuracy
        pred_binary = [1 if p > 0.5 else 0 for p in test_probs]
        acc = np.mean(pred_binary == test_outcomes)
        accuracies.append(acc)

    return {
        "mean_brier_score": np.mean(brier_scores),
        "std_brier_score": np.std(brier_scores),
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
    }


__all__ = [
    "SignalCalibrator",
    "CalibrationResult",
    "calibrate_signals",
    "cross_validate_calibration",
]
