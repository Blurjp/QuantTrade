"""Shared confidence label utility for the QuantTrade pipeline.

Provides a single canonical confidence_label() function so that all modules
use consistent thresholds and scale handling.
"""

from __future__ import annotations

import pandas as pd


def confidence_label(score: float, scale: str = "auto") -> str:
    """Convert a numeric confidence score to a human-readable label.

    Args:
        score: Numeric confidence value.
        scale: One of ``"auto"``, ``"0-1"``, or ``"0-100"``.
            * ``"auto"``  – if *score* > 1.0, treat it as 0-100 and
              normalise by dividing by 100.
            * ``"0-1"``   – treat *score* as already in the 0-1 range.
            * ``"0-100"`` – divide *score* by 100 before applying
              thresholds.

    Returns:
        ``"High"`` if the normalised score ≥ 0.75,
        ``"Medium"`` if ≥ 0.55,
        ``"Low"`` otherwise,
        ``"Unknown"`` when *score* is NaN.
    """
    if pd.isna(score):
        return "Unknown"

    if scale == "0-100":
        score = score / 100.0
    elif scale == "auto":
        if score > 1.0:
            score = score / 100.0

    if score >= 0.75:
        return "High"
    if score >= 0.55:
        return "Medium"
    return "Low"
