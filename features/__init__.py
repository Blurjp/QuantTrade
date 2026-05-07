"""
QuantTrade feature engineering layer.

Responsible for:
- Computing features from raw data
- Normalization and scaling
- Seasonality adjustments
- Quality scoring
- Cross-asset confirmation signals

This layer transforms raw data from data/ into features
that can be used by strategies/ for signal generation.

Feature modules should be stateless and composable.
"""

__all__ = []
