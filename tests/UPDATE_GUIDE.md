"""
P3.3 - Tests Update Guide

This document lists test files that need updating after refactoring.

FILES TO UPDATE:
1. tests/test_multi_signals.py
   - Imports: from pipeline.signals_multi → from strategies.*
   - Imports: from pipeline.backtest → from research.*
   - Update: Use new BaseStrategy interface

2. tests/test_brazil_meta_signal.py
   - May need updates for new meta signal structure

3. tests/test_pipeline.py
   - Check for compatibility with refactored modules

FILES TO CREATE:
1. tests/test_strategies_base.py
   - Test BaseStrategy protocol
   - Test ResearchSignal/TradeCandidate creation

2. tests/test_features.py
   - Test feature modules (quality, seasonality, normalization)

3. tests/test_execution.py
   - Test trade_mapper, portfolio_rules, risk utilities

4. tests/test_scoring.py
   - Test probability estimation, thresholding, calibration

MIGRATION GUIDE:

Old: from pipeline.signals_multi import generate_signal
New: from strategies.auto_inventory import AutoInventoryStrategy

Old: signal = generate_signal("agriculture", data)
New:
    strategy = AutoInventoryStrategy()
    features = strategy.build_features(data)
    signal = strategy.generate_signal(features)

Old: from pipeline.backtest import optimize_thresholds
New: from scoring.thresholding import find_optimal_threshold

---

BACKWARD COMPATIBILITY:

For now, to keep tests working, add stub functions in pipeline/:

```python
# pipeline/signals_multi_stub.py (temporary)
def generate_signal(monitoring_type, data):
    from strategies.auto_inventory import AutoInventoryStrategy
    from strategies.chokepoint import ChokepointStrategy
    from strategies.oil_storage import OilStorageStrategy

    strategy_map = {
        "auto_inventory": AutoInventoryStrategy,
        "agriculture": AutoInventoryStrategy,  # Use auto_inventory as proxy
        "chokepoint": ChokepointStrategy,
        "oil_storage": OilStorageStrategy,
    }

    strategy_class = strategy_map.get(monitoring_type)
    if strategy_class:
        strategy = strategy_class()
        features = strategy.build_features(data)
        return strategy.generate_signal(features).iloc[0].to_dict()

    raise ValueError(f"Unknown monitoring type: {monitoring_type}")
```

This allows gradual migration of tests while maintaining backward compatibility.
"""

# TODO: Update test files after verifying new schema works
# TODO: Create new tests for refactored modules
