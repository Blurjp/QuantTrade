"""
Asset tracking for portfolio value history.

Records daily portfolio values for equity curve and performance tracking.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def record_daily_assets(
    total_value: float,
    target_date: str,
    output_base: str = "outputs",
) -> Dict:
    """
    Record daily portfolio value.

    Args:
        total_value: Total portfolio value
        target_date: Date string (YYYY-MM-DD)
        output_base: Base output directory

    Returns:
        Dictionary with recorded data
    """
    output_path = Path(output_base)
    history_file = output_path / "asset_history.json"

    # Load existing history
    history = []
    if history_file.exists():
        try:
            history = json.loads(history_file.read_text())
            if not isinstance(history, list):
                history = []
        except (json.JSONDecodeError, Exception):
            history = []

    # Check if entry for this date already exists
    existing_idx = None
    for i, entry in enumerate(history):
        if entry.get("date") == target_date:
            existing_idx = i
            break

    entry = {
        "date": target_date,
        "total_value": round(total_value, 2),
        "recorded_at": datetime.now().isoformat(),
    }

    if existing_idx is not None:
        history[existing_idx] = entry
    else:
        history.append(entry)

    # Sort by date
    history.sort(key=lambda x: x["date"])

    # Save history
    output_path.mkdir(parents=True, exist_ok=True)
    history_file.write_text(json.dumps(history, indent=2))

    return {
        "status": "success",
        "date": target_date,
        "total_value": total_value,
        "history_length": len(history),
    }


def load_asset_history(output_base: str = "outputs") -> list:
    """
    Load asset history.

    Args:
        output_base: Base output directory

    Returns:
        List of historical asset records
    """
    history_file = Path(output_base) / "asset_history.json"

    if not history_file.exists():
        return []

    try:
        return json.loads(history_file.read_text())
    except (json.JSONDecodeError, Exception):
        return []


def get_latest_portfolio_value(output_base: str = "outputs") -> Optional[float]:
    """
    Get the most recent portfolio value.

    Args:
        output_base: Base output directory

    Returns:
        Latest portfolio value or None if no history
    """
    history = load_asset_history(output_base)

    if not history:
        return None

    # Return the most recent value
    return history[-1].get("total_value")


def calculate_performance_metrics(output_base: str = "outputs") -> Dict:
    """
    Calculate portfolio performance metrics.

    Args:
        output_base: Base output directory

    Returns:
        Dictionary with performance metrics
    """
    history = load_asset_history(output_base)

    if len(history) < 2:
        return {
            "total_return": None,
            "days_tracked": len(history),
            "message": "Insufficient data for metrics",
        }

    initial_value = history[0].get("total_value", 0)
    final_value = history[-1].get("total_value", 0)

    if initial_value <= 0:
        return {
            "total_return": None,
            "days_tracked": len(history),
            "message": "Invalid initial value",
        }

    total_return = (final_value - initial_value) / initial_value
    days_tracked = len(history)

    # Calculate daily returns for volatility
    daily_returns = []
    for i in range(1, len(history)):
        prev_val = history[i - 1].get("total_value", 0)
        curr_val = history[i].get("total_value", 0)
        if prev_val > 0:
            daily_returns.append((curr_val - prev_val) / prev_val)

    volatility = 0.0
    if daily_returns:
        import numpy as np
        volatility = float(np.std(daily_returns) * (252 ** 0.5))  # Annualized

    return {
        "total_return": round(total_return, 4),
        "total_return_pct": round(total_return * 100, 2),
        "initial_value": initial_value,
        "final_value": final_value,
        "days_tracked": days_tracked,
        "annualized_volatility": round(volatility, 4),
    }


__all__ = [
    "record_daily_assets",
    "load_asset_history",
    "get_latest_portfolio_value",
    "calculate_performance_metrics",
]
