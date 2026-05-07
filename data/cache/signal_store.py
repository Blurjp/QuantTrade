"""
Signal storage and persistence layer.

Provides:
- Signal JSON storage/retrieval
- Signal persistence state management
- Signal history queries

NOTE: This module was created during P0.3 refactoring to provide
a unified interface for signal storage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


def get_signal_persistence_file(output_base: str = "outputs") -> Path:
    """Get the path to the signal persistence state file."""
    return Path(output_base) / "signal_persistence_state.json"


def load_persistence_state(output_base: str = "outputs") -> Dict[str, Any]:
    """
    Load signal persistence state from disk.

    The persistence state tracks:
    - Current live action for each region/meta group
    - Pending action and confirmation count
    - Used for signal confirmation logic

    Args:
        output_base: Output directory base path

    Returns:
        Dict mapping region/meta_group IDs to state dicts
    """
    state_file = get_signal_persistence_file(output_base)
    if not state_file.exists():
        return {}
    return json.loads(state_file.read_text())


def save_persistence_state(output_base: str, state: Dict[str, Any]) -> None:
    """
    Save signal persistence state to disk.

    Args:
        output_base: Output directory base path
        state: State dict to save
    """
    state_file = get_signal_persistence_file(output_base)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2))


def get_signal_file(
    region_id: str,
    target_date: str,
    output_base: str = "outputs",
) -> Path:
    """
    Get the path to a signal file for a specific region and date.

    Args:
        region_id: Region identifier
        target_date: Date string (YYYY-MM-DD)
        output_base: Output directory base path

    Returns:
        Path to the signal JSON file
    """
    return Path(output_base) / target_date / "signals" / f"{region_id}_signal.json"


def save_signal(
    region_id: str,
    target_date: str,
    signal: Dict[str, Any],
    output_base: str = "outputs",
) -> Path:
    """
    Save a signal to disk.

    Args:
        region_id: Region identifier
        target_date: Date string (YYYY-MM-DD)
        signal: Signal data dict
        output_base: Output directory base path

    Returns:
        Path to saved signal file
    """
    signal_file = get_signal_file(region_id, target_date, output_base)
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    signal_file.write_text(json.dumps(signal, indent=2, default=str))
    return signal_file


def load_signal(
    region_id: str,
    target_date: str,
    output_base: str = "outputs",
) -> Optional[Dict[str, Any]]:
    """
    Load a signal from disk.

    Args:
        region_id: Region identifier
        target_date: Date string (YYYY-MM-DD)
        output_base: Output directory base path

    Returns:
        Signal data dict, or None if not found
    """
    signal_file = get_signal_file(region_id, target_date, output_base)
    if not signal_file.exists():
        return None
    return json.loads(signal_file.read_text())


def get_daily_summary_file(
    target_date: str,
    output_base: str = "outputs",
) -> Path:
    """
    Get the path to the daily summary file.

    Args:
        target_date: Date string (YYYY-MM-DD)
        output_base: Output directory base path

    Returns:
        Path to daily summary JSON file
    """
    return Path(output_base) / target_date / "daily_summary.json"


def save_daily_summary(
    target_date: str,
    summary: Dict[str, Any],
    output_base: str = "outputs",
) -> Path:
    """
    Save daily pipeline summary to disk.

    Args:
        target_date: Date string (YYYY-MM-DD)
        summary: Summary data dict
        output_base: Output directory base path

    Returns:
        Path to saved summary file
    """
    summary_file = get_daily_summary_file(target_date, output_base)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2, default=str))
    return summary_file


def load_daily_summary(
    target_date: str,
    output_base: str = "outputs",
) -> Optional[Dict[str, Any]]:
    """
    Load daily pipeline summary from disk.

    Args:
        target_date: Date string (YYYY-MM-DD)
        output_base: Output directory base path

    Returns:
        Summary data dict, or None if not found
    """
    summary_file = get_daily_summary_file(target_date, output_base)
    if not summary_file.exists():
        return None
    return json.loads(summary_file.read_text())


def list_available_dates(
    output_base: str = "outputs",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[str]:
    """
    List available dates with signal data.

    Args:
        output_base: Output directory base path
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        List of date strings (YYYY-MM-DD)
    """
    output_path = Path(output_base)
    if not output_path.exists():
        return []

    dates = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("20"):
            # Check if daily_summary.json exists
            if (item / "daily_summary.json").exists():
                dates.append(item.name)

    # Sort and filter
    dates = sorted(dates, reverse=True)

    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]

    return dates


def query_signal_history(
    region_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_base: str = "outputs",
) -> List[Dict[str, Any]]:
    """
    Query signal history for a region.

    Args:
        region_id: Region identifier
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        output_base: Output directory base path

    Returns:
        List of signal dicts, sorted by date (newest first)
    """
    signals = []
    dates = list_available_dates(output_base, start_date, end_date)

    for date_str in dates:
        summary = load_daily_summary(date_str, output_base)
        if summary and "signals" in summary:
            signal = summary["signals"].get(region_id)
            if signal:
                signal["date"] = date_str
                signals.append(signal)

    return signals


__all__ = [
    "load_persistence_state",
    "save_persistence_state",
    "save_signal",
    "load_signal",
    "save_daily_summary",
    "load_daily_summary",
    "list_available_dates",
    "query_signal_history",
]
