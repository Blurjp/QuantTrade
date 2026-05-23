"""
Shared signal generation used by UI, automation, and backtesting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.regions import list_regions, resolve_region_output_base


from pipeline.confidence_utils import confidence_label as _confidence_label


def _actionability_for(signal: str, confidence: str) -> str:
    if confidence == "Low":
        return "Ignore"
    if signal in {"Long disruption risk", "Short disruption risk"} and confidence in {"High", "Medium"}:
        return "Actionable"
    return "Watchlist"


def _bias_for_signal(signal: str) -> str:
    if signal == "Long disruption risk":
        return "Bullish crude / bullish disruption-sensitive assets"
    if signal == "Short disruption risk":
        return "Bearish crude risk premium / supportive for normalized flow assets"
    return "Signal mixed"


def load_region_metrics_history(output_base: str, region_id: str) -> pd.DataFrame:
    output_root = Path(resolve_region_output_base(output_base, region_id))
    corrected_path = output_root / "calibration" / "corrected_metrics.parquet"
    metrics_path = output_root / "metrics" / "daily.parquet"

    if corrected_path.exists():
        df = pd.read_parquet(corrected_path)
        df["signal_source"] = "throughput_index_corrected"
        return df

    if metrics_path.exists():
        df = pd.read_parquet(metrics_path)
        df["signal_source"] = "throughput_index_total"
        return df

    return pd.DataFrame()


def load_region_calibration_report(output_base: str, region_id: str) -> dict:
    report_path = Path(resolve_region_output_base(output_base, region_id)) / "calibration" / "calibration_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    return {}


def _local_signal_table(metrics_df: pd.DataFrame, version: str = "v2") -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    signal_source = metrics_df["signal_source"].iloc[0] if "signal_source" in metrics_df.columns else "throughput_index_total"
    if signal_source not in metrics_df.columns:
        signal_source = "throughput_index_total"
    if signal_source not in metrics_df.columns:
        return pd.DataFrame()

    df = metrics_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df[signal_source] = pd.to_numeric(df[signal_source], errors="coerce")
    df["coverage_score"] = pd.to_numeric(df.get("coverage_score"), errors="coerce").fillna(0.0)
    df["throughput_value"] = df[signal_source]
    df["day_of_week"] = df["date"].dt.dayofweek
    df["rolling_mean_7"] = df["throughput_value"].shift(1).rolling(7, min_periods=3).mean()
    df["rolling_std_28"] = df["throughput_value"].shift(1).rolling(28, min_periods=5).std()
    df["expanding_mean_prior"] = df["throughput_value"].shift(1).expanding(min_periods=1).mean()

    weekday_baseline = []
    weekday_history: dict[int, list[float]] = {}
    for _, row in df.iterrows():
        day = int(row["day_of_week"])
        history = weekday_history.get(day, [])
        baseline = float(np.mean(history)) if len(history) >= 2 else np.nan
        weekday_baseline.append(baseline)
        if pd.notna(row["throughput_value"]):
            weekday_history.setdefault(day, []).append(float(row["throughput_value"]))

    df["baseline_value"] = pd.Series(weekday_baseline, index=df.index)
    df["baseline_value"] = (
        df["baseline_value"]
        .fillna(df["rolling_mean_7"])
        .fillna(df["expanding_mean_prior"])
        .fillna(df["throughput_value"])
    )
    df["value_delta"] = df["throughput_value"] - df["baseline_value"]
    safe_std = df["rolling_std_28"].replace(0, np.nan)
    df["zscore"] = df["value_delta"] / safe_std
    df["zscore"] = df["zscore"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["dod_change"] = df["throughput_value"].diff()
    previous = df["throughput_value"].shift(1).replace(0, np.nan)
    df["dod_change_pct"] = (df["throughput_value"] - previous) / previous
    df["dod_change_pct"] = df["dod_change_pct"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["coverage_weight"] = df["coverage_score"].clip(0.0, 1.0)

    low_mask = (df["value_delta"] <= -0.05) | (df["zscore"] <= -0.75)
    high_mask = (df["value_delta"] >= 0.05) | (df["zscore"] >= 0.75)
    df["direction_state"] = np.select(
        [low_mask, high_mask],
        ["low", "high"],
        default="flat",
    )

    confirmations = []
    last_state = "flat"
    streak = 0
    for state in df["direction_state"]:
        if state != "flat" and state == last_state:
            streak += 1
        elif state != "flat":
            streak = 1
        else:
            streak = 0
        last_state = state
        confirmations.append(streak)
    df["confirmation_days"] = confirmations

    if version == "v1":
        long_mask = low_mask & (df["dod_change_pct"] <= -0.15)
        short_mask = high_mask & (df["dod_change_pct"] >= 0.15)
    else:
        long_mask = low_mask & (df["confirmation_days"] >= 2)
        short_mask = high_mask & (df["confirmation_days"] >= 2)

    df["signal"] = np.select(
        [long_mask, short_mask],
        ["Long disruption risk", "Short disruption risk"],
        default="No trade",
    )
    df["signal_strength"] = df["zscore"].abs().fillna(df["value_delta"].abs())
    df["reroute_flag"] = False
    df["calibration_weight"] = 0.5
    df["confidence_score"] = (
        0.45 * df["coverage_weight"]
        + 0.25 * (df["confirmation_days"].clip(0, 3) / 3.0)
        + 0.20 * df["calibration_weight"]
        + 0.10 * df["signal_strength"].clip(0, 2) / 2.0
    ).clip(0.0, 1.0)
    df["confidence"] = df["confidence_score"].apply(_confidence_label)
    df["actionability"] = [
        _actionability_for(signal, confidence)
        for signal, confidence in zip(df["signal"], df["confidence"])
    ]
    df["signal_source"] = signal_source
    return df


def _apply_calibration_weight(signal_df: pd.DataFrame, calibration_report: dict) -> pd.DataFrame:
    if signal_df.empty:
        return signal_df

    report = calibration_report.get("performance", {})
    r2 = float(report.get("r2", 0.0) or 0.0)
    n_samples = float(report.get("n_samples", 0.0) or 0.0)
    calibration_weight = max(0.0, min(1.0, r2 * min(1.0, n_samples / 20.0)))

    df = signal_df.copy()
    if df["signal_source"].iloc[0] != "throughput_index_corrected":
        calibration_weight = 0.35
    elif calibration_weight == 0.0:
        calibration_weight = 0.5

    df["calibration_weight"] = calibration_weight
    df["confidence_score"] = (
        0.45 * df["coverage_weight"]
        + 0.25 * (df["confirmation_days"].clip(0, 3) / 3.0)
        + 0.20 * df["calibration_weight"]
        + 0.10 * df["signal_strength"].clip(0, 2) / 2.0
    ).clip(0.0, 1.0)
    df["confidence"] = df["confidence_score"].apply(_confidence_label)
    df["actionability"] = [
        _actionability_for(signal, confidence)
        for signal, confidence in zip(df["signal"], df["confidence"])
    ]
    return df


def _downgrade_confidence(confidence: str) -> str:
    order = ["High", "Medium", "Low"]
    if confidence not in order:
        return confidence
    idx = min(order.index(confidence) + 1, len(order) - 1)
    return order[idx]


def _apply_reroute_logic(signal_df: pd.DataFrame, region_id: str, output_base: str, version: str) -> pd.DataFrame:
    if signal_df.empty or region_id not in {"hormuz", "bab_el_mandeb", "suez_south"}:
        return signal_df

    peers = [peer for peer in ["hormuz", "bab_el_mandeb", "suez_south"] if peer != region_id]
    peer_tables = []
    for peer in peers:
        peer_history = load_region_metrics_history(output_base, peer)
        peer_table = _local_signal_table(peer_history, version=version)
        if not peer_table.empty:
            peer_tables.append(peer_table[["date", "signal"]].rename(columns={"signal": f"{peer}_signal"}))

    if not peer_tables:
        return signal_df

    df = signal_df.copy()
    for peer_table in peer_tables:
        df = df.merge(peer_table, on="date", how="left")

    peer_cols = [column for column in df.columns if column.endswith("_signal")]
    reroute_mask = False
    for col in peer_cols:
        reroute_mask = reroute_mask | (
            (df["signal"] == "Long disruption risk")
            & (df[col] == "Short disruption risk")
        )

    df["reroute_flag"] = reroute_mask
    downgraded = df["reroute_flag"] & (df["actionability"] == "Actionable")
    df.loc[downgraded, "actionability"] = "Watchlist"
    df.loc[downgraded, "confidence"] = df.loc[downgraded, "confidence"].apply(_downgrade_confidence)
    return df.drop(columns=peer_cols, errors="ignore")


def build_region_signal_table(
    region_id: str,
    output_base: str = "outputs",
    version: str = "v2",
) -> pd.DataFrame:
    metrics_df = load_region_metrics_history(output_base, region_id)
    if metrics_df.empty:
        return pd.DataFrame()

    calibration_report = load_region_calibration_report(output_base, region_id)
    signal_df = _local_signal_table(metrics_df, version=version)
    signal_df = _apply_calibration_weight(signal_df, calibration_report)
    if version == "v2":
        signal_df = _apply_reroute_logic(signal_df, region_id, output_base, version)

    signal_df["region"] = region_id
    return signal_df


def latest_region_signal(
    region_id: str,
    output_base: str = "outputs",
    selected_day: str | None = None,
    version: str = "v2",
) -> dict | None:
    signal_df = build_region_signal_table(region_id, output_base=output_base, version=version)
    if signal_df.empty:
        return None

    if selected_day:
        selected = signal_df[signal_df["date"] == pd.Timestamp(selected_day)]
        row = selected.iloc[-1] if not selected.empty else signal_df.iloc[-1]
    else:
        row = signal_df.iloc[-1]

    return {
        "date": row["date"].date().isoformat(),
        "source": "selected_day" if selected_day and row["date"] == pd.Timestamp(selected_day) else "latest_available",
        "signal": row["signal"],
        "bias": _bias_for_signal(row["signal"]),
        "confidence": row["confidence"],
        "signal_strength": float(row["signal_strength"]) if pd.notna(row["signal_strength"]) else None,
        "throughput_index_corrected": float(row["throughput_value"]) if pd.notna(row["throughput_value"]) else None,
        "coverage_score": float(row["coverage_score"]) if pd.notna(row["coverage_score"]) else None,
        "rolling_mean_7": float(row["rolling_mean_7"]) if pd.notna(row["rolling_mean_7"]) else None,
        "baseline_value": float(row["baseline_value"]) if pd.notna(row["baseline_value"]) else None,
        "dod_change": float(row["dod_change"]) if pd.notna(row["dod_change"]) else None,
        "dod_change_pct": float(row["dod_change_pct"]) if pd.notna(row["dod_change_pct"]) else None,
        "confirmation_days": int(row["confirmation_days"]) if pd.notna(row["confirmation_days"]) else 0,
        "zscore": float(row["zscore"]) if pd.notna(row["zscore"]) else None,
        "coverage_weight": float(row["coverage_weight"]) if pd.notna(row["coverage_weight"]) else None,
        "calibration_weight": float(row["calibration_weight"]) if pd.notna(row["calibration_weight"]) else None,
        "reroute_flag": bool(row["reroute_flag"]),
        "actionability": row["actionability"],
        "signal_source": row["signal_source"],
        "rationale": (
            "Potential reroute pattern detected; downgraded from fully actionable."
            if bool(row["reroute_flag"])
            else "Signal derived from deviation versus baseline, confirmation, coverage, and calibration quality."
        ),
        "series": signal_df,
    }


def build_monitor_snapshot(output_base: str = "outputs", version: str = "v2") -> pd.DataFrame:
    rows = []
    for region in list_regions():
        latest = latest_region_signal(region["id"], output_base=output_base, version=version)
        if latest is None:
            rows.append({
                "region": region["id"],
                "date": None,
                "signal": "No data",
                "confidence": "Unknown",
                "actionability": "Ignore",
                "coverage_score": None,
                "signal_strength": None,
                "reroute_flag": False,
            })
            continue
        rows.append({
            "region": region["id"],
            "date": latest["date"],
            "signal": latest["signal"],
            "confidence": latest["confidence"],
            "actionability": latest["actionability"],
            "coverage_score": latest["coverage_score"],
            "signal_strength": latest["signal_strength"],
            "reroute_flag": latest["reroute_flag"],
        })
    return pd.DataFrame(rows)


def _get_seasonal_baseline(data: pd.DataFrame, value_col: str, date_col: str = "date") -> tuple:
    """
    Calculate seasonal baseline for comparison.

    Compares the most recent value to historical values from the same season.
    Falls back to all historical values if seasonal data is insufficient.

    Returns:
        Tuple of (baseline_mean, baseline_std, baseline_samples, current_value)
    """
    if data.empty or value_col not in data.columns:
        return None, None, 0, None

    # Ensure date column is datetime
    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Sort by date
    df = df.sort_values(date_col).reset_index(drop=True)

    # Get current (most recent) value
    current_value = df[value_col].iloc[-1]
    current_month = df[date_col].iloc[-1].month

    # Get historical values (exclude current)
    historical_df = df.iloc[:-1]

    if historical_df.empty:
        return None, None, 0, current_value

    # Try to get values from same season (±1 month)
    seasonal_values = []
    for i in range(len(historical_df)):
        month = historical_df[date_col].iloc[i].month
        # Check if same season (within ±1 month, accounting for year wrap)
        if abs(month - current_month) <= 1 or abs(month - current_month) >= 11:
            seasonal_values.append(historical_df[value_col].iloc[i])

    # Use seasonal values if we have enough, otherwise use all historical
    if len(seasonal_values) >= 3:
        baseline_values = seasonal_values
    else:
        # Fall back to all historical values
        baseline_values = historical_df[value_col].tolist()

    if not baseline_values:
        return None, None, 0, current_value

    baseline_mean = float(np.mean(baseline_values))
    baseline_std = float(np.std(baseline_values)) if len(baseline_values) > 1 else 0.1

    return baseline_mean, baseline_std, len(baseline_values), current_value


def generate_signal(monitoring_type: str, data: pd.DataFrame, **kwargs) -> dict:
    """
    Generate a trading signal for the given monitoring type.

    This is a unified interface for signal generation used by the daily pipeline
    and backtesting. It dispatches to type-specific signal generators.

    Args:
        monitoring_type: Type of monitoring (e.g., "agriculture", "auto_inventory", "oil_storage")
        data: DataFrame with historical data (must include 'date' column and a value column)
        **kwargs: Additional parameters for specific signal generators

    Returns:
        Dictionary with signal information including:
        - trading_action: "LONG", "SHORT", or "FLAT"
        - signal: Human-readable signal description
        - confidence: "High", "Medium", or "Low"
        - baseline_samples: Number of baseline samples used
        - Additional type-specific fields
    """
    # Normalize type name
    normalized_type = monitoring_type.lower().replace("-", "_").replace(" ", "_")

    # Dispatch to type-specific generators
    if normalized_type in ("agriculture", "agricultural"):
        return _generate_agriculture_signal(data, **kwargs)
    elif normalized_type in ("auto_inventory", "autoinventory"):
        return _generate_auto_inventory_signal(data, **kwargs)
    elif normalized_type in ("oil_storage", "oilstorage"):
        return _generate_oil_storage_signal(data, **kwargs)
    elif normalized_type in ("chokepoint", "port_logistics"):
        return _generate_chokepoint_signal(data, **kwargs)
    else:
        return {
            "trading_action": "FLAT",
            "signal": f"No signal generator for type: {monitoring_type}",
            "confidence": "Low",
            "baseline_samples": 0,
            "type": monitoring_type,
        }


def _generate_agriculture_signal(data: pd.DataFrame, **kwargs) -> dict:
    """Generate signal for agricultural monitoring (NDVI-based)."""
    # Find the value column (ndvi_mean or similar)
    value_col = None
    for col in ["ndvi_mean", "ndvi", "vegetation_index"]:
        if col in data.columns:
            value_col = col
            break

    if value_col is None:
        return {
            "trading_action": "FLAT",
            "signal": "No vegetation data available",
            "confidence": "Low",
            "baseline_samples": 0,
        }

    baseline_mean, baseline_std, baseline_samples, current_value = _get_seasonal_baseline(data, value_col)

    if baseline_mean is None:
        return {
            "trading_action": "FLAT",
            "signal": "Insufficient historical data",
            "confidence": "Low",
            "baseline_samples": 0,
        }

    # Handle None/NaN in current_value
    if current_value is None or (isinstance(current_value, float) and pd.isna(current_value)):
        current_value = baseline_mean  # Use baseline as fallback

    # Calculate change from baseline
    ndvi_change = current_value - baseline_mean
    ndvi_change_pct = (ndvi_change / baseline_mean) * 100 if baseline_mean > 0 else 0

    # Generate signal based on vegetation health
    # Low NDVI = crop stress = supply concerns = LONG (prices up)
    # High NDVI = good health = good supply = SHORT (prices down)

    threshold = kwargs.get("threshold", 0.05)

    if ndvi_change < -threshold:
        trading_action = "LONG"
        signal = "Long crop (supply concerns)"
        confidence = "High" if abs(ndvi_change) > 0.1 else "Medium"
    elif ndvi_change > threshold:
        trading_action = "SHORT"
        signal = "Short crop (good supply)"
        confidence = "High" if abs(ndvi_change) > 0.1 else "Medium"
    else:
        trading_action = "FLAT"
        signal = "Neutral crop conditions"
        confidence = "Low"

    return {
        "trading_action": trading_action,
        "signal": signal,
        "confidence": confidence,
        "baseline_samples": baseline_samples,
        "baseline_value": round(baseline_mean, 3),
        "current_value": round(current_value, 3),
        "ndvi_change": round(ndvi_change, 3),
        "ndvi_change_pct": round(ndvi_change_pct, 2),
    }


def _generate_auto_inventory_signal(data: pd.DataFrame, **kwargs) -> dict:
    """Generate signal for auto inventory monitoring.

    Uses NDVI as a proxy for parking lot occupancy in auto dealer lots.
    - Higher NDVI in parking areas suggests more vehicles (metal surfaces)
    - More vehicles = higher inventory = SHORT (oversupply, prices down)
    - Fewer vehicles = lower inventory = LONG (supply shortage, prices up)
    """
    # Find the value column
    value_col = None
    for col in ["ndvi_mean", "inventory", "count", "vehicles"]:
        if col in data.columns:
            value_col = col
            break

    if value_col is None:
        return {
            "trading_action": "FLAT",
            "signal": "No inventory data available",
            "confidence": "Low",
            "baseline_samples": 0,
            "inventory_source": "unknown",
        }

    baseline_mean, baseline_std, baseline_samples, current_value = _get_seasonal_baseline(data, value_col)

    if baseline_mean is None:
        return {
            "trading_action": "FLAT",
            "signal": "Insufficient historical data",
            "confidence": "Low",
            "baseline_samples": 0,
            "inventory_source": "optical inventory proxy"
        }
    # Handle None/NaN in current_value
    if current_value is None or (isinstance(current_value, float) and pd.isna(current_value)):
        current_value = baseline_mean  # Use baseline as fallback
    # Calculate change from baseline
    # For parking lots, values are typically small (0.05-0.25 range)
    inventory_change = current_value - baseline_mean
    inventory_change_pct = (inventory_change / baseline_mean) * 100 if baseline_mean > 0 else 0

    # Generate signal based on inventory levels
    # High NDVI change = more cars = higher inventory = SHORT
    # Low NDVI change = fewer cars = lower inventory = LONG

    # Use relative threshold since parking lot NDVI values are small
    threshold = kwargs.get("threshold", 0.02)  # 0.02 absolute change for small values

    if inventory_change > threshold:
        trading_action = "SHORT"
        signal = "Short auto (high inventory)"
        confidence = "High" if inventory_change_pct > 20 else "Medium"
    elif inventory_change < -threshold:
        trading_action = "LONG"
        signal = "Long auto (low inventory)"
        confidence = "High" if abs(inventory_change_pct) > 20 else "Medium"
    else:
        trading_action = "FLAT"
        signal = "Neutral inventory levels"
        confidence = "Low"

    return {
        "trading_action": trading_action,
        "signal": signal,
        "confidence": confidence,
        "baseline_samples": baseline_samples,
        "baseline_value": round(baseline_mean, 3),
        "current_value": round(current_value, 3),
        "inventory_change": round(inventory_change, 3),
        "inventory_change_pct": round(inventory_change_pct, 2),
        "inventory_source": "optical inventory proxy",
    }


def _generate_oil_storage_signal(data: pd.DataFrame, **kwargs) -> dict:
    """Generate signal for oil storage monitoring."""
    # Find the value column
    value_col = None
    for col in ["fill_pct", "fill_level", "storage_level", "ndvi_mean"]:
        if col in data.columns:
            value_col = col
            break

    if value_col is None:
        return {
            "trading_action": "FLAT",
            "signal": "No storage data available",
            "confidence": "Low",
            "baseline_samples": 0,
        }

    baseline_mean, baseline_std, baseline_samples, current_value = _get_seasonal_baseline(data, value_col)

    if baseline_mean is None:
        return {
            "trading_action": "FLAT",
            "signal": "Insufficient historical data",
            "confidence": "Low",
            "baseline_samples": 0,
        }
    # Handle None/NaN in current_value
    if current_value is None or (isinstance(current_value, float) and pd.isna(current_value)):
        current_value = baseline_mean  # Use baseline as fallback
    # Calculate change from baseline
    storage_change = current_value - baseline_mean
    storage_change_pct = (storage_change / baseline_mean) * 100 if baseline_mean > 0 else 0

    # Generate signal based on storage levels
    # High storage = oversupply = SHORT (prices down)
    # Low storage = supply concerns = LONG (prices up)

    threshold = kwargs.get("threshold", 5.0)  # 5% threshold

    if storage_change > threshold:
        trading_action = "SHORT"
        signal = "Short oil (high storage)"
        confidence = "High" if storage_change_pct > 10 else "Medium"
    elif storage_change < -threshold:
        trading_action = "LONG"
        signal = "Long oil (low storage)"
        confidence = "High" if abs(storage_change_pct) > 10 else "Medium"
    else:
        trading_action = "FLAT"
        signal = "Neutral storage levels"
        confidence = "Low"

    return {
        "trading_action": trading_action,
        "signal": signal,
        "confidence": confidence,
        "baseline_samples": baseline_samples,
        "baseline_value": round(baseline_mean, 2),
        "current_value": round(current_value, 2),
        "storage_change": round(storage_change, 2),
        "storage_change_pct": round(storage_change_pct, 2),
    }


def _generate_chokepoint_signal(data: pd.DataFrame, **kwargs) -> dict:
    """Generate signal for chokepoint/port monitoring."""
    # Find the value column
    value_col = None
    for col in ["detections", "throughput", "vessel_count", "count"]:
        if col in data.columns:
            value_col = col
            break

    if value_col is None:
        return {
            "trading_action": "FLAT",
            "signal": "No throughput data available",
            "confidence": "Low",
            "baseline_samples": 0,
        }

    baseline_mean, baseline_std, baseline_samples, current_value = _get_seasonal_baseline(data, value_col)

    if baseline_mean is None:
        return {
            "trading_action": "FLAT",
            "signal": "Insufficient historical data",
            "confidence": "Low",
            "baseline_samples": 0,
        }

    # Handle None/NaN in current_value
    if current_value is None or (isinstance(current_value, float) and pd.isna(current_value)):
        current_value = baseline_mean  # Use baseline as fallback

    # Calculate change from baseline
    throughput_change = current_value - baseline_mean
    throughput_change_pct = (throughput_change / baseline_mean) * 100 if baseline_mean > 0 else 0

    # Generate signal based on throughput
    # Low throughput = supply disruption = LONG (prices up)
    # High throughput = good flow = neutral/SHORT

    threshold = kwargs.get("threshold", 0.5)

    if throughput_change < -threshold * baseline_mean:
        trading_action = "LONG"
        signal = "Long (throughput disruption)"
        confidence = "High" if abs(throughput_change_pct) > 20 else "Medium"
    elif throughput_change > threshold * baseline_mean:
        trading_action = "SHORT"
        signal = "Short (high throughput)"
        confidence = "Medium"
    else:
        trading_action = "FLAT"
        signal = "Normal throughput"
        confidence = "Low"

    return {
        "trading_action": trading_action,
        "signal": signal,
        "confidence": confidence,
        "baseline_samples": baseline_samples,
        "baseline_value": round(baseline_mean, 2),
        "current_value": round(current_value, 2),
        "throughput_change": round(throughput_change, 2),
        "throughput_change_pct": round(throughput_change_pct, 2),
    }
