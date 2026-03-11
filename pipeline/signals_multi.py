"""
Multi-type signal generation for different monitoring categories.

Each monitoring type has its own signal logic.
"""

from datetime import date, timedelta
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np


def _resolve_value_column(data: pd.DataFrame, preferred: str, fallback: str = "detections") -> Optional[str]:
    if preferred in data.columns:
        return preferred
    if fallback in data.columns:
        return fallback
    return None


def _seasonal_baseline(
    data: pd.DataFrame,
    value_column: str,
    week_window: int = 2,
    min_history: int = 3,
) -> Tuple[float, float, int]:
    """Build a same-season baseline with a nearby-week fallback."""
    if data.empty:
        return 0.0, 0.0, 0

    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    current = df.iloc[-1]
    current_week = int(current["date"].isocalendar().week)

    historical = df.iloc[:-1].copy()
    if historical.empty:
        values = df[value_column].dropna()
        return float(values.mean()), float(values.std(ddof=0)), len(values)

    historical["iso_week"] = historical["date"].dt.isocalendar().week.astype(int)
    week_distance = (historical["iso_week"] - current_week).abs()
    wrapped_distance = np.minimum(week_distance, 52 - week_distance)
    seasonal = historical[wrapped_distance <= week_window][value_column].dropna()

    if len(seasonal) < min_history:
        seasonal = historical[value_column].dropna()

    if seasonal.empty:
        return 0.0, 0.0, 0

    return float(seasonal.mean()), float(seasonal.std(ddof=0)), int(len(seasonal))


def generate_chokepoint_signal(
    throughput_data: pd.DataFrame,
    baseline_window: int = 30,
) -> Dict:
    """
    Generate signal for shipping chokepoints.
    
    Signal logic:
    - Long disruption risk → SHORT oil (supply crunch expected)
    - Short disruption risk → CLOSE short (flow normal)
    """
    if throughput_data.empty:
        return {"signal": "No data", "confidence": "Low", "actionability": "Ignore"}

    value_column = _resolve_value_column(throughput_data, "throughput_index")
    if value_column is None:
        return {"signal": "Missing throughput column", "confidence": "Low", "actionability": "Ignore", "trading_action": "FLAT"}
    
    # Calculate rolling baseline
    recent = throughput_data.tail(baseline_window)
    baseline_mean = recent[value_column].mean()
    baseline_std = recent[value_column].std()
    
    current_throughput = throughput_data.iloc[-1][value_column]
    
    if baseline_std > 0:
        zscore = (current_throughput - baseline_mean) / baseline_std
    else:
        zscore = 0
    
    # Signal logic
    if current_throughput < 0.5 * baseline_mean:
        signal = "Long disruption risk"
        confidence = "High" if zscore < -1.5 else "Medium"
        actionability = "Actionable"
        bias = "Bullish crude / bullish disruption-sensitive assets"
        trading_action = "LONG"
    elif current_throughput > 1.2 * baseline_mean:
        signal = "Short disruption risk"
        confidence = "High" if zscore > 1.5 else "Medium"
        actionability = "Actionable"
        bias = "Bearish crude risk premium"
        trading_action = "SHORT"
    else:
        signal = "No trade"
        confidence = "Low"
        actionability = "Ignore"
        bias = "Neutral"
        trading_action = "FLAT"
    
    return {
        "signal": signal,
        "confidence": confidence,
        "actionability": actionability,
        "trading_action": trading_action,
        "bias": bias,
        "zscore": zscore,
        "throughput_current": current_throughput,
        "baseline_mean": baseline_mean,
    }


def generate_retail_signal(
    traffic_data: pd.DataFrame,
    baseline_window: int = 52,  # 1 year of weekly data
) -> Dict:
    """
    Generate signal for retail parking traffic.
    
    Signal logic:
    - Traffic above baseline → LONG stock (strong sales expected)
    - Traffic below baseline → SHORT stock (weak sales expected)
    """
    if traffic_data.empty:
        return {"signal": "No data", "confidence": "Low", "actionability": "Ignore"}

    value_column = _resolve_value_column(traffic_data, "vehicle_count")
    if value_column is None:
        return {"signal": "Missing traffic column", "confidence": "Low", "actionability": "Ignore", "trading_action": "FLAT"}
    
    # Compare to same period last year (seasonal adjustment)
    current_traffic = traffic_data.iloc[-1][value_column]
    
    # Get historical same-week data
    current_week = pd.Timestamp(traffic_data.iloc[-1]['date']).week
    historical_same_week = traffic_data[
        pd.to_datetime(traffic_data['date']).dt.week == current_week
    ]
    
    if len(historical_same_week) > 1:
        baseline = historical_same_week[:-1][value_column].mean()
        baseline_std = historical_same_week[:-1][value_column].std()
    else:
        baseline = traffic_data[value_column].mean()
        baseline_std = traffic_data[value_column].std()
    
    if baseline_std > 0:
        zscore = (current_traffic - baseline) / baseline_std
    else:
        zscore = 0
    
    # Signal logic
    traffic_pct = current_traffic / baseline if baseline > 0 else 1
    
    if traffic_pct > 1.15:
        signal = "Long retail traffic"
        confidence = "High" if zscore > 2 else "Medium"
        actionability = "Actionable"
        bias = "Bullish retail earnings"
        trading_action = "LONG"
    elif traffic_pct < 0.85:
        signal = "Short retail traffic"
        confidence = "High" if zscore < -2 else "Medium"
        actionability = "Actionable"
        bias = "Bearish retail earnings"
        trading_action = "SHORT"
    else:
        signal = "No trade"
        confidence = "Low"
        actionability = "Ignore"
        bias = "Neutral"
        trading_action = "FLAT"
    
    return {
        "signal": signal,
        "confidence": confidence,
        "actionability": actionability,
        "bias": bias,
        "trading_action": trading_action,
        "zscore": zscore,
        "traffic_current": current_traffic,
        "traffic_baseline": baseline,
        "traffic_pct": traffic_pct,
    }


def generate_storage_signal(
    tank_data: pd.DataFrame,
    eia_baseline: Optional[float] = None,
) -> Dict:
    """
    Generate signal for oil storage levels.
    
    Signal logic:
    - Storage rising above trend → BEARISH oil
    - Storage falling below trend → BULLISH oil
    """
    if tank_data.empty:
        return {"signal": "No data", "confidence": "Low", "actionability": "Ignore"}
    
    current_level = tank_data.iloc[-1]['fill_pct']
    
    # Trend analysis
    recent = tank_data.tail(4)  # Last 4 weeks
    if len(recent) > 1:
        trend = recent['fill_pct'].diff().mean()  # Weekly change
    else:
        trend = 0
    
    # Compare to EIA if available
    if eia_baseline is not None:
        vs_eia = (current_level - eia_baseline) / eia_baseline if eia_baseline > 0 else 0
    else:
        vs_eia = 0
    
    # Signal logic
    if current_level > 80 and trend > 0:
        signal = "Long storage / Short oil"
        confidence = "High"
        actionability = "Actionable"
        bias = "Bearish crude (oversupply)"
        trading_action = "SHORT"
    elif current_level < 50 and trend < 0:
        signal = "Short storage / Long oil"
        confidence = "High"
        actionability = "Actionable"
        bias = "Bullish crude (tight supply)"
        trading_action = "LONG"
    else:
        signal = "No trade"
        confidence = "Low"
        actionability = "Ignore"
        bias = "Neutral"
        trading_action = "FLAT"
    
    return {
        "signal": signal,
        "confidence": confidence,
        "actionability": actionability,
        "bias": bias,
        "trading_action": trading_action,
        "current_level": current_level,
        "trend": trend,
        "vs_eia": vs_eia,
    }


def generate_agricultural_signal(
    crop_data: pd.DataFrame,
    usda_forecast: Optional[float] = None,
    threshold: float = 0.05,
    week_window: int = 2,
) -> Dict:
    """
    Generate signal for crop yields.
    
    Signal logic:
    - NDVI above trend → SHORT crop (bumper harvest)
    - NDVI below trend → LONG crop (supply concerns)
    """
    if crop_data.empty:
        return {"signal": "No data", "confidence": "Low", "actionability": "Ignore"}
    
    current_ndvi = crop_data.iloc[-1]['ndvi_mean']
    baseline_ndvi, baseline_std, baseline_count = _seasonal_baseline(
        crop_data,
        "ndvi_mean",
        week_window=week_window,
    )

    if baseline_ndvi <= 0:
        baseline_ndvi = float(crop_data['ndvi_mean'].dropna().mean()) if not crop_data['ndvi_mean'].dropna().empty else 0
    
    if baseline_std > 0:
        zscore = (current_ndvi - baseline_ndvi) / baseline_std
    else:
        zscore = 0
    
    # Signal logic
    ndvi_pct = current_ndvi / baseline_ndvi if baseline_ndvi > 0 else 1
    ndvi_change = (current_ndvi - baseline_ndvi) / baseline_ndvi if baseline_ndvi > 0 else 0

    if ndvi_change > threshold:
        signal = "Short crop (bumper harvest)"
        confidence = "High" if zscore > 1.5 else "Medium"
        actionability = "Actionable"
        bias = "Bearish crop prices"
        trading_action = "SHORT"
    elif ndvi_change < -threshold:
        signal = "Long crop (supply concerns)"
        confidence = "High" if zscore < -1.5 else "Medium"
        actionability = "Actionable"
        bias = "Bullish crop prices"
        trading_action = "LONG"
    else:
        signal = "No trade"
        confidence = "Low"
        actionability = "Ignore"
        bias = "Neutral"
        trading_action = "FLAT"
    
    return {
        "signal": signal,
        "confidence": confidence,
        "actionability": actionability,
        "bias": bias,
        "trading_action": trading_action,
        "zscore": zscore,
        "ndvi_current": current_ndvi,
        "ndvi_baseline": baseline_ndvi,
        "ndvi_pct": ndvi_pct,
        "ndvi_change": ndvi_change,
        "baseline_samples": baseline_count,
    }


def generate_auto_inventory_signal(
    inventory_data: pd.DataFrame,
) -> Dict:
    """
    Generate signal for auto dealer inventory.
    
    Signal logic:
    - Inventory rising → SHORT auto (oversupply)
    - Inventory falling → LONG auto (strong demand)
    """
    if inventory_data.empty:
        return {"signal": "No data", "confidence": "Low", "actionability": "Ignore"}

    value_column = _resolve_value_column(inventory_data, "vehicle_count")
    if value_column is None:
        return {"signal": "Missing inventory column", "confidence": "Low", "actionability": "Ignore", "trading_action": "FLAT"}
    
    current_inventory = inventory_data.iloc[-1][value_column]
    
    # Calculate trend
    recent = inventory_data.tail(4)
    if len(recent) > 1:
        baseline = recent[:-1][value_column].mean()
        trend = (current_inventory - baseline) / baseline if baseline > 0 else 0
    else:
        baseline = current_inventory
        trend = 0
    
    # Signal logic
    if trend > 0.15:  # 15% increase
        signal = "Short auto (rising inventory)"
        confidence = "High" if trend > 0.25 else "Medium"
        actionability = "Actionable"
        bias = "Bearish auto stocks"
        trading_action = "SHORT"
    elif trend < -0.15:  # 15% decrease
        signal = "Long auto (falling inventory)"
        confidence = "High" if trend < -0.25 else "Medium"
        actionability = "Actionable"
        bias = "Bullish auto stocks"
        trading_action = "LONG"
    else:
        signal = "No trade"
        confidence = "Low"
        actionability = "Ignore"
        bias = "Neutral"
        trading_action = "FLAT"
    
    return {
        "signal": signal,
        "confidence": confidence,
        "actionability": actionability,
        "bias": bias,
        "trading_action": trading_action,
        "inventory_current": current_inventory,
        "inventory_baseline": baseline,
        "inventory_trend": trend,
    }


# Signal generator registry
SIGNAL_GENERATORS = {
    "chokepoint": generate_chokepoint_signal,
    "port_logistics": generate_chokepoint_signal,  # Same logic
    "retail_parking": generate_retail_signal,
    "oil_storage": generate_storage_signal,
    "agricultural": generate_agricultural_signal,
    "agriculture": generate_agricultural_signal,
    "auto_inventory": generate_auto_inventory_signal,
}


def generate_signal(monitoring_type: str, data: pd.DataFrame, **kwargs) -> Dict:
    """Generate signal based on monitoring type."""
    generator = SIGNAL_GENERATORS.get(monitoring_type)
    if generator is None:
        return {"signal": "Unknown type", "confidence": "Low", "actionability": "Ignore"}
    return generator(data, **kwargs)


if __name__ == "__main__":
    # Test signal generators
    print("Signal Generators:")
    for mtype, func in SIGNAL_GENERATORS.items():
        print(f"  {mtype}: {func.__name__}")
