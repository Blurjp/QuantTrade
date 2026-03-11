"""
Optimized signal generation with lower thresholds.
"""
from datetime import date, timedelta
from typing import Dict, Optional
import pandas as pd
import numpy as np


def generate_agricultural_signal_optimized(
    crop_data: pd.DataFrame,
    threshold: float = 0.03,  # 降低至3% (原来是5-10%)
) -> Dict:
    """
    Generate signal for crop yields with lower threshold.
    
    Signal logic:
    - NDVI above trend → SHORT crop (bumper harvest)
    - NDVI below trend → LONG crop (supply concerns)
    """
    if crop_data.empty:
        return {"signal": "No data", "confidence": "Low", "actionability": "Ignore"}
    
    current_ndvi = crop_data.iloc[-1]['ndvi_mean']
    
    # Compare to rolling baseline
    baseline_ndvi = crop_data['ndvi_mean'].rolling(7, min_periods=3).mean().iloc[-1]
    
    if baseline_ndvi > 0:
        ndvi_pct = current_ndvi / baseline_ndvi
        ndvi_change = (current_ndvi - baseline_ndvi) / baseline_ndvi
    else:
        ndvi_pct = 1.0
        ndvi_change = 0
    
    # Signal logic with lower threshold
    if ndvi_change > threshold:
        signal = "Short crop (bumper harvest)"
        confidence = "High" if ndvi_change > threshold * 2 else "Medium"
        actionability = "Actionable"
        bias = "Bearish crop prices"
        trading_action = "SHORT"
    elif ndvi_change < -threshold:
        signal = "Long crop (supply concerns)"
        confidence = "High" if ndvi_change < -threshold * 2 else "Medium"
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
        "ndvi_current": current_ndvi,
        "ndvi_baseline": baseline_ndvi,
        "ndvi_change": ndvi_change,
        "ndvi_pct": ndvi_pct,
    }


def generate_chokepoint_signal_optimized(
    throughput_data: pd.DataFrame,
    threshold: float = 0.15,  # 降低至15% (原来是30%)
) -> Dict:
    """
    Generate signal for shipping chokepoints with lower threshold.
    
    Signal logic:
    - Low throughput → LONG disruption (supply risk)
    - High throughput → SHORT disruption (normal flow)
    """
    if throughput_data.empty:
        return {"signal": "No data", "confidence": "Low", "actionability": "Ignore"}
    
    current_throughput = throughput_data.iloc[-1].get('detections', 0)
    
    # Calculate baseline
    baseline = throughput_data['detections'].rolling(7, min_periods=3).mean().iloc[-1]
    
    if baseline > 0:
        throughput_change = (current_throughput - baseline) / baseline
    else:
        throughput_change = 0
    
    # Signal logic with lower threshold
    if throughput_change < -threshold:
        signal = "Long disruption risk"
        confidence = "High" if throughput_change < -threshold * 2 else "Medium"
        actionability = "Actionable"
        bias = "Bullish crude / bullish disruption-sensitive assets"
        trading_action = "LONG"
    elif throughput_change > threshold:
        signal = "Short disruption risk"
        confidence = "High" if throughput_change > threshold * 2 else "Medium"
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
        "bias": bias,
        "trading_action": trading_action,
        "throughput_current": current_throughput,
        "throughput_baseline": baseline,
        "throughput_change": throughput_change,
    }


if __name__ == "__main__":
    print("🔧 Optimized Signal Generator")
    print("="*60)
    print()
    print("改进:")
    print("  • Agricultural: 阈值 10% → 3%")
    print("  • Chokepoint: 阈值 30% → 15%")
    print()
    print("预期效果:")
    print("  • 更多long/short信号")
    print("  • 更少neutral信号")
    print("  • 更高的信号质量")
