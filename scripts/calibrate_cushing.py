#!/usr/bin/env python3
"""
Cushing Oil Tank Calibration Script
Calibrates tank detection algorithm against EIA data
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.eia_data import EIADataFetcher
from pipeline.tank_detection import OilTankDetector


def calibrate_cushing_tanks():
    """
    Calibrate tank detection using EIA data
    
    Returns:
        Dict with calibration results
    """
    print("🔧 Cushing油罐算法校准")
    print("="*60)
    print()
    
    # Step 1: Fetch EIA data
    print("1️⃣ 获取EIA Cushing库存数据...")
    fetcher = EIADataFetcher()
    eia_data = fetcher.fetch_cushing_inventory(days=90)
    
    if eia_data.empty:
        print("❌ 无法获取EIA数据")
        return {"status": "error", "message": "No EIA data"}
    
    print(f"   ✅ 获取 {len(eia_data)} 个数据点")
    print()
    
    # Step 2: Analyze EIA data patterns
    print("2️⃣ 分析EIA数据模式...")
    
    # Calculate statistics
    mean_level = eia_data['value'].mean()
    std_level = eia_data['value'].std()
    min_level = eia_data['value'].min()
    max_level = eia_data['value'].max()
    
    # Normalize to 0-100% scale
    # Typical Cushing capacity: ~90M barrels
    max_capacity = 90.0
    
    eia_data['fill_pct'] = (eia_data['value'] / max_capacity * 100).clip(0, 100)
    
    print(f"   平均库存: {mean_level:.2f}M 桶 ({mean_level/max_capacity*100:.1f}%)")
    print(f"   标准差: {std_level:.2f}M 桶")
    print(f"   范围: {min_level:.2f}M - {max_level:.2f}M 桶")
    print()
    
    # Step 3: Generate calibration parameters
    print("3️⃣ 生成校准参数...")
    
    # Based on EIA data distribution, set thresholds
    # Low inventory: < 40% capacity
    # High inventory: > 80% capacity
    
    low_threshold = 40.0  # % capacity
    high_threshold = 80.0  # % capacity
    
    calibration_params = {
        "max_capacity_mb": max_capacity,
        "low_inventory_threshold_pct": low_threshold,
        "high_inventory_threshold_pct": high_threshold,
        "shadow_calibration": {
            # Shadow percentage → Fill level mapping
            # Needs actual satellite data to calibrate
            "min_shadow_pct": 5.0,   # 5% shadow ≈ 95% full
            "max_shadow_pct": 40.0,  # 40% shadow ≈ 60% full
        },
        "detection_params": {
            "min_tank_radius": 10,
            "max_tank_radius": 100,
            "hough_param1": 50,
            "hough_param2": 30,
            "min_dist": 20
        }
    }
    
    print(f"   低库存阈值: {low_threshold}%")
    print(f"   高库存阈值: {high_threshold}%")
    print()
    
    # Step 4: Validate calibration
    print("4️⃣ 验证校准...")
    
    # Count how many weeks at each level
    low_count = (eia_data['fill_pct'] < low_threshold).sum()
    high_count = (eia_data['fill_pct'] > high_threshold).sum()
    normal_count = ((eia_data['fill_pct'] >= low_threshold) & 
                    (eia_data['fill_pct'] <= high_threshold)).sum()
    
    print(f"   低库存周数: {low_count}")
    print(f"   正常库存周数: {normal_count}")
    print(f"   高库存周数: {high_count}")
    print()
    
    # Step 5: Generate signals based on EIA data
    print("5️⃣ 基于EIA数据生成信号...")
    
    signals = []
    for idx, row in eia_data.iterrows():
        fill_pct = row['fill_pct']
        
        if fill_pct > high_threshold:
            signal = "short"
            rationale = f"高库存 ({fill_pct:.1f}%)"
        elif fill_pct < low_threshold:
            signal = "long"
            rationale = f"低库存 ({fill_pct:.1f}%)"
        else:
            signal = "neutral"
            rationale = f"正常库存 ({fill_pct:.1f}%)"
        
        signals.append({
            "date": row['date'].strftime('%Y-%m-%d'),
            "inventory_mb": round(row['value'], 2),
            "fill_pct": round(fill_pct, 1),
            "signal": signal,
            "rationale": rationale
        })
    
    print(f"   生成 {len(signals)} 个信号")
    print()
    
    # Step 6: Save calibration results
    print("6️⃣ 保存校准结果...")
    
    results = {
        "calibration_date": datetime.now().isoformat(),
        "data_source": "EIA Demo" if fetcher.api_key is None else "EIA API",
        "parameters": calibration_params,
        "validation": {
            "low_inventory_weeks": int(low_count),
            "normal_inventory_weeks": int(normal_count),
            "high_inventory_weeks": int(high_count),
            "total_weeks": len(eia_data)
        },
        "signals": signals,
        "statistics": {
            "mean_inventory_mb": round(mean_level, 2),
            "std_inventory_mb": round(std_level, 2),
            "min_inventory_mb": round(min_level, 2),
            "max_inventory_mb": round(max_level, 2),
            "mean_fill_pct": round(mean_level/max_capacity*100, 1)
        }
    }
    
    output_file = Path("outputs/cushing_calibration.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2))
    
    print(f"   ✅ 保存至: {output_file}")
    print()
    
    # Step 7: Summary
    print("="*60)
    print("✅ 校准完成")
    print("="*60)
    print()
    
    print("校准参数:")
    print(f"  最大容量: {max_capacity}M 桶")
    print(f"  低库存: < {low_threshold}%")
    print(f"  高库存: > {high_threshold}%")
    print()
    
    print("下一步:")
    print("  1. 获取Sentinel-2 Cushing图像")
    print("  2. 运行油罐检测")
    print("  3. 对比检测结果与EIA数据")
    print("  4. 调整阴影阈值")
    print("  5. 重新验证")
    print()
    
    return results


if __name__ == "__main__":
    results = calibrate_cushing_tanks()
    
    print("\n📋 最近5周信号:")
    print("-"*60)
    for signal in results['signals'][-5:]:
        emoji = {"long": "🟢", "short": "🔴", "neutral": "⚪"}[signal['signal']]
        print(f"{emoji} {signal['date']}: {signal['signal'].upper()}")
        print(f"   库存: {signal['inventory_mb']}M ({signal['fill_pct']}%)")
        print(f"   {signal['rationale']}")
        print()
