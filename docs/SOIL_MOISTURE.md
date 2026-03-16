# Soil Moisture Monitoring - User Guide

## 💧 Overview

The soil moisture monitoring module uses satellite data to track soil moisture levels for predicting crop yields, drought conditions, and irrigation needs. Critical indicator for agricultural productivity.

**Data Source:** SMAP (Soil Moisture Active Passive) and Sentinel-1
**Update Frequency:** Daily
**Latency:** 1-3 days
**Cost:** Free (via NASA/Planetary Computer)

## 📊 How It Works

### Signal Logic

**Drought Conditions (moisture < -25% normal):**
```
Soil moisture ↓ → Crop stress → SHORT agricultural commodities
Critical period: Higher confidence
```

**Optimal Conditions (moisture +5% to +15%):**
```
Optimal moisture → Excellent growth → LONG agricultural commodities
```

**Waterlogged (moisture > +25%):**
```
Excess moisture → Root damage → SHORT agricultural commodities
```

### Detection Method

1. **Fetch soil moisture data** for target region
2. **Calculate baseline** (90-day average)
3. **Calculate moisture anomaly** (current - baseline)
4. **Determine moisture status**
5. **Generate signal** with confidence score

### Soil Moisture Status Thresholds

- **Severe Drought:** Moisture < -40% of baseline
- **Drought:** Moisture -25% to -40% of baseline
- **Dry:** Moisture -15% to -25% of baseline
- **Slightly Dry:** Moisture -5% to -15% of baseline
- **Normal:** Moisture -5% to +5% of baseline
- **Optimal:** Moisture +5% to +15% of baseline
- **Wet:** Moisture +15% to +25% of baseline
- **Waterlogged:** Moisture > +25% of baseline

### Critical Season Multiplier

During critical growing months (varies by region and crop), moisture anomalies have 1.5x impact on signal confidence.

## 🗺️ Monitored Regions (9)

### USA Crop Regions (2 regions)

| Region | Baseline | Critical Months | Crops | Trading Instruments |
|--------|----------|-----------------|-------|---------------------|
| **US Midwest Corn Belt** | 0.25 m³/m³ | Apr-Sep | Corn, Soybeans | CORN, SOYB, WEAT |
| **US Great Plains Wheat** | 0.18 m³/m³ | Mar-Jul | Wheat, Sorghum | WEAT, KWK, SORGHUM |

### South America (2 regions)

| Region | Baseline | Critical Months | Crops | Trading Instruments |
|--------|----------|-----------------|-------|---------------------|
| **Brazil Central Soybeans** | 0.28 m³/m³ | Oct-Mar | Soybeans, Corn | SOYB, CORN, COTTON |
| **Argentina Pampas** | 0.22 m³/m³ | Oct-Mar | Soybeans, Corn, Wheat | SOYB, CORN, WEAT |

### Europe (1 region)

| Region | Baseline | Critical Months | Crops | Trading Instruments |
|--------|----------|-----------------|-------|---------------------|
| **Central European Plains** | 0.24 m³/m³ | Apr-Aug | Wheat, Barley, Corn | WEAT, EXI1, CORN |

### Asia (2 regions)

| Region | Baseline | Critical Months | Crops | Trading Instruments |
|--------|----------|-----------------|-------|---------------------|
| **India Gangetic Plain** | 0.20 m³/m³ | Jun-Mar | Rice, Wheat, Sugar | RICE, WHEAT, SUGAR |
| **China North Plain** | 0.18 m³/m³ | Mar-Oct | Wheat, Corn | FXI, WEAT, CORN |

### Africa (1 region)

| Region | Baseline | Critical Months | Crops | Trading Instruments |
|--------|----------|-----------------|-------|---------------------|
| **Africa Sahel Belt** | 0.12 m³/m³ | Jun-Oct | Millet, Sorghum | COTTON, COCOA, SHEA |

### Australia (1 region)

| Region | Baseline | Critical Months | Crops | Trading Instruments |
|--------|----------|-----------------|-------|---------------------|
| **Australia Wheat Belt** | 0.15 m³/m³ | May-Oct | Wheat, Barley | WEAT, AWB, BARLEY |

## 🚀 Usage

### Generate signals for all regions

```bash
python3 -m pipeline.soil_moisture
```

### Generate signal for specific region

```python
from pipeline.soil_moisture import SoilMoistureMonitor

monitor = SoilMoistureMonitor()

# Generate signal for US Midwest
signal = monitor.generate_signal("usa_midwest")

print(f"Direction: {signal['direction']}")
print(f"Confidence: {signal['confidence']}%")
print(f"Surface Moisture: {signal['surface_moisture']:.3f} m³/m³")
print(f"Status: {signal['status']}")
print(f"Anomaly: {signal['moisture_anomaly_pct']:+.1f}%")
print(f"PAW: {signal['plant_available_water']:.2f}")
print(f"Irrigation Need: {signal['irrigation_need']:.1f}%")
print(f"Instruments: {signal['instruments']}")
```

### Get regional summary

```python
summary = monitor.get_regional_summary()
print(f"Monitoring {summary['total_regions']} regions")
print(f"Total crops monitored: {len(set(c for r in summary['regions'].values() for c in r['crops']))}")
```

## 📈 Signal Output

### Single Region Signal

```json
{
  "region_id": "usa_midwest",
  "region_name": "US Midwest Corn Belt",
  "region_type": "row_crops",
  "country": "USA",
  "date": "2026-03-16",
  "signal_type": "soil_moisture",
  "direction": "short",
  "confidence": 72.5,
  "rationale": "Severe soil moisture deficit in US Midwest Corn Belt. -28.3% below normal. Crop stress likely.",
  "instruments": ["CORN", "SOYB", "WEAT"],
  "surface_moisture": 0.179,
  "root_zone_moisture": 0.152,
  "moisture_anomaly_pct": -28.3,
  "status": "drought",
  "plant_available_water": 0.39,
  "irrigation_need": 22.0,
  "is_critical_season": false,
  "baseline_moisture": 0.25,
  "moisture_z_score": -1.95,
  "impact_score": 42.5,
  "soil_type": "loam",
  "data_quality": "good",
  "timestamp": "2026-03-16T12:13:45.123456"
}
```

### Daily Summary

```json
{
  "date": "2026-03-16",
  "total_regions": 9,
  "signals_generated": 9,
  "long_signals": 2,
  "short_signals": 3,
  "neutral_signals": 4,
  "drought_regions": 2,
  "optimal_regions": 1,
  "waterlogged_regions": 0,
  "critical_season_regions": 4,
  "by_region_type": {
    "row_crops": {"count": 8, "long": 2, "short": 2, "neutral": 4},
    "rainfed": {"count": 1, "long": 0, "short": 1, "neutral": 0}
  },
  "signals": [...]
}
```

## 🎯 Trading Strategy

### Recommended Approach

1. **Monitor critical seasons:**
   - Higher impact during growing months
   - Lower impact outside growing season

2. **Combine with precipitation:**
   - Soil moisture + precipitation = stronger signal
   - Cross-validate drought detection

3. **Irrigation monitoring:**
   - Track irrigation need scores
   - Assess irrigation capacity

### Drought Trading Strategy

**During Drought (moisture < -25%):**
```
SHORT:
  • CORN, SOYB (yield reduction)
  • WEAT (crop stress)
  
Hedge:
  • LONG regions with optimal moisture
```

**Example - US Midwest Drought:**
```
June (critical season):
  • Moisture -30% → SHORT CORN, SOYB
  • Confidence: 75-80%
  • Position size: 2-3%

March (not critical):
  • Moisture -30% → SHORT CORN, SOYB
  • Confidence: 60-65%
  • Position size: 1-2%
```

### Optimal Moisture Strategy

**Optimal Conditions (+5% to +15%):**
```
LONG:
  • Affected crops
  • Excellent growth potential
  
Confidence: 58-65%
```

### Waterlogged Strategy

**Waterlogged (> +25%):**
```
SHORT:
  • All affected crops
  • Root damage risk
  • Delayed planting/harvest
```

### Regional Strategy Examples

**US Midwest (Apr-Sep critical):**
```
Drought in June → SHORT CORN, SOYB
Optimal in June → LONG CORN, SOYB
```

**Brazil (Oct-Mar critical):**
```
Drought in Jan → SHORT SOYB
Optimal in Jan → LONG SOYB
```

**Australia (May-Oct critical):**
```
Drought in Aug → SHORT WEAT
Optimal in Aug → LONG WEAT
```

### Risk Management

```
✅ Monitor critical seasons
✅ Use smaller positions outside critical season
✅ Combine with precipitation data
✅ Track irrigation capacity
✅ Consider soil types
```

## 📊 Historical Performance

**Signal Accuracy:** TBD (need 20+ signals for statistical significance)

**Expected Lead Time:**
- Crop prices: 1-3 weeks
- Yield reports: 4-8 weeks
- Drought declarations: 2-4 weeks

## 🔧 Technical Details

### Data Processing Pipeline

```
1. Download SMAP/Sentinel-1 data (NASA/Planetary Computer)
2. Extract regional soil moisture
3. Calculate metrics:
   - Surface moisture (0-5cm)
   - Root zone moisture (0-1m)
   - Plant available water
   - Irrigation need
4. Compare to 90-day baseline
5. Generate signal with confidence score
```

### Quality Control

- **Vegetation filtering:** Remove densely vegetated areas
- **Cloud screening:** Remove cloudy observations
- **Spatial averaging:** Regional mean moisture
- **Temporal smoothing:** 7-day running mean

### Soil Moisture Ranges

| Soil Type | Wilting Point | Field Capacity | Optimal Range |
|-----------|---------------|----------------|---------------|
| Sand | 0.05 | 0.15 | 0.08-0.12 |
| Sandy Loam | 0.10 | 0.20 | 0.12-0.17 |
| Loam | 0.15 | 0.30 | 0.18-0.25 |
| Clay Loam | 0.20 | 0.35 | 0.23-0.30 |
| Clay | 0.25 | 0.40 | 0.28-0.35 |

### Plant Available Water (PAW)

```
PAW = (current_moisture - wilting_point) / (field_capacity - wilting_point)

Range: 0 to 1
  • < 0.25: Severe stress
  • 0.25-0.50: Moderate stress
  • 0.50-0.75: Adequate
  • > 0.75: Excellent
```

## 📁 File Locations

```
outputs/soil_moisture/
├── signal_usa_midwest_2026-03-16.json
├── signal_brazil_central_2026-03-16.json
├── ...
└── summary_2026-03-16.json
```

## 🔮 Future Enhancements

### Phase 2 (Next 2-4 weeks)

1. **Real data integration:**
   - Connect to NASA SMAP API
   - Download actual SMAP data
   - Remove simulation code

2. **Additional metrics:**
   - Evapotranspiration
   - Soil temperature
   - Groundwater levels

3. **Additional regions:**
   - Russia wheat
   - Ukraine corn
   - Indonesia palm oil

### Phase 3 (1-2 months)

4. **Machine learning:**
   - Train on historical yield data
   - Predict crop yields
   - Optimize trading thresholds

5. **Real-time alerts:**
   - Discord notifications
   - Email alerts
   - Web dashboard

6. **Drought forecasting:**
   - 7-14 day forecast
   - Seasonal outlooks
   - Climate model integration

## 📚 References

- [SMAP Mission](https://smap.jpl.nasa.gov/)
- [Sentinel-1](https://sentinel.esa.int/web/sentinel/missions/sentinel-1)
- [NASA Earth Data](https://earthdata.nasa.gov/)
- [Soil Moisture Basics](https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/survey/office/ssr10/tr/?cid=nrcs144p2_074233)

## ⚠️ Limitations

1. **Latency:** 1-3 day delay from satellite
2. **Spatial resolution:** Regional, not field-level
3. **Vegetation interference:** Dense canopy affects readings
4. **Soil heterogeneity:** Variable soil types within region
5. **Sample size:** Need more historical data

## 💡 Best Practices

1. **Focus on critical seasons**
   - Higher confidence during growing months
   - Lower confidence outside

2. **Combine with precipitation**
   - Soil moisture + precipitation = stronger signal
   - Cross-validate drought detection

3. **Start small**
   - Use 1-2% position sizes
   - Build confidence over time

4. **Track performance**
   - Log all signals
   - Calculate accuracy
   - Refine thresholds

5. **Consider irrigation**
   - Irrigated regions less sensitive
   - Rainfed regions more critical

## 🆚 Comparison to Other Signals

| Signal Type | Lead Time | Accuracy | Best For |
|-------------|-----------|----------|----------|
| **Soil Moisture** | 1-3 weeks | TBD | Agriculture |
| Vegetation Health | 2-4 weeks | TBD | Agriculture |
| Precipitation | 2-4 weeks | TBD | Agriculture |
| SST/ENSO | 2-4 weeks | 70-80% | Agriculture |
| Atmospheric | 1-2 weeks | TBD | Industrial |

## 🌾 Agricultural Applications

This module is particularly valuable for:

1. **Crop Yield Prediction:**
   - Early drought detection
   - Yield forecasting
   - Production estimates

2. **Commodity Trading:**
   - Time entries/exits
   - Regional arbitrage
   - Cross-commodity strategies

3. **Irrigation Planning:**
   - Irrigation scheduling
   - Water resource management
   - Drought preparedness

4. **Risk Management:**
   - Weather risk hedging
   - Crop insurance pricing
   - Supply chain planning

---

**Note:** This module currently uses simulated data for demonstration. For production use, connect to the NASA SMAP API to fetch real soil moisture data.
