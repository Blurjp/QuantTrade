# Precipitation Monitoring - User Guide

## 🌧️ Overview

The precipitation monitoring module uses satellite data to track global rainfall patterns for predicting crop yields, agricultural production, and commodity prices. Leading indicator for agricultural markets.

**Data Source:** GPM (Global Precipitation Measurement) and IMERG
**Update Frequency:** Daily
**Latency:** 1-3 days
**Cost:** Free (via NASA GES DISC)

## 📊 How It Works

### Signal Logic

**Agricultural Regions:**
```
Drought (precipitation < -20% normal):
  • Crop yields ↓ → SHORT agricultural commodities
  
Flood (precipitation > +40% normal):
  • Crop damage → SHORT agricultural commodities
  
Normal precipitation:
  • Good growing conditions → LONG agricultural commodities
```

### Detection Method

1. **Fetch precipitation data** for target region
2. **Calculate baseline** (90-day average)
3. **Calculate precipitation anomaly** (current - baseline)
4. **Determine drought/flood status**
5. **Generate signal** with confidence score

### Status Thresholds

- **Severe Drought:** Precipitation < -40% of baseline
- **Drought:** Precipitation -20% to -40% of baseline
- **Dry:** Precipitation -10% to -20% of baseline
- **Normal:** Precipitation -10% to +10% of baseline
- **Slightly Wet:** Precipitation +10% to +20% of baseline
- **Wet:** Precipitation +20% to +40% of baseline
- **Flood:** Precipitation > +40% of baseline

### Critical Season Multiplier

During critical growing months (varies by region and crop), precipitation anomalies have 1.5x impact on signal confidence.

## 🗺️ Monitored Regions (9)

### USA Crop Regions (3 regions)

| Region | Baseline | Critical Months | Crops | Trading Instruments |
|--------|----------|-----------------|-------|---------------------|
| **US Corn Belt** | 85 mm/month | Apr-Aug | Corn, Soybeans | CORN, SOYB, WEAT |
| **US Winter Wheat Belt** | 65 mm/month | Mar-Jun | Winter Wheat | WEAT, KWK |
| **US Cotton Belt** | 95 mm/month | Apr-Sep | Cotton | COTTON, BAL |

### South America (2 regions)

| Region | Baseline | Critical Months | Crops | Trading Instruments |
|--------|----------|-----------------|-------|---------------------|
| **Brazil Soybean Region** | 180 mm/month | Oct-Mar | Soybeans, Corn | SOYB, CORN |
| **Argentina Pampas** | 95 mm/month | Oct-Mar | Soybeans, Corn, Wheat | SOYB, CORN, WEAT |

### Asia (3 regions)

| Region | Baseline | Critical Months | Crops | Trading Instruments |
|--------|----------|-----------------|-------|---------------------|
| **India Monsoon Region** | 250 mm/month | Jun-Oct | Cotton, Sugar, Rice, Tea | COTTON, SUGAR, RICE, TEA |
| **China Wheat Region** | 70 mm/month | Mar-Jun, Sep-Oct | Wheat, Corn | FXI, WEAT |
| **Australia Wheat Belt** | 50 mm/month | May-Oct | Wheat, Barley | WEAT, AWB |

### Africa (1 region)

| Region | Baseline | Critical Months | Crops | Trading Instruments |
|--------|----------|-----------------|-------|---------------------|
| **West Africa Cocoa Belt** | 150 mm/month | Mar-Oct | Cocoa | NIB, CHOC |

## 🚀 Usage

### Generate signals for all regions

```bash
python3 -m pipeline.precipitation
```

### Generate signal for specific region

```python
from pipeline.precipitation import PrecipitationMonitor

monitor = PrecipitationMonitor()

# Generate signal for US Corn Belt
signal = monitor.generate_signal("usa_corn_belt")

print(f"Direction: {signal['direction']}")
print(f"Confidence: {signal['confidence']}%")
print(f"Precipitation: {signal['current_precip_mm']:.1f} mm/month")
print(f"Status: {signal['status']}")
print(f"Anomaly: {signal['precip_anomaly_pct']:+.1f}%")
print(f"Critical Season: {signal['is_critical_season']}")
print(f"Instruments: {signal['instruments']}")
```

### Get regional summary

```python
summary = monitor.get_regional_summary()
print(f"Monitoring {summary['total_regions']} agricultural regions")
print(f"Total crops monitored: {len(set(c for r in summary['regions'].values() for c in r['crops']))}")
```

## 📈 Signal Output

### Single Region Signal

```json
{
  "region_id": "usa_corn_belt",
  "region_name": "US Corn Belt",
  "region_type": "row_crops",
  "country": "USA",
  "date": "2026-03-16",
  "signal_type": "precipitation",
  "direction": "short",
  "confidence": 72.5,
  "rationale": "Drought conditions in US Corn Belt. Precipitation -25.3% below normal. Crop yield at risk.",
  "instruments": ["CORN", "SOYB", "WEAT"],
  "current_precip_mm": 63.5,
  "precip_anomaly_pct": -25.3,
  "status": "drought",
  "is_critical_season": false,
  "baseline_precip_mm": 85.0,
  "precip_z_score": -1.82,
  "impact_score": 37.9,
  "crops": ["corn", "soybeans"],
  "data_quality": "good",
  "timestamp": "2026-03-16T08:11:23.456789"
}
```

### Daily Summary

```json
{
  "date": "2026-03-16",
  "total_regions": 9,
  "signals_generated": 9,
  "long_signals": 4,
  "short_signals": 3,
  "neutral_signals": 2,
  "drought_regions": 2,
  "flood_regions": 0,
  "critical_season_regions": 5,
  "by_region_type": {
    "row_crops": {"count": 7, "long": 3, "short": 2, "neutral": 2},
    "monsoon_agriculture": {"count": 1, "long": 1, "short": 0, "neutral": 0},
    "tree_crops": {"count": 1, "long": 0, "short": 1, "neutral": 0}
  },
  "signals": [...]
}
```

## 🎯 Trading Strategy

### Recommended Approach

1. **Monitor critical seasons:**
   - Higher impact during growing months
   - Lower impact outside growing season

2. **Combine with other signals:**
   - SST/ENSO for long-term outlook
   - Weather models for short-term

3. **Regional diversification:**
   - Monitor multiple regions
   - Cross-regional hedging

### Drought Trading Strategy

**During Drought (precipitation < -20%):**
```
SHORT:
  • CORN, SOYB (yield reduction)
  • WEAT (drought stress)
  • COTTON (water-intensive)
  
Hedge:
  • LONG regions with normal rain
```

**Example - US Corn Belt Drought:**
```
March (not critical season):
  • Confidence: 60-65%
  • Position size: 1-2%

June (critical season):
  • Confidence: 75-80%
  • Position size: 2-3%
```

### Flood Trading Strategy

**During Flood (precipitation > +40%):**
```
SHORT:
  • All affected crops
  • Quality degradation
  • Harvest delays
  
Special case:
  • Rice may benefit from flooding
```

### Normal Conditions Strategy

**Normal Precipitation (±10%):**
```
LONG:
  • Affected crops
  • Good yield potential
  
Confidence: 55-60%
```

### Regional Strategy Examples

**US Corn Belt (Apr-Aug critical):**
```
Drought in June → SHORT CORN, SOYB
Normal in June → LONG CORN, SOYB
Flood in harvest → SHORT CORN
```

**Brazil Soybeans (Oct-Mar critical):**
```
Drought in Dec → SHORT SOYB
Normal in Jan → LONG SOYB
```

**India Monsoon (Jun-Oct critical):**
```
Weak monsoon → SHORT COTTON, SUGAR
Normal monsoon → LONG RICE, SUGAR
```

### Risk Management

```
✅ Monitor critical seasons
✅ Use smaller positions outside critical season
✅ Diversify across regions
✅ Combine with weather forecasts
✅ Consider crop insurance data
```

## 📊 Historical Performance

**Signal Accuracy:** TBD (need 20+ signals for statistical significance)

**Expected Lead Time:**
- Crop prices: 2-4 weeks
- Futures markets: 1-2 weeks
- Spot prices: 1-4 weeks

## 🔧 Technical Details

### Data Processing Pipeline

```
1. Download GPM/IMERG data (NASA GES DISC)
2. Extract regional precipitation
3. Calculate metrics:
   - Daily precipitation (mm)
   - Monthly estimate
   - Anomaly from baseline
   - Drought/flood status
4. Determine critical season
5. Generate signal with confidence score
```

### Quality Control

- **Cloud screening:** Remove invalid observations
- **Spatial averaging:** Regional mean precipitation
- **Temporal smoothing:** 7-day running mean
- **Anomaly calculation:** Remove seasonal cycle

### Precipitation Ranges

| Crop Type | Normal Range | Drought | Flood |
|-----------|--------------|---------|-------|
| Row crops | 60-100 mm/month | <48 mm | >140 mm |
| Monsoon crops | 200-300 mm/month | <160 mm | >420 mm |
| Tree crops | 120-180 mm/month | <96 mm | >252 mm |

## 📁 File Locations

```
outputs/precipitation/
├── signal_usa_corn_belt_2026-03-16.json
├── signal_brazil_soybeans_2026-03-16.json
├── ...
└── summary_2026-03-16.json
```

## 🔮 Future Enhancements

### Phase 2 (Next 2-4 weeks)

1. **Real data integration:**
   - Connect to NASA GES DISC API
   - Download actual GPM/IMERG data
   - Remove simulation code

2. **Soil moisture:**
   - Add soil moisture monitoring
   - Improve drought detection
   - Better yield predictions

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

6. **Forecasting:**
   - 7-14 day precipitation forecast
   - Seasonal outlooks
   - ENSO integration

## 📚 References

- [GPM Mission](https://gpm.nasa.gov/)
- [IMERG Data](https://gpm.nasa.gov/data/imerg)
- [NASA GES DISC](https://disc.gsfc.nasa.gov/)
- [Crop Weather](https://www.nass.usda.gov/Research_and_Science/Crop-Weather/index.php)

## ⚠️ Limitations

1. **Latency:** 1-3 day delay from satellite
2. **Spatial resolution:** Regional, not field-level
3. **Crop stage:** Doesn't account for growth stage
4. **Irrigation:** Doesn't account for irrigated areas
5. **Sample size:** Need more historical data

## 💡 Best Practices

1. **Focus on critical seasons**
   - Higher confidence during growing months
   - Lower confidence outside

2. **Combine with other data**
   - Soil moisture
   - Temperature
   - Crop progress reports

3. **Start small**
   - Use 1-2% position sizes
   - Build confidence over time

4. **Track performance**
   - Log all signals
   - Calculate accuracy
   - Refine thresholds

5. **Understand crop cycles**
   - Planting → Growing → Harvest
   - Different sensitivity at each stage

## 🆚 Comparison to Other Signals

| Signal Type | Lead Time | Accuracy | Best For |
|-------------|-----------|----------|----------|
| **Precipitation** | 2-4 weeks | TBD | Agriculture |
| SST/ENSO | 2-4 weeks | 70-80% | Agriculture |
| Atmospheric | 1-2 weeks | TBD | Industrial |
| Thermal IR | 2-4 weeks | TBD | Production |
| Solar | 1-2 weeks | TBD | Energy |

## 🌾 Agricultural Applications

This module is particularly valuable for:

1. **Crop Yield Prediction:**
   - Early warning of drought
   - Yield forecasting
   - Production estimates

2. **Commodity Trading:**
   - Time entries/exits
   - Regional arbitrage
   - Cross-commodity strategies

3. **Risk Management:**
   - Weather risk hedging
   - Crop insurance pricing
   - Supply chain planning

4. **Food Security:**
   - Global food supply monitoring
   - Early warning systems
   - Humanitarian response

---

**Note:** This module currently uses simulated data for demonstration. For production use, connect to the NASA GES DISC API to fetch real GPM and IMERG data.
