# Vegetation Health Monitoring - User Guide

## 🌿 Overview

The vegetation health monitoring module uses satellite NDVI/EVI data to track vegetation health for predicting crop yields, forest conditions, and agricultural productivity. Leading indicator for agricultural and forestry markets.

**Data Source:** MODIS (Terra/Aqua) and Sentinel-2
**Update Frequency:** Daily
**Latency:** 1-3 days
**Cost:** Free (via Planetary Computer)

## 📊 How It Works

### Signal Logic

**Crop Regions:**
```
Vegetation stress (NDVI < -10% normal):
  • Crop yields ↓ → SHORT agricultural commodities
  
Excellent vegetation (NDVI > +10% normal):
  • High yield potential → LONG agricultural commodities
  
Normal vegetation:
  • Expected yields → NEUTRAL
```

### Detection Method

1. **Fetch NDVI/EVI data** for target region
2. **Calculate baseline** (90-day average)
3. **Calculate vegetation anomaly** (current - baseline)
4. **Determine stress status**
5. **Generate signal** with confidence score

### Vegetation Status Thresholds

- **Severe Stress:** NDVI < -20% of baseline
- **Stress:** NDVI -10% to -20% of baseline
- **Slight Stress:** NDVI -5% to -10% of baseline
- **Normal:** NDVI -5% to +5% of baseline
- **Good:** NDVI +5% to +10% of baseline
- **Excellent:** NDVI > +10% of baseline

### Critical Season Multiplier

During critical growing months (varies by region and crop), vegetation anomalies have 1.5x impact on signal confidence.

## 🗺️ Monitored Regions (10)

### USA Crop Regions (2 regions)

| Region | Baseline NDVI | Critical Months | Crops | Trading Instruments |
|--------|---------------|-----------------|-------|---------------------|
| **US Corn & Soybeans Belt** | 0.65 | Jun-Sep | Corn, Soybeans | CORN, SOYB |
| **US Wheat Plains** | 0.45 | Apr-Jul | Winter/Spring Wheat | WEAT, KWK |

### South America (2 regions)

| Region | Baseline NDVI | Critical Months | Crops | Trading Instruments |
|--------|---------------|-----------------|-------|---------------------|
| **Brazil Cerrado Soybeans** | 0.60 | Nov-Feb | Soybeans, Corn | SOYB, CORN |
| **Argentina Pampas** | 0.55 | Nov-Mar | Soybeans, Corn, Wheat | SOYB, CORN, WEAT |

### Europe (2 regions)

| Region | Baseline NDVI | Critical Months | Crops | Trading Instruments |
|--------|---------------|-----------------|-------|---------------------|
| **European Wheat Belt** | 0.58 | Apr-Jul | Wheat, Barley | WEAT, EXI1 |
| **Ukraine Grain Region** | 0.52 | Apr-Jul | Wheat, Corn, Sunflower | WEAT, CORN |

### Asia (2 regions)

| Region | Baseline NDVI | Critical Months | Crops | Trading Instruments |
|--------|---------------|-----------------|-------|---------------------|
| **India Punjab Wheat** | 0.50 | Nov-Mar, Oct | Wheat, Rice | RICE, WHEAT |
| **China Northeast Corn Belt** | 0.62 | May-Sep | Corn, Soybeans, Rice | FXI, CORN |

### Forestry (2 regions)

| Region | Baseline NDVI | Monitoring | Products | Trading Instruments |
|--------|---------------|------------|----------|---------------------|
| **Amazon Rainforest** | 0.85 | Year-round | Timber, Pulp | WOOD, PAPER |
| **Indonesia Palm Oil** | 0.75 | Year-round | Palm Oil | PALM, CPO |

## 🚀 Usage

### Generate signals for all regions

```bash
python3 -m pipeline.vegetation_health
```

### Generate signal for specific region

```python
from pipeline.vegetation_health import VegetationHealthMonitor

monitor = VegetationHealthMonitor()

# Generate signal for US Corn Belt
signal = monitor.generate_signal("usa_corn_soybeans")

print(f"Direction: {signal['direction']}")
print(f"Confidence: {signal['confidence']}%")
print(f"NDVI: {signal['current_ndvi']:.3f}")
print(f"Status: {signal['status']}")
print(f"Anomaly: {signal['ndvi_anomaly_pct']:+.1f}%")
print(f"LAI: {signal['lai_estimate']:.2f}")
print(f"Instruments: {signal['instruments']}")
```

### Get regional summary

```python
summary = monitor.get_regional_summary()
print(f"Monitoring {summary['total_regions']} regions")
print(f"Region types: {', '.join(summary['region_types'])}")
```

## 📈 Signal Output

### Single Region Signal

```json
{
  "region_id": "usa_corn_soybeans",
  "region_name": "US Corn & Soybeans Belt",
  "region_type": "row_crops",
  "country": "USA",
  "date": "2026-03-16",
  "signal_type": "vegetation_health",
  "direction": "long",
  "confidence": 68.5,
  "rationale": "Excellent vegetation health in US Corn & Soybeans Belt. NDVI +12.3% above normal. Strong yield potential.",
  "instruments": ["CORN", "SOYB"],
  "current_ndvi": 0.73,
  "current_evi": 0.62,
  "ndvi_anomaly_pct": 12.3,
  "status": "excellent",
  "is_critical_season": false,
  "baseline_ndvi": 0.65,
  "ndvi_z_score": 2.15,
  "impact_score": 18.5,
  "lai_estimate": 4.38,
  "chlorophyll_content": 73.0,
  "data_quality": "good",
  "timestamp": "2026-03-16T08:20:15.123456"
}
```

### Daily Summary

```json
{
  "date": "2026-03-16",
  "total_regions": 10,
  "signals_generated": 10,
  "long_signals": 4,
  "short_signals": 2,
  "neutral_signals": 4,
  "stress_regions": 2,
  "excellent_regions": 1,
  "critical_season_regions": 3,
  "by_region_type": {
    "row_crops": {"count": 8, "long": 3, "short": 2, "neutral": 3},
    "forest": {"count": 1, "long": 0, "short": 0, "neutral": 1},
    "plantation": {"count": 1, "long": 1, "short": 0, "neutral": 0}
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
   - NDVI stress + drought = HIGH confidence SHORT
   - NDVI excellent + good rain = HIGH confidence LONG

3. **Regional diversification:**
   - Monitor multiple regions
   - Cross-regional hedging

### Stress Trading Strategy

**During Vegetation Stress (NDVI < -10%):**
```
SHORT:
  • CORN, SOYB (yield reduction)
  • WEAT (crop stress)
  
Hedge:
  • LONG regions with excellent NDVI
```

**Example - US Corn Belt Stress:**
```
June (critical season):
  • NDVI -15% → SHORT CORN, SOYB
  • Confidence: 75-80%
  • Position size: 2-3%

March (not critical):
  • NDVI -15% → SHORT CORN, SOYB
  • Confidence: 60-65%
  • Position size: 1-2%
```

### Excellent Vegetation Strategy

**Excellent NDVI (> +10%):**
```
LONG:
  • Affected crops
  • High yield potential
  
Confidence: 65-70%
```

### Forestry Strategy

**Forest Stress:**
```
SHORT:
  • WOOD, PAPER (timber/pulp supply)
  • Longer-term signal
  • Lower confidence (forests less sensitive)
```

### Regional Strategy Examples

**US Corn Belt (Jun-Sep critical):**
```
Stress in July → SHORT CORN, SOYB
Excellent in July → LONG CORN, SOYB
```

**Brazil Soybeans (Nov-Feb critical):**
```
Stress in Jan → SHORT SOYB
Excellent in Jan → LONG SOYB
```

**Amazon Rainforest:**
```
Severe stress → SHORT WOOD, PAPER
Monitor for deforestation + climate
```

### Risk Management

```
✅ Monitor critical seasons
✅ Use smaller positions outside critical season
✅ Combine with precipitation data
✅ Diversify across regions
✅ Consider crop growth stages
```

## 📊 Historical Performance

**Signal Accuracy:** TBD (need 20+ signals for statistical significance)

**Expected Lead Time:**
- Crop prices: 2-4 weeks
- Yield reports: 4-8 weeks
- Forestry: 2-6 months

## 🔧 Technical Details

### Data Processing Pipeline

```
1. Download MODIS/Sentinel-2 data (Planetary Computer)
2. Extract regional NDVI/EVI
3. Calculate metrics:
   - NDVI (Normalized Difference Vegetation Index)
   - EVI (Enhanced Vegetation Index)
   - LAI (Leaf Area Index)
   - Chlorophyll content
4. Compare to 90-day baseline
5. Generate signal with confidence score
```

### Vegetation Indices

**NDVI (Normalized Difference Vegetation Index):**
```
NDVI = (NIR - Red) / (NIR + Red)
Range: -1 to 1
  • < 0.2: Bare soil, water, urban
  • 0.2-0.4: Sparse vegetation
  • 0.4-0.6: Moderate vegetation
  • > 0.6: Dense vegetation
```

**EVI (Enhanced Vegetation Index):**
```
EVI = 2.5 × ((NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1))
Range: 0 to 1
Better for dense canopies
```

### NDVI Ranges by Crop Type

| Crop Type | Normal Range | Stress | Excellent |
|-----------|--------------|--------|-----------|
| Row crops | 0.45-0.70 | <0.40 | >0.75 |
| Plantations | 0.70-0.80 | <0.65 | >0.85 |
| Forests | 0.75-0.90 | <0.70 | >0.90 |

## 📁 File Locations

```
outputs/vegetation_health/
├── signal_usa_corn_soybeans_2026-03-16.json
├── signal_brazil_cerrado_2026-03-16.json
├── ...
└── summary_2026-03-16.json
```

## 🔮 Future Enhancements

### Phase 2 (Next 2-4 weeks)

1. **Real data integration:**
   - Connect to Planetary Computer API
   - Download actual MODIS/Sentinel-2 data
   - Remove simulation code

2. **Additional indices:**
   - SAVI (Soil-Adjusted VI)
   - NDWI (Water Index)
   - Thermal stress detection

3. **Additional regions:**
   - Russia wheat
   - Australia wheat
   - Africa Sahel

### Phase 3 (1-2 months)

4. **Machine learning:**
   - Train on historical yield data
   - Predict crop yields
   - Optimize trading thresholds

5. **Real-time alerts:**
   - Discord notifications
   - Email alerts
   - Web dashboard

6. **Yield forecasting:**
   - Integrate with crop models
   - USDA report prediction
   - Supply chain impact

## 📚 References

- [MODIS Vegetation Indices](https://modis.gsfc.nasa.gov/data/dataprod/mod13.php)
- [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
- [Planetary Computer](https://planetarycomputer.microsoft.com/)
- [NDVI Basics](https://www.nasa.gov/mission_pages/sage3/ndvi.html)

## ⚠️ Limitations

1. **Latency:** 1-3 day delay from satellite
2. **Cloud cover:** Can obscure observations
3. **Spatial resolution:** Regional, not field-level
4. **Crop stage:** Doesn't distinguish growth stages
5. **Sample size:** Need more historical data

## 💡 Best Practices

1. **Focus on critical seasons**
   - Higher confidence during growing months
   - Lower confidence outside

2. **Combine with precipitation**
   - NDVI + precipitation = stronger signal
   - Cross-validate drought detection

3. **Start small**
   - Use 1-2% position sizes
   - Build confidence over time

4. **Track performance**
   - Log all signals
   - Calculate accuracy
   - Refine thresholds

5. **Understand crop cycles**
   - Planting → Growing → Harvest
   - Different NDVI at each stage

## 🆚 Comparison to Other Signals

| Signal Type | Lead Time | Accuracy | Best For |
|-------------|-----------|----------|----------|
| **Vegetation Health** | 2-4 weeks | TBD | Agriculture |
| Precipitation | 2-4 weeks | TBD | Agriculture |
| SST/ENSO | 2-4 weeks | 70-80% | Agriculture |
| Atmospheric | 1-2 weeks | TBD | Industrial |
| Thermal IR | 2-4 weeks | TBD | Production |

## 🌾 Agricultural Applications

This module is particularly valuable for:

1. **Crop Yield Prediction:**
   - Early warning of stress
   - Yield forecasting
   - Production estimates

2. **Commodity Trading:**
   - Time entries/exits
   - Regional arbitrage
   - Cross-commodity strategies

3. **Forestry Management:**
   - Forest health monitoring
   - Timber supply tracking
   - Deforestation detection

4. **Climate Monitoring:**
   - Drought detection
   - Climate change impacts
   - Carbon sequestration

---

**Note:** This module currently uses simulated data for demonstration. For production use, connect to the Planetary Computer API to fetch real MODIS and Sentinel-2 data.
