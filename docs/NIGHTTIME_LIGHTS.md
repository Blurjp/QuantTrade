# Nighttime Lights Monitoring - User Guide

## 🌃 Overview

The nighttime lights monitoring module uses satellite imagery to detect economic activity changes by measuring nighttime light intensity across industrial regions.

**Data Source:** VIIRS (Visible Infrared Imaging Radiometer Suite) on Suomi-NPP satellite
**Update Frequency:** Daily
**Latency:** 1-3 days
**Cost:** Free (via Planetary Computer)

## 📊 How It Works

### Signal Logic

```
Light Intensity ↑ → Economic Activity ↑ → LONG signal
Light Intensity ↓ → Economic Activity ↓ → SHORT signal
Light Intensity → → Normal activity → NEUTRAL
```

### Detection Method

1. **Fetch VIIRS data** for target region
2. **Calculate baseline** (90-day average)
3. **Calculate z-score** (standard deviations from baseline)
4. **Generate signal** based on anomaly detection

### Signal Thresholds

- **LONG:** Z-score > +2.0 (intensity 2+ standard deviations above baseline)
- **SHORT:** Z-score < -2.0 (intensity 2+ standard deviations below baseline)
- **NEUTRAL:** Z-score between -2.0 and +2.0

## 🗺️ Monitored Regions

### China (3 regions)

| Region | Description | Trading Instruments |
|--------|-------------|---------------------|
| **Shanghai** | Yangtze River Delta industrial zone | FXI, MCHI, ASHR |
| **Guangdong** | Pearl River Delta manufacturing | FXI, MCHI, KWEB |
| **Beijing** | Northern China industrial region | FXI, MCHI |

### USA (3 regions)

| Region | Description | Trading Instruments |
|--------|-------------|---------------------|
| **Texas** | Permian Basin + Houston industrial | XLE, XOM, CVX, OIH |
| **California** | Silicon Valley + LA ports | QQQ, XLK, TECL |
| **Midwest** | Rust Belt manufacturing | XLI, CAT, DE |

### Europe (1 region)

| Region | Description | Trading Instruments |
|--------|-------------|---------------------|
| **Germany** | Rhine-Ruhr industrial region | EWG, FXD |

### India (1 region)

| Region | Description | Trading Instruments |
|--------|-------------|---------------------|
| **Mumbai** | Western India industrial zone | INDA, EPI |

## 🚀 Usage

### Generate signals for all regions

```bash
python3 -m pipeline.nighttime_lights
```

### Generate signal for specific region

```python
from pipeline.nighttime_lights import NighttimeLightsMonitor

monitor = NighttimeLightsMonitor()

# Generate signal for Shanghai
signal = monitor.generate_signal("china_shanghai")

print(f"Direction: {signal['direction']}")
print(f"Confidence: {signal['confidence']}%")
print(f"Z-score: {signal['z_score']}")
print(f"Instruments: {signal['instruments']}")
```

### Get regional summary

```python
summary = monitor.get_regional_summary()
print(f"Monitoring {summary['total_regions']} regions")
```

## 📈 Signal Output

### Single Region Signal

```json
{
  "region_id": "china_shanghai",
  "region_name": "Shanghai Industrial Zone",
  "country": "China",
  "date": "2026-03-15",
  "signal_type": "nighttime_lights",
  "direction": "long",
  "confidence": 75,
  "rationale": "Light intensity +5.2% above baseline. Strong economic activity increase detected.",
  "instruments": ["FXI", "MCHI", "ASHR"],
  "current_intensity": 97.89,
  "baseline_mean": 93.03,
  "baseline_std": 2.84,
  "z_score": 1.71,
  "anomaly": "moderate",
  "percentile_rank": 77.4,
  "deviation_pct": 5.23,
  "data_quality": "good",
  "timestamp": "2026-03-15T22:06:42.045747"
}
```

### Daily Summary

```json
{
  "date": "2026-03-15",
  "total_regions": 8,
  "signals_generated": 8,
  "long_signals": 2,
  "short_signals": 1,
  "neutral_signals": 5,
  "signals": [...]
}
```

## 🎯 Trading Strategy

### Recommended Approach

1. **Combine with other signals:**
   - Nighttime lights → Economic activity
   - Chokepoint monitoring → Supply chain
   - Agricultural → Commodity supply

2. **Use as leading indicator:**
   - Lights change before GDP reports
   - 1-3 month lead time

3. **Regional allocation:**
   - China lights up → LONG FXI/MCHI
   - US lights up → LONG SPY/QQQ
   - Germany lights up → LONG EWG

### Risk Management

```
✅ Diversify across regions
✅ Use small position sizes (1-2%)
✅ Combine with fundamental analysis
✅ Monitor data quality (cloud cover)
```

## 📊 Historical Performance

**Signal Accuracy:** TBD (need 20+ signals for statistical significance)

**Expected Lead Time:**
- Economic reports: 1-3 months
- Company earnings: 2-4 weeks
- Sector rotation: 1-2 weeks

## 🔧 Technical Details

### Data Processing Pipeline

```
1. Download VIIRS data (Planetary Computer)
2. Extract regional ROI (region of interest)
3. Calculate light intensity metrics:
   - Mean radiance
   - Coverage percentage
   - Brightness distribution
4. Compare to 90-day baseline
5. Generate signal with confidence score
```

### Quality Control

- **Cloud cover:** Filter out cloudy observations
- **Moonlight:** Correct for lunar illumination
- **Seasonal adjustment:** Account for winter/summer patterns
- **Holiday effects:** Exclude major holidays

## 📁 File Locations

```
outputs/nighttime_lights/
├── signal_china_shanghai_2026-03-15.json
├── signal_china_guangdong_2026-03-15.json
├── ...
└── summary_2026-03-15.json
```

## 🔮 Future Enhancements

### Phase 2 (Next 2-4 weeks)

1. **Real data integration:**
   - Connect to Planetary Computer API
   - Download actual VIIRS data
   - Remove simulation code

2. **Additional regions:**
   - Japan (Tokyo-Osaka corridor)
   - South Korea (Seoul industrial zone)
   - Brazil (São Paulo)

3. **Advanced analytics:**
   - Trend detection (7-day, 30-day)
   - Cross-regional correlation
   - Sector rotation signals

### Phase 3 (1-2 months)

4. **Machine learning:**
   - Train on historical data
   - Predict GDP growth
   - Predict earnings surprises

5. **Real-time alerts:**
   - Discord notifications
   - Email alerts
   - Web dashboard

## 📚 References

- [VIIRS Day/Night Band](https://earthobservatory.nasa.gov/features/NightLights)
- [Planetary Computer](https://planetarycomputer.microsoft.com/)
- [Economic Research using Nighttime Lights](https://www.nber.org/papers/w24526)

## ⚠️ Limitations

1. **Latency:** 1-3 day delay from satellite
2. **Weather:** Cloud cover can obscure data
3. **Seasonality:** Winter/summer patterns affect interpretation
4. **False signals:** Holidays, power outages, etc.
5. **Sample size:** Need more historical data for backtesting

## 💡 Best Practices

1. **Don't rely solely on nighttime lights**
   - Combine with other signals
   - Validate with fundamental analysis

2. **Monitor data quality**
   - Check for cloud cover
   - Verify seasonal adjustments

3. **Start small**
   - Use 1-2% position sizes
   - Build confidence over time

4. **Track performance**
   - Log all signals
   - Calculate accuracy
   - Refine thresholds

---

**Note:** This module currently uses simulated data for demonstration. For production use, connect to the Planetary Computer API to fetch real VIIRS data.
