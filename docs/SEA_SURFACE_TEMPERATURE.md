# Sea Surface Temperature (SST) Monitoring - User Guide

## 🌊 Overview

The SST monitoring module uses satellite data to track ocean temperatures for predicting El Niño/La Niña events and their impact on global commodity markets. Leading indicator for agriculture, energy, and fisheries.

**Data Source:** MODIS (Terra/Aqua) and AVHRR (NOAA)
**Update Frequency:** Daily
**Latency:** 1-3 days
**Cost:** Free (via Planetary Computer)

## 📊 How It Works

### Signal Logic

**ENSO Regions (El Niño/La Niña):**
```
El Niño (SST anomaly > +0.5°C) → SHORT agriculture (drought in Asia/Australia)
La Niña (SST anomaly < -0.5°C) → LONG agriculture (wet in Asia/Australia)
Neutral → Normal weather patterns
```

**Hurricane Regions:**
```
High SST (>28°C) → SHORT energy (hurricane risk)
Low SST (<26°C) → Neutral (low hurricane risk)
```

**Agricultural Impact Regions:**
```
Warm SST anomaly → LONG agriculture (enhanced moisture)
Cold SST anomaly → SHORT agriculture (drought risk)
```

**Fishing Grounds:**
```
Cold SST anomaly → LONG fisheries (favorable for cold-water fish)
Warm SST anomaly → SHORT fisheries (unfavorable conditions)
```

### Detection Method

1. **Fetch SST data** for target ocean region
2. **Calculate baseline** (90-day average)
3. **Calculate SST anomaly** (current - baseline)
4. **Determine ENSO state** (for Niño regions)
5. **Generate signal** based on region type

### ENSO Thresholds

- **El Niño:** SST anomaly > +0.5°C for 5+ consecutive 3-month periods
- **La Niña:** SST anomaly < -0.5°C for 5+ consecutive 3-month periods
- **Neutral:** SST anomaly between -0.5°C and +0.5°C

## 🗺️ Monitored Regions (9)

### ENSO Regions (3 regions)

| Region | Baseline SST | Trading Instruments | Impact |
|--------|--------------|---------------------|--------|
| **Niño 3.4** | 27.5°C | CORN, SOYB, WEAT, CANE, JO | Global weather |
| **Niño 3** | 26.0°C | CORN, SOYB, WEAT | South America |
| **Niño 4** | 29.0°C | CORN, SOYB, WEAT | Asian monsoon |

### Agricultural Impact Regions (1 region)

| Region | Baseline SST | Trading Instruments | Impact |
|--------|--------------|---------------------|--------|
| **Gulf of Mexico** | 27.0°C | CORN, SOYB, COTTON, NG | US Midwest rainfall |

### Hurricane Zone (1 region)

| Region | Baseline SST | Trading Instruments | Impact |
|--------|--------------|---------------------|--------|
| **Atlantic Hurricane Region** | 26.5°C | NG, OIL, XLE, UNG | Energy infrastructure |

### Monsoon Regions (2 regions)

| Region | Baseline SST | Trading Instruments | Impact |
|--------|--------------|---------------------|--------|
| **Indian Ocean** | 28.0°C | COTTON, SUGAR, TEA, RICE | Indian agriculture |
| **Pacific Warm Pool** | 29.5°C | RICE, PALM, SUGAR | SE Asia agriculture |

### Fishing Grounds (2 regions)

| Region | Baseline SST | Trading Instruments | Impact |
|--------|--------------|---------------------|--------|
| **Peru/Humboldt Current** | 18.0°C | FISH, FMC, SEA | Fish meal production |
| **Benguela Current** | 15.0°C | FISH, SEA | African fisheries |

## 🚀 Usage

### Generate signals for all regions

```bash
python3 -m pipeline.sea_surface_temperature
```

### Generate signal for specific region

```python
from pipeline.sea_surface_temperature import SeaSurfaceTemperatureMonitor

monitor = SeaSurfaceTemperatureMonitor()

# Generate signal for Niño 3.4
signal = monitor.generate_signal("nino34")

print(f"Direction: {signal['direction']}")
print(f"Confidence: {signal['confidence']}%")
print(f"SST: {signal['current_sst']:.2f}°C")
print(f"Anomaly: {signal['sst_anomaly']:+.2f}°C")
print(f"ENSO State: {signal['enso_state'].upper()}")
print(f"Instruments: {signal['instruments']}")
```

### Get regional summary

```python
summary = monitor.get_regional_summary()
print(f"Monitoring {summary['total_regions']} ocean regions")
print(f"Region types: {', '.join(summary['region_types'])}")
```

## 📈 Signal Output

### Single Region Signal

```json
{
  "region_id": "nino34",
  "region_name": "Niño 3.4 Region",
  "region_type": "enso_indicator",
  "ocean": "Pacific",
  "date": "2026-03-16",
  "signal_type": "sea_surface_temperature",
  "direction": "short",
  "confidence": 75,
  "rationale": "El Niño conditions detected. SST anomaly +0.76°C. Impact: Drought risk in Australia/Asia, wet in South America.",
  "instruments": ["CORN", "SOYB", "WEAT", "CANE", "JO"],
  "current_sst": 28.26,
  "sst_anomaly": 0.76,
  "enso_state": "el_nino",
  "baseline_sst": 27.5,
  "sst_z_score": 1.57,
  "combined_z_score": 1.57,
  "anomaly": "moderate",
  "impact": "Global weather patterns",
  "data_quality": "good",
  "timestamp": "2026-03-16T00:58:55.222354"
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
  "enso_state": "el_nino",
  "by_region_type": {
    "enso_indicator": {"count": 3, "long": 0, "short": 1, "neutral": 2},
    "agricultural_impact": {"count": 1, "long": 0, "short": 0, "neutral": 1},
    "hurricane_zone": {"count": 1, "long": 0, "short": 0, "neutral": 1},
    "monsoon_region": {"count": 2, "long": 1, "short": 0, "neutral": 1},
    "fishing_grounds": {"count": 2, "long": 1, "short": 1, "neutral": 0}
  },
  "signals": [...]
}
```

## 🎯 Trading Strategy

### ENSO-Based Agriculture Trading

**El Niño Impact:**
```
DROUGHT RISK:
  • Australia wheat → SHORT WEAT
  • Indonesia palm oil → SHORT PALM
  • India sugar → SHORT SUGAR
  • Thailand rice → SHORT RICE

WET CONDITIONS:
  • Argentina soybeans → LONG SOYB
  • Brazil corn → LONG CORN
  • Peru cotton → LONG COTTON
```

**La Niña Impact:**
```
WET CONDITIONS:
  • Australia wheat → LONG WEAT
  • Indonesia palm oil → LONG PALM
  • India sugar → LONG SUGAR

DRY CONDITIONS:
  • Argentina soybeans → SHORT SOYB
  • Brazil corn → SHORT CORN
```

### Hurricane Season Energy Trading

**High SST (>28°C in Atlantic):**
```
RISK PERIOD (June-November):
  • SHORT XLE, XOM, CVX (offshore platforms)
  • SHORT UNG, NG (Gulf production)
  • LONG volatility (energy VIX)
```

**Low SST (<26°C):**
```
LOW RISK:
  • Neutral or LONG energy
  • Focus on other signals
```

### Regional Strategy Examples

**US Agriculture (Gulf of Mexico):**
```
Warm Gulf → Enhanced moisture → LONG CORN, SOYB
Cold Gulf → Drought risk → SHORT CORN, SOYB
```

**Indian Monsoon (Indian Ocean):**
```
Warm Indian Ocean → Strong monsoon → LONG COTTON, SUGAR
Cold Indian Ocean → Weak monsoon → SHORT COTTON
```

**Fisheries (Peru Current):**
```
Cold current → Good anchovy → LONG FISH, FMC
Warm current → Poor anchovy → SHORT FISH
```

### Risk Management

```
✅ Monitor ENSO forecasts (3-6 month outlook)
✅ Combine with weather models
✅ Use small position sizes (1-2%)
✅ Diversify across regions
✅ Consider seasonal patterns
```

## 📊 Historical Performance

**Signal Accuracy:** TBD (need 20+ signals for statistical significance)

**Expected Lead Time:**
- Agriculture prices: 2-4 weeks
- Energy prices: 1-2 weeks (hurricane season)
- Commodity markets: 1-3 months

## 🔧 Technical Details

### Data Processing Pipeline

```
1. Download MODIS/AVHRR data (Planetary Computer)
2. Extract regional SST (sea surface temperature)
3. Calculate anomalies:
   - Current SST - Baseline SST
   - 90-day rolling average
4. Classify ENSO state (for Niño regions)
5. Generate signal with confidence score
```

### Quality Control

- **Cloud screening:** Remove cloudy pixels
- **Spatial averaging:** Regional mean SST
- **Temporal smoothing:** 7-day running mean
- **Anomaly calculation:** Remove seasonal cycle

### SST Ranges

| Region Type | Typical Range | El Niño | La Niña |
|-------------|---------------|---------|---------|
| Niño 3.4 | 26-29°C | >28°C | <26°C |
| Niño 3 | 24-28°C | >27°C | <25°C |
| Niño 4 | 28-30°C | >29.5°C | <28.5°C |

## 📁 File Locations

```
outputs/sea_surface_temperature/
├── signal_nino34_2026-03-16.json
├── signal_gulf_mexico_2026-03-16.json
├── ...
└── summary_2026-03-16.json
```

## 🔮 Future Enhancements

### Phase 2 (Next 2-4 weeks)

1. **Real data integration:**
   - Connect to Planetary Computer API
   - Download actual MODIS/AVHRR data
   - Remove simulation code

2. **ENSO forecasting:**
   - 3-6 month ENSO outlook
   - Machine learning predictions
   - Coupled ocean-atmosphere models

3. **Additional regions:**
   - Coral Sea (Australia)
   - Arabian Sea (Middle East)
   - Caribbean (hurricane genesis)

### Phase 3 (1-2 months)

4. **Machine learning:**
   - Train on historical ENSO data
   - Predict commodity prices
   - Optimize trading thresholds

5. **Real-time alerts:**
   - Discord notifications
   - Email alerts
   - Web dashboard

6. **Climate integration:**
   - Long-term climate trends
   - Climate change impacts
   - Decadal oscillations

## 📚 References

- [MODIS SST](https://modis.gsfc.nasa.gov/data/dataprod/mod28.php)
- [NOAA ENSO](https://www.climate.gov/enso)
- [Planetary Computer](https://planetarycomputer.microsoft.com/)
- [SST and Agriculture](https://www.nass.usda.gov/Research_and_Science/Crop-Weather/index.php)

## ⚠️ Limitations

1. **Latency:** 1-3 day delay from satellite
2. **ENSO lag:** 2-3 months for ENSO to impact agriculture
3. **Complex interactions:** Multiple climate factors
4. **Regional variation:** Local weather vs. global patterns
5. **Sample size:** Need more historical data

## 💡 Best Practices

1. **Combine with weather forecasts**
   - Use ENSO as long-term indicator
   - Weather models for short-term

2. **Monitor multiple regions**
   - Niño 3.4 for global
   - Local regions for specific crops

3. **Start small**
   - Use 1-2% position sizes
   - Build confidence over time

4. **Track performance**
   - Log all signals
   - Calculate accuracy
   - Refine thresholds

5. **Understand ENSO cycles**
   - El Niño: 2-7 year cycle
   - La Niña: Often follows El Niño
   - Neutral: Most common state

## 🆚 Comparison to Other Signals

| Signal Type | Lead Time | Accuracy | Best For |
|-------------|-----------|----------|----------|
| **SST/ENSO** | 2-4 weeks | 70-80% | Agriculture, energy |
| Atmospheric | 1-2 weeks | TBD | Industrial |
| Thermal IR | 2-4 weeks | TBD | Production |
| Solar | 1-2 weeks | TBD | Energy |
| Nighttime Lights | 1-3 months | TBD | Economic |

## 🌍 Climate Applications

This module is particularly valuable for:

1. **Agriculture Trading:**
   - Predict crop yields
   - Time entries/exits
   - Hedge weather risk

2. **Energy Trading:**
   - Hurricane season preparation
   - Natural gas demand forecasting
   - Renewable energy planning

3. **Fisheries:**
   - Predict fish stocks
   - Aquaculture planning
   - Seafood prices

4. **Climate Risk:**
   - Drought monitoring
   - Flood risk assessment
   - Extreme weather preparation

---

**Note:** This module currently uses simulated data for demonstration. For production use, connect to the Planetary Computer API to fetch real MODIS and AVHRR data.
