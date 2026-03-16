# Thermal Infrared Monitoring - User Guide

## 🔥 Overview

The thermal infrared monitoring module uses satellite imagery to detect industrial production activity by measuring surface temperature changes at key facilities.

**Data Source:** Landsat 8/9 TIRS and Sentinel-3 SLSTR
**Update Frequency:** Daily to weekly
**Latency:** 1-3 days
**Cost:** Free (via Planetary Computer)

## 📊 How It Works

### Signal Logic

```
Temperature ↑ → Production ↑ → LONG signal
Temperature ↓ → Production ↓ → SHORT signal
Temperature → → Normal production → NEUTRAL
```

### Detection Method

1. **Fetch thermal IR data** for target facility
2. **Calculate baseline** (90-day average)
3. **Calculate z-scores** (temperature and hotspot coverage)
4. **Generate signal** based on combined anomaly detection

### Signal Thresholds

- **LONG:** Combined z-score > +2.0
- **SHORT:** Combined z-score < -2.0
- **NEUTRAL:** Combined z-score between -2.0 and +2.0

## 🏭 Monitored Facilities

### Power Generation (2 facilities)

| Facility | Location | Trading Instruments |
|----------|----------|---------------------|
| **Texas Power Complex** | Gulf Coast | XLU, VST, PEG, NEE |
| **Ohio River Plants** | Ohio Valley | XLU, AEP, D, DUK |

### Data Centers (2 facilities)

| Facility | Location | Trading Instruments |
|----------|----------|---------------------|
| **Virginia Data Hub** | Loudoun County | AMZN, GOOGL, MSFT, META |
| **Oregon Cluster** | The Dalles | GOOGL, AMZN, FB |

### Steel Production (2 facilities)

| Facility | Location | Trading Instruments |
|----------|----------|---------------------|
| **Pittsburgh Steel** | Mon Valley | X, NUE, STLD, AKS |
| **Birmingham Steel** | Birmingham, AL | X, NUE, STLD |

### Oil Refineries (2 facilities)

| Facility | Location | Trading Instruments |
|----------|----------|---------------------|
| **Houston Refineries** | Ship Channel | XOM, CVX, PSX, VLO |
| **Louisiana Refineries** | Gulf Coast | XOM, CVX, MPC, XLE |

### Manufacturing (2 facilities)

| Facility | Location | Trading Instruments |
|----------|----------|---------------------|
| **Detroit Auto** | Detroit Metro | F, GM, STLA |
| **Arizona Semiconductor** | Phoenix | INTC, TSM, AMD, NVDA |

## 🚀 Usage

### Generate signals for all facilities

```bash
python3 -m pipeline.thermal_infrared
```

### Generate signal for specific facility

```python
from pipeline.thermal_infrared import ThermalInfraredMonitor

monitor = ThermalInfraredMonitor()

# Generate signal for Texas power plant
signal = monitor.generate_signal("power_plant_texas")

print(f"Direction: {signal['direction']}")
print(f"Confidence: {signal['confidence']}%")
print(f"Temperature: {signal['current_temp']:.1f}°C")
print(f"Activity: {signal['activity_level']}")
print(f"Instruments: {signal['instruments']}")
```

### Get facility summary

```python
summary = monitor.get_facility_summary()
print(f"Monitoring {summary['total_facilities']} facilities")
```

## 📈 Signal Output

### Single Facility Signal

```json
{
  "facility_id": "power_plant_texas",
  "facility_name": "Texas Power Generation Complex",
  "facility_type": "power_generation",
  "location": "Texas Gulf Coast",
  "date": "2026-03-15",
  "signal_type": "thermal_infrared",
  "direction": "long",
  "confidence": 75,
  "rationale": "Temperature +15.2% above baseline. Production activity significantly increased.",
  "instruments": ["XLU", "VST", "PEG", "NEE"],
  "current_temp": 52.3,
  "current_coverage": 68.5,
  "activity_level": "high",
  "baseline_temp_mean": 45.4,
  "baseline_temp_std": 3.8,
  "temp_z_score": 1.82,
  "coverage_z_score": 2.15,
  "combined_z_score": 1.99,
  "anomaly": "moderate",
  "data_quality": "good",
  "timestamp": "2026-03-15T22:12:36.338048"
}
```

### Daily Summary

```json
{
  "date": "2026-03-15",
  "total_facilities": 10,
  "signals_generated": 10,
  "long_signals": 2,
  "short_signals": 1,
  "neutral_signals": 7,
  "by_facility_type": {
    "power_generation": {"count": 2, "long": 1, "short": 0, "neutral": 1},
    "datacenter": {"count": 2, "long": 0, "short": 1, "neutral": 1},
    ...
  },
  "signals": [...]
}
```

## 🎯 Trading Strategy

### Recommended Approach

1. **Combine with other signals:**
   - Thermal IR → Production levels
   - Nighttime lights → Economic activity
   - Chokepoint → Supply chain

2. **Use as leading indicator:**
   - Temperature changes before earnings
   - 2-4 week lead time

3. **Sector-specific signals:**
   - Power plants hot → LONG XLU
   - Steel mills hot → LONG X, NUE
   - Data centers hot → LONG AMZN, GOOGL

### Facility-Specific Strategies

**Power Generation:**
```
Temp ↑ → Power demand ↑ → LONG XLU, VST
Temp ↓ → Power demand ↓ → SHORT XLU
```

**Data Centers:**
```
Temp ↑ → Computing demand ↑ → LONG AMZN, GOOGL, MSFT
Temp ↓ → Computing demand ↓ → SHORT tech
```

**Steel Production:**
```
Temp ↑ → Steel production ↑ → LONG X, NUE
Temp ↓ → Steel production ↓ → SHORT X
```

**Oil Refineries:**
```
Temp ↑ → Refining activity ↑ → LONG XOM, CVX
Temp ↓ → Refining activity ↓ → SHORT XLE
```

### Risk Management

```
✅ Diversify across facility types
✅ Use small position sizes (1-2%)
✅ Combine with fundamental analysis
✅ Monitor data quality (cloud cover)
```

## 📊 Historical Performance

**Signal Accuracy:** TBD (need 20+ signals for statistical significance)

**Expected Lead Time:**
- Company earnings: 2-4 weeks
- Sector reports: 1-3 weeks
- Economic data: 1-2 weeks

## 🔧 Technical Details

### Data Processing Pipeline

```
1. Download thermal IR data (Planetary Computer)
2. Extract facility ROI (region of interest)
3. Calculate temperature metrics:
   - Mean temperature
   - Hotspot coverage
   - Temperature distribution
4. Compare to 90-day baseline
5. Generate signal with confidence score
```

### Quality Control

- **Cloud cover:** Filter out cloudy observations
- **Seasonal adjustment:** Account for ambient temperature
- **Time of day:** Correct for solar heating effects
- **Sensor calibration:** Validate temperature readings

### Temperature Ranges by Facility Type

| Facility Type | Normal Range (°C) | Activity Threshold |
|---------------|-------------------|-------------------|
| Power Generation | 30-55 | 38-40 |
| Data Centers | 20-45 | 28-32 |
| Steel Mills | 40-80 | 52-55 |
| Oil Refineries | 32-70 | 45-48 |
| Auto Manufacturing | 20-45 | 30 |
| Semiconductor | 25-50 | 35 |

## 📁 File Locations

```
outputs/thermal_infrared/
├── signal_power_plant_texas_2026-03-15.json
├── signal_datacenter_virginia_2026-03-15.json
├── ...
└── summary_2026-03-15.json
```

## 🔮 Future Enhancements

### Phase 2 (Next 2-4 weeks)

1. **Real data integration:**
   - Connect to Planetary Computer API
   - Download actual Landsat/Sentinel data
   - Remove simulation code

2. **Additional facilities:**
   - Chemical plants (Dow, DuPont)
   - Mining operations (BHP, Rio Tinto)
   - Logistics hubs (FedEx, UPS)

3. **Advanced analytics:**
   - Trend detection (7-day, 30-day)
   - Cross-facility correlation
   - Supply chain signals

### Phase 3 (1-2 months)

4. **Machine learning:**
   - Train on historical data
   - Predict earnings surprises
   - Predict economic reports

5. **Real-time alerts:**
   - Discord notifications
   - Email alerts
   - Web dashboard

## 📚 References

- [Landsat 8/9 TIRS](https://www.usgs.gov/landsat-missions/landsat-thermal-infrared-sensor)
- [Sentinel-3 SLSTR](https://www.esa.int/ESA_Missions/Sentinel-3)
- [Planetary Computer](https://planetarycomputer.microsoft.com/)
- [Industrial Heat Detection](https://earthobservatory.nasa.gov/features/MeasuringHeat)

## ⚠️ Limitations

1. **Latency:** 1-3 day delay from satellite
2. **Weather:** Cloud cover can obscure data
3. **Seasonality:** Ambient temperature affects readings
4. **False signals:** Maintenance, shutdowns, etc.
5. **Sample size:** Need more historical data for backtesting

## 💡 Best Practices

1. **Don't rely solely on thermal IR**
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

5. **Understand facility operations**
   - Maintenance schedules
   - Planned shutdowns
   - Seasonal patterns

## 🆚 Comparison to Other Signals

| Signal Type | Lead Time | Accuracy | Coverage |
|-------------|-----------|----------|----------|
| **Thermal IR** | 2-4 weeks | TBD | Facility-specific |
| Nighttime Lights | 1-3 months | TBD | Regional |
| Chokepoint | 1-2 weeks | 72-100% | Global trade |
| Agricultural | 1-3 months | 66-100% | Regional |

---

**Note:** This module currently uses simulated data for demonstration. For production use, connect to the Planetary Computer API to fetch real Landsat and Sentinel data.
