# Atmospheric Monitoring - User Guide

## 💨 Overview

The atmospheric monitoring module uses satellite data to track industrial activity by measuring atmospheric gas concentrations (NO2, SO2, CO2, CH4). Leading indicator for production, energy consumption, and carbon emissions.

**Data Source:** TROPOMI (Sentinel-5P) and OCO-2/3
**Update Frequency:** Daily
**Latency:** 1-5 days
**Cost:** Free (via Planetary Computer)

## 📊 How It Works

### Signal Logic

```
Gas Emissions ↑ → Production ↑ → LONG signal
Gas Emissions ↓ → Production ↓ → SHORT signal
Gas Emissions → → Normal production → NEUTRAL
```

### Detection Method

1. **Fetch atmospheric data** for target region
2. **Calculate baseline** (90-day average)
3. **Calculate z-scores** for each gas (NO2, SO2, CO2, CH4)
4. **Generate signal** based on combined anomaly detection

### Signal Thresholds

- **LONG:** Combined z-score > +2.0
- **SHORT:** Combined z-score < -2.0
- **NEUTRAL:** Combined z-score between -2.0 and +2.0

### Gas Weighting

Different gases indicate different industrial activities:

```
Combined Z-score = NO2 (40%) + SO2 (30%) + CO2 (20%) + CH4 (10%)
```

**Why this weighting?**
- **NO2 (40%):** Best indicator of combustion/industrial activity
- **SO2 (30%):** Coal burning and industrial processes
- **CO2 (20%):** General fossil fuel consumption
- **CH4 (10%):** Oil & gas production leaks

## 🗺️ Monitored Regions (9)

### China (2 regions)

| Region | Type | Key Gases | Trading Instruments |
|--------|------|-----------|---------------------|
| **Eastern China Industrial Belt** | Mixed industrial | NO2, SO2, CO2 | FXI, MCHI, ASHR |
| **Shanxi Coal & Steel** | Coal + Steel | SO2, NO2, CO | FXI, KOL, HWA |

### USA (3 regions)

| Region | Type | Key Gases | Trading Instruments |
|--------|------|-----------|---------------------|
| **Gulf Coast Petrochemical** | Petrochemical | NO2, SO2, CH4 | XLE, XOM, CVX, PSX, VLO |
| **Midwest Steel Belt** | Steel + Mfg | NO2, CO, SO2 | X, NUE, STLD, AKS |
| **Permian Basin** | Oil & Gas | CH4, NO2 | XLE, XOM, CVX, PXD, FANG |

### Europe (2 regions)

| Region | Type | Key Gases | Trading Instruments |
|--------|------|-----------|---------------------|
| **Rhine-Ruhr Industrial** | Mixed industrial | NO2, CO2, SO2 | EWG, FXD, EXI1 |
| **Poland Coal Region** | Coal + Power | SO2, NO2, CO2 | EPOL, TLW |

### Other (2 regions)

| Region | Type | Key Gases | Trading Instruments |
|--------|------|-----------|---------------------|
| **Western India Industrial** | Mixed industrial | NO2, SO2, CO2 | INDA, EPI, INP |
| **Middle East Oil Fields** | Oil & Gas | CH4, NO2, SO2 | USO, BNO, OIH |

## 🚀 Usage

### Generate signals for all regions

```bash
python3 -m pipeline.atmospheric
```

### Generate signal for specific region

```python
from pipeline.atmospheric import AtmosphericMonitor

monitor = AtmosphericMonitor()

# Generate signal for Eastern China
signal = monitor.generate_signal("china_industrial_east")

print(f"Direction: {signal['direction']}")
print(f"Confidence: {signal['confidence']}%")
print(f"NO2: {signal['current_no2']:.1f} μmol/m²")
print(f"SO2: {signal['current_so2']:.1f} μmol/m²")
print(f"CO2: {signal['current_co2']:.1f} ppm")
print(f"Activity: {signal['activity_level']}")
print(f"Instruments: {signal['instruments']}")
```

### Get regional summary

```python
summary = monitor.get_regional_summary()
print(f"Monitoring {summary['total_regions']} regions")
print(f"Gases: {', '.join(summary['gases_monitored'])}")
```

## 📈 Signal Output

### Single Region Signal

```json
{
  "region_id": "china_industrial_east",
  "region_name": "Eastern China Industrial Belt",
  "region_type": "industrial_mixed",
  "country": "China",
  "date": "2026-03-15",
  "signal_type": "atmospheric",
  "direction": "long",
  "confidence": 75,
  "rationale": "Industrial emissions +15.2% above baseline. Production activity significantly increased.",
  "instruments": ["FXI", "MCHI", "ASHR"],
  "current_no2": 22.5,
  "current_so2": 2.8,
  "current_co2": 420.5,
  "current_ch4": 1920,
  "activity_level": "high",
  "baseline_no2": 17.6,
  "baseline_so2": 2.4,
  "baseline_co2": 415.3,
  "baseline_ch4": 1848,
  "no2_z_score": 2.15,
  "so2_z_score": 1.82,
  "co2_z_score": 1.95,
  "combined_z_score": 2.05,
  "anomaly": "significant",
  "data_quality": "good",
  "timestamp": "2026-03-15T22:49:09.784238"
}
```

### Daily Summary

```json
{
  "date": "2026-03-15",
  "total_regions": 9,
  "signals_generated": 9,
  "long_signals": 2,
  "short_signals": 1,
  "neutral_signals": 6,
  "by_region_type": {
    "industrial_mixed": {"count": 3, "long": 1, "short": 0, "neutral": 2},
    "oil_gas": {"count": 2, "long": 1, "short": 1, "neutral": 0},
    ...
  },
  "signals": [...]
}
```

## 🎯 Trading Strategy

### Recommended Approach

1. **Combine with other signals:**
   - Atmospheric → Industrial production
   - Thermal IR → Facility activity
   - Nighttime lights → Economic activity

2. **Use as leading indicator:**
   - Gas changes before economic reports
   - 1-2 week lead time

3. **Region-specific signals:**
   - China NO2↑ → LONG FXI/MCHI
   - US Gulf emissions↑ → LONG XLE
   - Permian CH4↑ → LONG oil stocks

### Region-Specific Strategies

**China Industrial:**
```
NO2/SO2 ↑ → Manufacturing ↑ → LONG FXI, MCHI
NO2/SO2 ↓ → Manufacturing ↓ → SHORT FXI
```

**US Petrochemical:**
```
Emissions ↑ → Refining ↑ → LONG XOM, CVX, PSX
Emissions ↓ → Refining ↓ → SHORT XLE
```

**Oil & Gas Regions:**
```
CH4 ↑ → Production ↑ → LONG XLE, USO
CH4 ↓ → Production ↓ → SHORT oil
```

**Steel/Coal:**
```
SO2 ↑ → Steel production ↑ → LONG X, NUE
SO2 ↓ → Steel production ↓ → SHORT X
```

### Risk Management

```
✅ Diversify across regions
✅ Use small position sizes (1-2%)
✅ Combine with fundamental analysis
✅ Monitor data quality (cloud cover)
✅ Consider seasonal patterns
```

## 📊 Historical Performance

**Signal Accuracy:** TBD (need 20+ signals for statistical significance)

**Expected Lead Time:**
- Economic reports: 1-2 weeks
- Company earnings: 2-3 weeks
- Commodity prices: 1-2 weeks

## 🔧 Technical Details

### Data Processing Pipeline

```
1. Download TROPOMI/OCO-2 data (Planetary Computer)
2. Extract regional ROI (region of interest)
3. Calculate gas concentrations:
   - NO2: Nitrogen dioxide (industrial combustion)
   - SO2: Sulfur dioxide (coal burning)
   - CO2: Carbon dioxide (fossil fuel use)
   - CH4: Methane (oil & gas leaks)
4. Compare to 90-day baseline
5. Generate signal with confidence score
```

### Quality Control

- **Cloud cover:** Filter out cloudy observations
- **Wind patterns:** Account for gas transport
- **Seasonal adjustment:** Account for heating/cooling
- **Background levels:** Remove natural variations

### Gas Concentration Units

| Gas | Unit | Typical Range | High Activity |
|-----|------|---------------|---------------|
| NO2 | μmol/m² | 3-20 | >15 |
| SO2 | μmol/m² | 0.5-5 | >3 |
| CO2 | ppm | 405-425 | >415 |
| CH4 | ppb | 1750-1950 | >1900 |

## 📁 File Locations

```
outputs/atmospheric/
├── signal_china_industrial_east_2026-03-15.json
├── signal_usa_petrochemical_gulf_2026-03-15.json
├── ...
└── summary_2026-03-15.json
```

## 🔮 Future Enhancements

### Phase 2 (Next 2-4 weeks)

1. **Real data integration:**
   - Connect to Planetary Computer API
   - Download actual TROPOMI/OCO-2 data
   - Remove simulation code

2. **Additional regions:**
   - Southeast Asia (Vietnam, Thailand)
   - Brazil (industrial zones)
   - Australia (mining regions)

3. **Advanced analytics:**
   - Trend detection (7-day, 30-day)
   - Cross-region correlation
   - Commodity price prediction

### Phase 3 (1-2 months)

4. **Machine learning:**
   - Train on historical data
   - Predict GDP growth
   - Predict economic reports

5. **Real-time alerts:**
   - Discord notifications
   - Email alerts
   - Web dashboard

6. **Carbon trading:**
   - Predict carbon credit prices
   - ESG investment signals

## 📚 References

- [TROPOMI (Sentinel-5P)](https://sentinel.esa.int/web/sentinel/missions/sentinel-5p)
- [OCO-2/3](https://oco.jpl.nasa.gov/)
- [Planetary Computer](https://planetarycomputer.microsoft.com/)
- [Air Quality & Economic Activity](https://www.nasa.gov/feature/goddard/2020/nasa-satellite-data-show-air-pollution-decreases-over-southeast-asia)

## ⚠️ Limitations

1. **Latency:** 1-5 day delay from satellite
2. **Weather:** Cloud cover can obscure data
3. **Wind patterns:** Gas transport affects readings
4. **Seasonality:** Heating/cooling affects emissions
5. **Sample size:** Need more historical data for backtesting

## 💡 Best Practices

1. **Don't rely solely on atmospheric data**
   - Combine with other signals
   - Validate with fundamental analysis

2. **Monitor data quality**
   - Check for cloud cover
   - Verify wind patterns
   - Account for seasonality

3. **Start small**
   - Use 1-2% position sizes
   - Build confidence over time

4. **Track performance**
   - Log all signals
   - Calculate accuracy
   - Refine thresholds

5. **Understand regional characteristics**
   - Industrial mix
   - Seasonal patterns
   - Regulatory environment

## 🆚 Comparison to Other Signals

| Signal Type | Lead Time | Accuracy | Coverage |
|-------------|-----------|----------|----------|
| **Atmospheric** | 1-2 weeks | TBD | Regional |
| Thermal IR | 2-4 weeks | TBD | Facility |
| Nighttime Lights | 1-3 months | TBD | Regional |
| Chokepoint | 1-2 weeks | 72-100% | Global trade |

## 🌍 ESG & Carbon Trading

This module is particularly valuable for:

1. **ESG Investing:**
   - Track company/region carbon footprint
   - Identify cleaner producers
   - Support sustainable investing

2. **Carbon Credits:**
   - Predict emission levels
   - Forecast carbon credit demand
   - Trade carbon futures

3. **Regulatory Compliance:**
   - Monitor emission trends
   - Predict regulatory changes
   - Assess compliance risks

---

**Note:** This module currently uses simulated data for demonstration. For production use, connect to the Planetary Computer API to fetch real TROPOMI and OCO-2 data.
