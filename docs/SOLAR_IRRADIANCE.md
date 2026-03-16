# Solar Irradiance Monitoring - User Guide

## ☀️ Overview

The solar irradiance monitoring module uses satellite data to track solar power generation potential and predict energy market dynamics. Leading indicator for solar stocks, natural gas demand, and electricity prices.

**Data Source:** MODIS (Terra/Aqua) and Sentinel-3 SLSTR
**Update Frequency:** Daily
**Latency:** 1-3 days
**Cost:** Free (via Planetary Computer)

## 📊 How It Works

### Signal Logic

**Solar Farm Regions:**
```
Irradiance ↑ → Solar Generation ↑ → LONG solar stocks
Irradiance ↓ → Solar Generation ↓ → SHORT solar stocks
```

**Cloudy Regions (Gas Demand Proxy):**
```
Cloud Cover ↑ → Solar ↓ → Gas Power ↑ → LONG natural gas
Cloud Cover ↓ → Solar ↑ → Gas Power ↓ → SHORT natural gas
```

### Detection Method

1. **Fetch irradiance data** for target region
2. **Calculate baseline** (90-day average)
3. **Calculate z-scores** (irradiance, cloud cover, generation)
4. **Generate signal** based on region type

### Signal Thresholds

- **LONG:** Combined z-score > +2.0
- **SHORT:** Combined z-score < -2.0
- **NEUTRAL:** Combined z-score between -2.0 and +2.0

## 🗺️ Monitored Regions (9)

### USA Solar Markets (3 regions)

| Region | Capacity | Irradiance | Trading Instruments |
|--------|----------|------------|---------------------|
| **California Solar Belt** | 35 GW | 5.5 kWh/m²/day | TAN, FSLR, SPWR, SEDG, ENPH |
| **Texas Solar Corridor** | 20 GW | 5.2 kWh/m²/day | TAN, FSLR, NOVA, XLU |
| **Arizona Desert Solar** | 8 GW | 6.2 kWh/m²/day | TAN, FSLR, SPWR |

### Europe Solar (2 regions)

| Region | Capacity | Irradiance | Trading Instruments |
|--------|----------|------------|---------------------|
| **Spain Solar Hub** | 18 GW | 5.0 kWh/m²/day | TAN, ICLN, PBW |
| **Germany Solar Region** | 60 GW | 3.2 kWh/m²/day | TAN, ICLN, QCLN |

### Asia Solar (2 regions)

| Region | Capacity | Irradiance | Trading Instruments |
|--------|----------|------------|---------------------|
| **Western China Solar Base** | 150 GW | 5.8 kWh/m²/day | FXI, MCHI, TAN |
| **Rajasthan Solar Park** | 25 GW | 5.9 kWh/m²/day | INDA, TAN, ICLN |

### Gas Demand Proxies (2 regions)

| Region | Type | Trading Instruments |
|--------|------|---------------------|
| **US Northeast Cloud Cover** | Gas demand proxy | UNG, XLU, D |
| **UK Cloud Cover Region** | Gas demand proxy | UNG, XLU |

## 🚀 Usage

### Generate signals for all regions

```bash
python3 -m pipeline.solar_irradiance
```

### Generate signal for specific region

```python
from pipeline.solar_irradiance import SolarIrradianceMonitor

monitor = SolarIrradianceMonitor()

# Generate signal for California
signal = monitor.generate_signal("usa_california_solar")

print(f"Direction: {signal['direction']}")
print(f"Confidence: {signal['confidence']}%")
print(f"Irradiance: {signal['current_irradiance']:.2f} kWh/m²/day")
print(f"Generation: {signal['current_generation']:.1f} GWh")
print(f"Capacity Factor: {signal['capacity_factor']:.1%}")
print(f"Instruments: {signal['instruments']}")
```

### Get regional summary

```python
summary = monitor.get_regional_summary()
print(f"Monitoring {summary['total_regions']} regions")
print(f"Total capacity: {summary['total_installed_capacity_gw']:.0f} GW")
```

## 📈 Signal Output

### Single Region Signal

```json
{
  "region_id": "usa_california_solar",
  "region_name": "California Solar Belt",
  "region_type": "solar_farm_cluster",
  "country": "USA",
  "date": "2026-03-15",
  "signal_type": "solar_irradiance",
  "direction": "long",
  "confidence": 75,
  "rationale": "Solar irradiance +15.2% above baseline. Solar generation significantly increased.",
  "instruments": ["TAN", "FSLR", "SPWR", "SEDG", "ENPH"],
  "current_irradiance": 6.3,
  "current_cloud_cover": 15.2,
  "current_generation": 425.8,
  "capacity_factor": 0.72,
  "baseline_irradiance": 5.5,
  "baseline_cloud_cover": 30.0,
  "baseline_generation": 380.5,
  "irradiance_z_score": 2.15,
  "cloud_z_score": -1.82,
  "combined_z_score": 2.05,
  "anomaly": "significant",
  "grid_region": "CAISO",
  "data_quality": "good",
  "timestamp": "2026-03-15T22:54:15.123456"
}
```

### Daily Summary

```json
{
  "date": "2026-03-15",
  "total_regions": 9,
  "signals_generated": 9,
  "long_signals": 3,
  "short_signals": 1,
  "neutral_signals": 5,
  "total_generation_gwh": 2150.5,
  "by_region_type": {
    "solar_farm_cluster": {"count": 7, "long": 2, "short": 1, "neutral": 4},
    "gas_demand_proxy": {"count": 2, "long": 1, "short": 0, "neutral": 1}
  },
  "signals": [...]
}
```

## 🎯 Trading Strategy

### Recommended Approach

1. **Dual strategy:**
   - Solar regions → Trade solar stocks
   - Cloudy regions → Trade natural gas

2. **Use as leading indicator:**
   - Irradiance changes before earnings
   - 1-2 week lead time

3. **Cross-commodity:**
   - High solar → Low gas demand
   - Low solar → High gas demand

### Region-Specific Strategies

**Solar Farm Regions:**
```
Irradiance ↑ → Solar earnings ↑ → LONG TAN, FSLR, ENPH
Irradiance ↓ → Solar earnings ↓ → SHORT TAN

California (CAISO):
  Sunny → LONG solar stocks
  Cloudy → SHORT solar stocks

Texas (ERCOT):
  Sunny → LONG TAN, XLU (low gas)
  Cloudy → SHORT TAN, LONG UNG
```

**Gas Demand Proxies:**
```
Cloud Cover ↑ → Gas demand ↑ → LONG UNG
Cloud Cover ↓ → Gas demand ↓ → SHORT UNG

Northeast:
  Very cloudy → LONG UNG, XLU
  Very sunny → SHORT UNG
```

### Cross-Commodity Strategy

**Solar vs Natural Gas:**
```
High irradiance:
  • LONG TAN (solar ETF)
  • SHORT UNG (natural gas)
  
Low irradiance:
  • SHORT TAN
  • LONG UNG
```

### Risk Management

```
✅ Hedge solar with gas
✅ Use small position sizes (1-2%)
✅ Monitor weather forecasts
✅ Consider seasonal patterns
```

## 📊 Historical Performance

**Signal Accuracy:** TBD (need 20+ signals for statistical significance)

**Expected Lead Time:**
- Solar earnings: 1-2 weeks
- Gas prices: 1-3 days
- Electricity prices: 1-2 days

## 🔧 Technical Details

### Data Processing Pipeline

```
1. Download MODIS/Sentinel-3 data (Planetary Computer)
2. Extract regional ROI (region of interest)
3. Calculate irradiance metrics:
   - Surface solar irradiance (kWh/m²/day)
   - Cloud cover percentage
   - Clear sky vs actual irradiance
4. Estimate power generation
5. Compare to 90-day baseline
6. Generate signal with confidence score
```

### Quality Control

- **Cloud detection:** Validate cloud cover estimates
- **Aerosol correction:** Account for dust, smoke
- **Seasonal adjustment:** Solar declination angle
- **Time of day:** Solar noon measurements

### Irradiance Ranges

| Region Type | Typical Range | High Generation | Low Generation |
|-------------|---------------|-----------------|----------------|
| Desert | 5.5-7.0 kWh/m²/day | >6.5 | <4.5 |
| Temperate | 3.0-5.5 kWh/m²/day | >5.0 | <2.5 |
| Cloudy | 2.0-4.0 kWh/m²/day | >3.5 | <1.5 |

## 📁 File Locations

```
outputs/solar_irradiance/
├── signal_usa_california_solar_2026-03-15.json
├── signal_usa_texas_solar_2026-03-15.json
├── ...
└── summary_2026-03-15.json
```

## 🔮 Future Enhancements

### Phase 2 (Next 2-4 weeks)

1. **Real data integration:**
   - Connect to Planetary Computer API
   - Download actual MODIS/Sentinel-3 data
   - Remove simulation code

2. **Additional regions:**
   - Australia solar
   - Middle East solar
   - Chile solar

3. **Advanced analytics:**
   - 7-day irradiance forecast
   - Weather pattern prediction
   - Grid demand modeling

### Phase 3 (1-2 months)

4. **Machine learning:**
   - Train on historical data
   - Predict solar earnings
   - Predict gas prices

5. **Real-time alerts:**
   - Discord notifications
   - Email alerts
   - Web dashboard

6. **Grid integration:**
   - CAISO real-time data
   - ERCOT demand forecasts
   - European grid data

## 📚 References

- [MODIS (Terra/Aqua)](https://modis.gsfc.nasa.gov/)
- [Sentinel-3 SLSTR](https://www.esa.int/ESA_Missions/Sentinel-3)
- [Planetary Computer](https://planetarycomputer.microsoft.com/)
- [Solar Energy Prediction](https://www.nrel.gov/grid/solar-resource.html)

## ⚠️ Limitations

1. **Latency:** 1-3 day delay from satellite
2. **Weather:** Rapid weather changes
3. **Seasonality:** Winter/summer patterns
4. **Storage:** Battery storage dampens impact
5. **Sample size:** Need more historical data

## 💡 Best Practices

1. **Combine with weather forecasts**
   - Use 7-day weather predictions
   - Validate with ground truth

2. **Hedge positions**
   - Long solar + short gas
   - Diversify across regions

3. **Start small**
   - Use 1-2% position sizes
   - Build confidence over time

4. **Track performance**
   - Log all signals
   - Calculate accuracy
   - Refine thresholds

5. **Consider seasonality**
   - Summer: high solar
   - Winter: high gas

## 🆚 Comparison to Other Signals

| Signal Type | Lead Time | Accuracy | Best For |
|-------------|-----------|----------|----------|
| **Solar Irradiance** | 1-2 weeks | TBD | Energy trading |
| Atmospheric | 1-2 weeks | TBD | Industrial |
| Thermal IR | 2-4 weeks | TBD | Production |
| Nighttime Lights | 1-3 months | TBD | Economic |

## ⚡ Energy Market Applications

This module is particularly valuable for:

1. **Solar Stock Trading:**
   - Predict earnings surprises
   - Time entries/exits
   - Hedge positions

2. **Natural Gas Trading:**
   - Predict demand spikes
   - Cross-commodity arbitrage
   - Seasonal positioning

3. **Electricity Trading:**
   - Predict generation levels
   - Grid demand forecasting
   - Price forecasting

4. **Utilities:**
   - XLU positioning
   - Power company earnings
   - Grid stress signals

---

**Note:** This module currently uses simulated data for demonstration. For production use, connect to the Planetary Computer API to fetch real MODIS and Sentinel-3 data.
