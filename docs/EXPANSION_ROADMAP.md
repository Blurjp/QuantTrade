# QuantTrade Multi-Asset Satellite Trading System

## Overview

QuantTrade uses satellite imagery to detect real-world economic activity and generate trading signals across multiple asset classes.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SATELLITE DATA SOURCES                    │
│  Sentinel-1 (SAR)  │  Sentinel-2 (Optical)  │  Maxar/Planet  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    DETECTION PIPELINE                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Ships        │  │ Vehicles     │  │ Crops        │       │
│  │ (CFAR SAR)   │  │ (YOLO/OD)    │  │ (NDVI)       │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    SIGNAL GENERATION                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Throughput   │  │ Traffic      │  │ Yield        │       │
│  │ Anomaly      │  │ vs Baseline  │  │ Estimate     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    TRADING DECISIONS                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Multi-Asset Portfolio Manager                        │   │
│  │  - Position sizing                                    │   │
│  │  - Risk management (stop loss / take profit)          │   │
│  │  - Sector exposure limits                             │   │
│  │  - Signal-to-action mapping                           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Supported Monitoring Types

| Type | Data Source | Detection Target | Instruments | Update Freq |
|------|-------------|------------------|-------------|-------------|
| **Chokepoints** | Sentinel-1 SAR | Ships | WTI, Brent, XLE | Daily |
| **Retail Parking** | Sentinel-2/Maxar | Vehicles | WMT, COST, XRT | Weekly |
| **Auto Inventory** | Sentinel-2 | Parked cars | F, GM, CARZ | Weekly |
| **Agriculture** | Sentinel-2/Landsat | NDVI, crop color | Corn, Soy, Wheat | Weekly |
| **Port Logistics** | Sentinel-1 | Containers, ships | FDX, UPS | Daily |
| **Oil Storage** | Sentinel-1 | Tank roof height | WTI, USO | Weekly |
| **Coal Stockpiles** | Sentinel-2 | Pile size | Coal futures | Weekly |

## Current Active Monitors

### 1. Strait of Hormuz (✅ ACTIVE)
- **Type:** Chokepoint
- **Detection:** Sentinel-1 SAR → CFAR ship detection
- **Trading:** WTI/Brent futures
- **Signal:** Throughput anomaly vs baseline
- **Status:** Running daily at 6:00 AM

### Planned Monitors

| Priority | Target | Type | Tickers | Status |
|----------|--------|------|---------|--------|
| 2 | Walmart parking | Retail | WMT | 🔲 Config needed |
| 2 | Costco parking | Retail | COST | 🔲 Config needed |
| 2 | Cushing storage | Oil | WTI | 🔲 Pipeline needed |
| 3 | Iowa corn | Agri | Corn | 🔲 Pipeline needed |
| 3 | Detroit auto lots | Auto | F, GM | 🔲 Pipeline needed |
| 3 | LA/Long Beach | Port | XLI | 🔲 Pipeline needed |

## Paper Trading System

### Current Portfolio (as of 2026-03-09)

| Position | Direction | Entry | Size | Target | Stop |
|----------|-----------|-------|------|--------|------|
| WTI | SHORT | $120 | $5,000 | $102 | $125 |

### Risk Parameters

- **Initial Capital:** $100,000
- **Max Positions:** 10
- **Max Position Size:** 10% of capital
- **Max Sector Exposure:** 25% of capital
- **Default Stop Loss:** 4-5%
- **Default Take Profit:** 15-20%

### Daily Report

Automated daily report generated at 6:00 AM EST including:
- Portfolio P&L
- Position status
- Active signals
- Sector exposure
- Trading opportunities

## Extending the System

### Adding a New Monitoring Target

1. **Define AOI (Area of Interest)**
   ```bash
   # Create GeoJSON file
   vi configs/aoi_new_target.geojson
   ```

2. **Add to trading_targets.json**
   ```json
   {
     "new_target": {
       "name": "Target Name",
       "type": "retail_parking",
       "location": {"lat": 0.0, "lon": 0.0},
       "instruments": [{"ticker": "XYZ", "type": "equity"}],
       "active": true
     }
   }
   ```

3. **Create detection pipeline** (if new type)
   ```python
   # pipeline/detection_retail.py
   def detect_vehicles(scene_path: str) -> list:
       # Use YOLO or similar for vehicle detection
       pass
   ```

4. **Add signal logic**
   ```python
   # pipeline/signals.py
   def generate_retail_signal(traffic_data: dict) -> dict:
       # Compare traffic vs baseline
       pass
   ```

5. **Update automation**
   ```bash
   vi scripts/daily_run.sh
   # Add new region to daily run
   ```

### Adding a New Asset Class

1. **Update multi_asset_portfolio.py**
   ```python
   self.asset_classes["crypto"] = {
       "description": "Cryptocurrency",
       "examples": ["BTC", "ETH"],
       "default_stop_loss": 0.10,
       "default_take_profit": 0.30,
   }
   ```

2. **Map to sector**
   ```python
   self.sector_map["BTC"] = "crypto"
   ```

3. **Add price feed** (if needed)
   ```python
   def get_crypto_price(ticker: str) -> float:
       # Fetch from exchange API
       pass
   ```

## File Structure

```
QuantTrade/
├── configs/
│   ├── regions/           # Region registry
│   ├── aoi_*.geojson      # Areas of interest
│   ├── gate_*.geojson     # Gate polygons
│   ├── monitoring_types.json    # Detection types
│   └── trading_targets.json     # Trading targets
├── pipeline/
│   ├── run.py             # Main pipeline
│   ├── detection.py       # CFAR detection
│   ├── signals.py         # Signal generation
│   └── loader.py          # STAC data loader
├── paper_trading/
│   ├── multi_asset_portfolio.py  # Portfolio manager
│   ├── daily_multi_report.py     # Report generator
│   └── account_state.json        # Saved state
├── automation/
│   ├── daily.py           # Daily automation
│   └── alerts.py          # Alert system
├── outputs/
│   ├── YYYY-MM-DD/        # Daily outputs
│   ├── global_tracklets/  # Multi-day tracking
│   └── paper_trading/     # Trading state
└── scripts/
    ├── daily_run.sh       # Daily cron script
    └── daily_report.sh    # Report script
```

## Data Access

### Free Data (Currently Using)
- **Sentinel-1:** SAR, 6-day revisit, cloud-independent
- **Sentinel-2:** Optical, 5-day revisit, 10m resolution
- **Landsat:** Optical, 16-day revisit, 30m resolution

### Paid Data (Future)
- **Planet:** Daily 3m resolution
- **Maxar:** 30cm resolution, on-demand tasking
- **Iceye:** SAR, hourly revisit for specific targets

## Latency & Edge

| Data Source | Latency | Edge vs Official Reports |
|-------------|---------|--------------------------|
| Sentinel-1/2 | 1-3 days | Beats EIA by 2 days |
| Planet | Same day | Beats earnings by weeks |
| Maxar | Hours | Real-time supply chain |

## Risks & Limitations

1. **Detection Accuracy**
   - SAR: Good for large ships, misses small boats
   - Optical: Cloud cover blocks data
   - Resolution: May miss small vehicles

2. **Data Latency**
   - Free data: 1-3 days
   - Market can move before data arrives

3. **Coverage Gaps**
   - Sentinel revisit: 5-6 days
   - May miss short-term events

4. **Political Risk**
   - Satellite data shows what happened
   - Cannot predict future political decisions

## Next Steps

1. ✅ Hormuz chokepoint monitoring
2. ✅ Multi-asset paper trading
3. 🔲 Add retail parking monitoring (Walmart, Costco)
4. 🔲 Add oil storage monitoring (Cushing)
5. 🔲 Add agriculture monitoring (Corn belt)
6. 🔲 Integrate real-time price feeds
7. 🔲 Add alert system for actionable signals
8. 🔲 Backtest historical signals vs actual prices

---

**Disclaimer:** This is a paper trading simulation for research purposes. Not financial advice. Past satellite data does not guarantee future trading success.
