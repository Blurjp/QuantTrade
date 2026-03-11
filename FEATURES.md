# QuantTrade Enhancement Features

## 🎉 New Features Added (2026-03-09)

### 1. 📊 Web Dashboard (Streamlit)

Real-time monitoring and visualization dashboard.

**Features:**
- Portfolio overview with P&L tracking
- Signal performance visualization
- Backtest results display
- Monitoring targets management
- Interactive charts

**Usage:**
```bash
cd /Users/jianpinghuang/.openclaw/workspace/projects/QuantTrade
source .venv/bin/activate
streamlit run dashboard/app.py
```

**Access:** http://localhost:8501

---

### 2. ⛽ EIA Data Integration

Fetches weekly crude oil inventory data from EIA for Cushing storage validation.

**Features:**
- Real-time Cushing inventory data (with API key)
- Demo mode with realistic simulated data
- Signal validation against official data
- Trend analysis

**Usage:**
```python
from pipeline.eia_data import fetch_eia_cushing_report

# Without API key (demo mode)
report = fetch_eia_cushing_report()
print(f"Cushing Inventory: {report['inventory_mb']}M barrels")

# With API key (real data)
import os
os.environ['EIA_API_KEY'] = 'your-key-here'
report = fetch_eia_cushing_report(os.environ['EIA_API_KEY'])
```

**Get API Key:** https://www.eia.gov/opendata/register.php (Free)

---

### 3. 🚗 Vehicle Detection (YOLOv8)

Detects and counts vehicles in parking lot imagery for retail signal generation.

**Features:**
- YOLOv8-based vehicle detection
- Supports images and GeoTIFF files
- Parking lot occupancy analysis
- Signal generation (long/short/neutral)

**Usage:**
```python
from pipeline.vehicle_detection import detect_vehicles_in_parking_lot

# Detect vehicles in an image
result = detect_vehicles_in_parking_lot('parking_lot.jpg')
print(f"Vehicles detected: {result['total_vehicles']}")
print(f"Occupancy: {result['occupancy_pct']}%")
print(f"Signal: {result['signal']}")
```

**Note:** YOLOv8 is trained on ground-level imagery. For satellite imagery:
- Use high-resolution data (Planet 3m or better)
- Calibrate with ground truth data
- Consider as rough estimate only

---

## 📁 New Files

```
QuantTrade/
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── pipeline/
│   ├── eia_data.py         # EIA data integration
│   └── vehicle_detection.py # YOLO vehicle detection
└── outputs/
    └── eia_cache/          # Cached EIA data
```

---

## 🚀 Quick Start

### Start Dashboard
```bash
cd ~/clawd/projects/QuantTrade
source .venv/bin/activate
streamlit run dashboard/app.py
```

### Fetch EIA Data
```bash
python3 -m pipeline.eia_data
```

### Test Vehicle Detection
```bash
python3 -m pipeline.vehicle_detection
```

---

## ⚙️ Configuration

### EIA API Key (Optional)

To get real inventory data instead of demo data:

1. Register at https://www.eia.gov/opendata/register.php
2. Get your free API key
3. Set environment variable:
   ```bash
   export EIA_API_KEY="your-key-here"
   ```

---

## 📊 Integration with Existing System

### Cushing Signal Validation

The EIA data can be used to validate Cushing oil storage signals:

```python
from pipeline.eia_data import EIADataFetcher
from pipeline.signals_multi import generate_signal

# Get satellite signal
satellite_signal = generate_signal("cushing", detection_data)

# Validate with EIA data
fetcher = EIADataFetcher(api_key)
eia_data = fetcher.fetch_cushing_inventory()

validation = fetcher.validate_signal(satellite_signal["direction"], eia_data)

if validation["validated"]:
    print(f"✅ Signal validated: {validation['reason']}")
else:
    print(f"⚠️  Signal not validated: {validation['reason']}")
```

### Retail Parking Analysis

Vehicle detection can improve retail parking signals:

```python
from pipeline.vehicle_detection import VehicleDetector
from pipeline.backfill_multi import backfill_optical_region

# Get Sentinel-2 imagery
result = backfill_optical_region("walmart_hq", "2026-01-01", "2026-03-09")

# Detect vehicles in each scene
detector = VehicleDetector()

for scene in result["scenes"]:
    detections = detector.detect_from_geotiff(scene["path"])
    analysis = detector.analyze_parking_lot(detections, total_spots=1000)
    
    print(f"{scene['date']}: {analysis['occupancy_pct']}% occupancy")
    print(f"  Signal: {analysis['signal']}")
```

---

## 🎯 Next Steps

1. **Dashboard Enhancement**
   - Add real-time price charts
   - Implement alert notifications
   - Add historical performance tracking

2. **EIA Integration**
   - Set up weekly data refresh
   - Add to daily automation
   - Create validation reports

3. **Vehicle Detection**
   - Test with actual Sentinel-2 data
   - Calibrate detection thresholds
   - Add to retail signal generation

---

## 📝 Notes

- **Dashboard**: Auto-refreshes every 5 minutes (configurable)
- **EIA Data**: Updates weekly (Wednesdays)
- **Vehicle Detection**: Works best with imagery < 1m resolution
- **All modules**: Integrated with existing QuantTrade system

---

## 🔗 Related Documentation

- [QuantTrade README](README.md)
- [Signal Types](docs/signals.md)
- [Backtest Results](outputs/backtest/)
- [Portfolio State](outputs/paper_trading/multi_asset_portfolio.json)
