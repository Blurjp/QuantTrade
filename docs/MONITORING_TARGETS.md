# QuantTrade Monitoring Targets - Complete List

## Active Targets (6)

### 1. 🛢️ Strait of Hormuz (Production Ready)
| Attribute | Value |
|-----------|-------|
| **Type** | Chokepoint |
| **Status** | ✅ Active |
| **Instruments** | WTI, Brent, XLE |
| **Data Source** | Sentinel-1 SAR |
| **Detection** | Ships (CFAR) |
| **Update** | Daily |
| **Signal** | Throughput anomaly |

**Current Signal:** Long disruption risk (Low confidence)
**Trading Position:** SHORT WTI @ $120, Target $102

---

### 2. 🛒 Walmart HQ Parking
| Attribute | Value |
|-----------|-------|
| **Type** | Retail Parking |
| **Status** | ✅ Active |
| **Instruments** | WMT |
| **Location** | Bentonville, AR |
| **Data Source** | Sentinel-2 / Maxar |
| **Detection** | Vehicles (YOLO) |
| **Update** | Weekly |

**Signal Logic:**
- Traffic > baseline → LONG WMT
- Traffic < baseline → SHORT WMT

---

### 3. 🛒 Costco HQ Parking
| Attribute | Value |
|-----------|-------|
| **Type** | Retail Parking |
| **Status** | ✅ Active |
| **Instruments** | COST |
| **Location** | Issaquah, WA |
| **Data Source** | Sentinel-2 / Maxar |
| **Detection** | Vehicles (YOLO) |
| **Update** | Weekly |

---

### 4. ⛽ Cushing Oil Storage
| Attribute | Value |
|-----------|-------|
| **Type** | Oil Storage |
| **Status** | ✅ Active |
| **Instruments** | WTI, USO |
| **Location** | Cushing, OK |
| **Capacity** | ~90M barrels |
| **Data Source** | Sentinel-1 / Landsat |
| **Detection** | Tank roof height |
| **Update** | Weekly |

**Signal Logic:**
- Levels rising → BEARISH oil
- Levels falling → BULLISH oil
- Compare vs EIA weekly report

---

### 5. 🌾 Iowa Corn Belt
| Attribute | Value |
|-----------|-------|
| **Type** | Agricultural |
| **Status** | ✅ Active |
| **Instruments** | Corn, Soybeans |
| **Location** | Iowa, USA |
| **Data Source** | Sentinel-2 |
| **Detection** | NDVI |
| **Update** | Weekly |
| **Season** | April-October |

**Signal Logic:**
- NDVI > baseline → SHORT crop (bumper harvest)
- NDVI < baseline → LONG crop (supply concerns)

---

### 6. 🚗 Detroit Auto Inventory
| Attribute | Value |
|-----------|-------|
| **Type** | Auto Inventory |
| **Status** | ✅ Active |
| **Instruments** | F, GM, CARZ |
| **Location** | Detroit, MI |
| **Data Source** | Sentinel-2 |
| **Detection** | Parked vehicles |
| **Update** | Weekly |

**Signal Logic:**
- Inventory rising → SHORT auto (oversupply)
- Inventory falling → LONG auto (strong demand)

---

### 7. 📦 LA/Long Beach Port
| Attribute | Value |
|-----------|-------|
| **Type** | Port Logistics |
| **Status** | ✅ Active |
| **Instruments** | XLI, FDX, UPS |
| **Location** | Los Angeles, CA |
| **Significance** | 40% of US imports |
| **Data Source** | Sentinel-1 |
| **Detection** | Ships, containers |
| **Update** | Daily |

**Signal Logic:**
- Ship count anomaly → Supply chain activity indicator

---

## Inactive Targets (2)

| Target | Type | Status | Reason |
|--------|------|--------|--------|
| Suez Canal | Chokepoint | 🔲 Inactive | Lower priority |
| Strait of Malacca | Chokepoint | 🔲 Inactive | Lower priority |

---

## Implementation Status

| Monitoring Type | Detection | Signal Gen | Status |
|-----------------|-----------|------------|--------|
| **Chokepoint** | ✅ CFAR SAR | ✅ Production | Ready |
| **Retail Parking** | 🔧 YOLO needed | ✅ Logic ready | Needs model |
| **Oil Storage** | 🔧 Algorithm needed | ✅ Logic ready | Needs pipeline |
| **Agricultural** | 🔧 NDVI pipeline | ✅ Logic ready | Needs pipeline |
| **Auto Inventory** | 🔧 YOLO needed | ✅ Logic ready | Needs model |
| **Port Logistics** | ✅ SAR ships | ✅ Logic ready | Ready |

---

## Next Steps

1. **Immediate** (Today)
   - ✅ AOI configurations created
   - ✅ Signal logic implemented
   - ✅ Registry updated

2. **Short-term** (This week)
   - 🔲 Add vehicle detection model (YOLOv8)
   - 🔲 Add NDVI calculation pipeline
   - 🔲 Add tank level detection algorithm
   - 🔲 Integrate price feeds

3. **Medium-term** (This month)
   - 🔲 Backtest signals vs actual prices
   - 🔲 Add more retail targets (Target, Home Depot)
   - 🔲 Add China ports (Shanghai, Shenzhen)
   - 🔲 Add Brazil soybeans

---

## File Structure

```
configs/
├── aoi_hormuz.geojson          # Hormuz chokepoint
├── aoi_retail_walmart.geojson  # Walmart parking
├── aoi_retail_costco.geojson   # Costco parking
├── aoi_storage_cushing.geojson # Cushing tanks
├── aoi_agri_iowa.geojson       # Iowa corn
├── aoi_auto_detroit.geojson    # Detroit lots
├── aoi_port_lalongbeach.geojson # LA port
├── monitoring_types.json       # Type definitions
├── trading_targets.json        # Trading config
└── regions/
    └── registry_v2.json        # Region registry

pipeline/
├── detection_multi.py          # Multi-type detection
└── signals_multi.py            # Multi-type signals

paper_trading/
├── multi_asset_portfolio.py    # Portfolio manager
└── daily_multi_report.py       # Daily reports
```

---

**Total Active Targets: 6**
**Total Instruments Covered: 15+**
