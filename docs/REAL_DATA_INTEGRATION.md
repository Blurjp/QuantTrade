# Real Data Integration Guide

## 🛰️ Overview

This guide explains how to connect your satellite monitoring modules to real satellite data sources, replacing the simulated data with production-ready implementations.

## ✅ Current Status

**Satellite Data Client:** `pipeline/satellite_data_client.py`

The client is designed to:
- Fetch real data from multiple satellite APIs
- Cache data locally to minimize API calls
- Fall back to calculated estimates when APIs are unavailable
- Apply rate limiting to respect API quotas

## 🔑 Required API Keys

### 1. Planetary Computer (Microsoft) - FREE

**What you get:**
- MODIS (Terra/Aqua): NDVI, thermal, surface reflectance
- Sentinel-2: High-resolution imagery
- Sentinel-3: Solar irradiance, SST
- Landsat: Thermal IR, surface temperature
- VIIRS: Nighttime lights

**How to sign up:**
1. Go to: https://planetarycomputer.microsoft.com/
2. Click "Sign Up" or "Get Started"
3. Create a Microsoft account or use existing
4. Navigate to Account Settings → API Keys
5. Generate a new API key

**Environment variable:**
```bash
export PC_SDK_SUBSCRIPTION_KEY="your-key-here"
```

**Add to your shell profile (~/.zshrc or ~/.bashrc):**
```bash
# Planetary Computer API Key
export PC_SDK_SUBSCRIPTION_KEY="your-key-here"
```

### 2. NASA Earthdata - FREE

**What you get:**
- SMAP: Soil moisture data
- GPM: Precipitation data
- MODIS: Vegetation indices
- Other NASA Earth science data

**How to sign up:**
1. Go to: https://urs.earthdata.nasa.gov/
2. Click "Register"
3. Fill out the registration form
4. Verify your email
5. Accept the data usage agreement

**Environment variables:**
```bash
export NASA_EARTHDATA_USERNAME="your-username"
export NASA_EARTHDATA_PASSWORD="your-password"
```

**Add to your shell profile:**
```bash
# NASA Earthdata
export NASA_EARTHDATA_USERNAME="your-username"
export NASA_EARTHDATA_PASSWORD="your-password"
```

### 3. Copernicus Open Access Hub - FREE

**What you get:**
- Sentinel-1: Radar data (soil moisture, oil spills)
- Sentinel-2: High-resolution optical imagery
- Sentinel-3: Ocean/land monitoring
- Sentinel-5P: Atmospheric gases (NO2, SO2, CO2, CH4)

**How to sign up:**
1. Go to: https://scihub.copernicus.eu/dhus/
2. Click "Sign up"
3. Fill out the registration form
4. Verify your email

**Environment variables:**
```bash
export COPERNICUS_USERNAME="your-username"
export COPERNICUS_PASSWORD="your-password"
```

**Add to your shell profile:**
```bash
# Copernicus Open Access Hub
export COPERNICUS_USERNAME="your-username"
export COPERNICUS_PASSWORD="your-password"
```

### 4. NOAA - FREE

**What you get:**
- AVHRR SST: Sea surface temperature
- GOES: Weather satellite data
- Other NOAA satellite products

**How to sign up:**
1. Go to: https://www.ncdc.noaa.gov/
2. Create an account
3. Request API access

**Environment variable:**
```bash
export NOAA_TOKEN="your-token"
```

## 🚀 Quick Start

### Step 1: Set Up Environment Variables

Create a `.env` file in your project root:

```bash
# .env file
# Planetary Computer
PC_SDK_SUBSCRIPTION_KEY=your-key-here

# NASA Earthdata
NASA_EARTHDATA_USERNAME=your-username
NASA_EARTHDATA_PASSWORD=your-password

# Copernicus
COPERNICUS_USERNAME=your-username
COPERNICUS_PASSWORD=your-password

# NOAA
NOAA_TOKEN=your-token
```

**⚠️ Important:** Add `.env` to your `.gitignore`:
```bash
echo ".env" >> .gitignore
```

### Step 2: Load Environment Variables

Add to your Python scripts:

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Access variables
pc_key = os.getenv("PC_SDK_SUBSCRIPTION_KEY")
```

Or install `python-dotenv`:
```bash
pip install python-dotenv
```

### Step 3: Test Data Retrieval

```bash
cd /Users/jianpinghuang/.openclaw/workspace/projects/QuantTrade
python3 -m pipeline.satellite_data_client
```

Expected output:
```
🛰️ Testing Satellite Data Client
============================================================

1. MODIS NDVI:
   Mean NDVI: 0.65

2. GPM Precipitation:
   Precipitation: 12.5 mm

3. SMAP Soil Moisture:
   Soil Moisture: 0.25 m³/m³

✅ All tests completed!
```

## 📊 Data Sources by Module

| Module | Primary Source | Secondary Source | Update Frequency |
|--------|---------------|------------------|------------------|
| Nighttime Lights | VIIRS (Planetary Computer) | DMSP-OLS | Daily |
| Thermal IR | Landsat (Planetary Computer) | Sentinel-3 | Daily |
| Atmospheric | TROPOMI (Copernicus) | OMI | Daily |
| Solar Irradiance | MODIS/Sentinel-3 | CERES | Daily |
| Sea Surface Temp | MODIS/AVHRR | Sentinel-3 | Daily |
| Precipitation | GPM (NASA) | IMERG | Daily |
| Vegetation Health | MODIS NDVI (Planetary Computer) | Sentinel-2 | Daily |
| Soil Moisture | SMAP (NASA) | Sentinel-1 | Daily |

## 🔧 Implementation Details

### Cache System

Data is cached locally to minimize API calls:

```
data/satellite_cache/
├── ndvi/
│   ├── china_shanghai_2026-03-16.json
│   └── ...
├── precipitation/
│   └── ...
├── soil_moisture/
│   └── ...
└── ...
```

Cache duration: 24 hours (configurable)

### Rate Limiting

All API calls are rate-limited to respect quotas:
- Minimum 1 second between requests
- Automatic retry on 429 (rate limit) errors
- Exponential backoff for repeated failures

### Fallback System

When APIs are unavailable:
1. Check cache first
2. Try primary API
3. Try secondary API
4. Fall back to calculated estimate
5. Log warning

## 📈 Production Checklist

Before deploying to production:

- [ ] Set up all API keys
- [ ] Test each data source
- [ ] Configure cache directory
- [ ] Set up monitoring/alerting
- [ ] Document API quotas
- [ ] Implement error handling
- [ ] Add logging
- [ ] Test fallback mechanisms
- [ ] Set up automated testing

## 🆘 Troubleshooting

### Issue: "API key not found"

**Solution:** Make sure environment variables are set:
```bash
# Check if variables are set
echo $PC_SDK_SUBSCRIPTION_KEY
echo $NASA_EARTHDATA_USERNAME

# If not set, add to ~/.zshrc or ~/.bashrc
source ~/.zshrc  # or ~/.bashrc
```

### Issue: "Rate limit exceeded"

**Solution:** Wait and retry. The client automatically handles this with exponential backoff.

### Issue: "No data available for this date"

**Solution:** Some satellites have latency (1-3 days). Try an earlier date.

### Issue: "Authentication failed"

**Solution:** Double-check your credentials:
1. Verify username/password are correct
2. Check if account is active
3. Ensure you've accepted terms of service

## 💰 Cost

All data sources listed here are **FREE**:
- Planetary Computer: Free tier with generous quota
- NASA Earthdata: Free with registration
- Copernicus: Free for research/commercial use
- NOAA: Free with registration

## 📚 API Documentation

- [Planetary Computer](https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/)
- [NASA Earthdata](https://earthdata.nasa.gov/collaborate/open-data-services-and-software/api)
- [Copernicus Open Access Hub](https://scihub.copernicus.eu/twiki/do/view/SciHubUserGuide/FullTextSearch)
- [NOAA API](https://www.ncdc.noaa.gov/cdo-web/webservices/v2)

## 🔐 Security Best Practices

1. **Never commit API keys to git**
   ```bash
   # Add to .gitignore
   .env
   *_key.txt
   *_credentials.json
   ```

2. **Use environment variables**
   ```python
   # Good
   api_key = os.getenv("PC_SDK_SUBSCRIPTION_KEY")
   
   # Bad
   api_key = "sk-abc123..."  # Never do this!
   ```

3. **Rotate keys regularly**
   - Change API keys every 3-6 months
   - Use different keys for development/production

4. **Monitor usage**
   - Check API usage regularly
   - Set up alerts for unusual activity

## 🚀 Next Steps

1. **Set up API keys** (see above)
2. **Test data retrieval**: `python3 -m pipeline.satellite_data_client`
3. **Monitor data quality**: Check cache files in `data/satellite_cache/`
4. **Integrate with trading system**: All modules will automatically use real data
5. **Set up monitoring**: Track API usage and data freshness

## 📊 Expected Data Quality

| Data Type | Resolution | Latency | Accuracy |
|-----------|------------|---------|----------|
| NDVI | 250m | 1-2 days | High |
| Precipitation | 10km | 1-3 days | Medium-High |
| Soil Moisture | 9km | 1-3 days | Medium |
| SST | 1km | 1-2 days | High |
| Atmospheric gases | 5.5km | 1-5 days | Medium-High |
| Nighttime lights | 500m | 1-3 days | High |
| Thermal IR | 100m | 1-2 days | High |
| Solar irradiance | 1km | 1-2 days | Medium-High |

---

**Note:** Even without API keys, the system will continue to function using calculated estimates based on seasonal patterns and historical data. The estimates are reasonable for demonstration purposes but real data should be used for production trading.
