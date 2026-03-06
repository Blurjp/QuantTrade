# AIS Data Schema

This document describes the expected schema for AIS (Automatic Identification System) data
used in the SAR Throughput Pipeline calibration module.

## Required Columns (Parquet Format)

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `mmsi` | int64 | Maritime Mobile Service Identity | 9-digit vessel ID |
| `datetime` | datetime64[ns] | UTC timestamp | ISO 8601 format |
| `latitude` | float64 | Latitude in decimal degrees | [-90, 90] |
| `longitude` | float64 | Longitude in decimal degrees | [-180, 180] |

## Optional Columns

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `speed` | float64 | Speed over ground (knots) | [0, 102.2] |
| `heading` | float64 | True heading (degrees) | [0, 359], 511=N/A |
| `course` | float64 | Course over ground (degrees) | [0, 360] |
| `status` | int32 | Navigational status | 0-15 (AIS standard) |
| `vessel_name` | string | Vessel name | max 20 chars |
| `vessel_type` | int32 | Ship type code | 0-99 |
| `length` | float64 | Vessel length (meters) | > 0 |
| `width` | float64 | Vessel width (meters) | > 0 |
| `draught` | float64 | Vessel draught (meters) | > 0 |

## Example Parquet Schema (PyArrow)

```python
import pyarrow as pa

schema = pa.schema([
    ('mmsi', pa.int64()),
    ('datetime', pa.timestamp('ns', tz='UTC')),
    ('latitude', pa.float64()),
    ('longitude', pa.float64()),
    ('speed', pa.float64()),
    ('heading', pa.float64()),
    ('course', pa.float64()),
    ('status', pa.int32()),
    ('vessel_name', pa.string()),
    ('vessel_type', pa.int32()),
    ('length', pa.float64()),
    ('width', pa.float64()),
])
```

## Data Sources

### Recommended AIS Data Providers

1. **MarineCadastre (US)**
   - Free US coastal AIS data
   - https://marinecadastre.gov/ais/

2. **Spire Maritime**
   - Commercial global AIS
   - https://spire.com/maritime/

3. **Exact Earth**
   - Commercial satellite AIS
   - https://www.exactearth.com/

4. **CLS (Collecte Localisation Satellites)**
   - Commercial AIS data
   - https://www.cls.fr/

### Data Preparation

Convert raw AIS to Parquet format:

```python
import pandas as pd

# Load raw AIS (CSV, JSON, etc.)
ais_df = pd.read_csv('ais_data.csv')

# Ensure required columns exist
required = ['mmsi', 'datetime', 'latitude', 'longitude']
assert all(col in ais_df.columns for col in required)

# Parse datetime
ais_df['datetime'] = pd.to_datetime(ais_df['datetime'], utc=True)

# Filter to AOI (Strait of Hormuz example)
ais_df = ais_df[
    (ais_df['latitude'] >= 24.5) &
    (ais_df['latitude'] <= 27.0) &
    (ais_df['longitude'] >= 56.0) &
    (ais_df['longitude'] <= 58.0)
]

# Save to Parquet
ais_df.to_parquet('data/ais_hormuz.parquet', index=False)
```

## Validation

Use the schema validator in `pipeline/ais.py`:

```python
from pipeline.ais import validate_ais_data

ais_df = pd.read_parquet('data/ais_hormuz.parquet')
is_valid, errors = validate_ais_data(ais_df)
if not is_valid:
    print(f"Validation errors: {errors}")
```
