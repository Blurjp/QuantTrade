# QuantTrade - Hormuz SAR Throughput Pipeline

Production-ready pipeline for estimating Hormuz chokepoint throughput using Sentinel-1 SAR detections, calibrated and validated against AIS data.

## Overview

This pipeline produces a **daily Hormuz Throughput Index** (inbound/outbound) derived from Sentinel-1 SAR detections. It uses open-source tools throughout:

- **PySTAC Client** for STAC API search
- **ODC-STAC** for loading Sentinel-1 scenes into xarray
- **Raster Vision** for geospatial CV pipeline
- **GeoPandas + Shapely** for geospatial operations
- **Parquet** for storage

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline for a single day
python -m pipeline.run --date 2024-01-15

# Run full backfill
python -m pipeline.run --start 2023-01-01 --end 2024-01-01
```

## Architecture

```
┌─────────────────┐
│  STAC Search    │ → Manifest (Parquet)
└────────┬────────┘
         │
┌────────▼────────┐
│  Scene Loading  │ → xarray.Dataset (VV/VH)
└────────┬────────┘
         │
┌────────▼────────┐
│  Preprocessing  │ → Water mask, normalization
└────────┬────────┘
         │
┌────────▼────────┐
│ Ship Detection  │ → Detections (Parquet)
└────────┬────────┘
         │
┌────────▼────────┐
│ Tracklet Linking│ → Tracklets + Crossings
└────────┬────────┘
         │
┌────────▼────────┐
│ Daily Metrics   │ → Throughput Index
└────────┬────────┘
         │
┌────────▼────────┐
│ AIS Calibration │ → Bias-corrected Index
└─────────────────┘
```

## Project Structure

```
QuantTrade/
├── configs/
│   ├── aoi_hormuz.geojson
│   └── gate_line.geojson
├── pipeline/
│   ├── manifest.py        # STAC search → manifest
│   ├── loader.py          # ODC-STAC scene loading
│   ├── preprocess.py      # SAR preprocessing
│   ├── detection.py       # Ship detection (Raster Vision)
│   ├── tracking.py        # Tracklet linking
│   ├── crossings.py       # Gate crossing inference
│   ├── metrics.py         # Daily aggregation
│   └── calibration.py     # AIS validation
├── outputs/
│   ├── manifests/
│   ├── detections/
│   ├── tracklets/
│   ├── metrics/
│   └── validation/
├── tests/
├── requirements.txt
└── README.md
```

## Outputs

### Primary (Daily)
- `date`
- `gc_in`, `gc_out` (gate crossings)
- `swt_in`, `swt_out` (size-weighted throughput)
- `coverage_score` (0-1)
- `throughput_index_in`, `throughput_index_out`, `throughput_index_total`

### Secondary (QA/Debug)
- Scene manifests
- Detection parquet
- Tracklet parquet
- Calibration reports

## Configuration

Edit `configs/aoi_hormuz.geojson` and `configs/gate_line.geojson` to define:
- AOI polygon (corridor around chokepoint)
- Gate line (crossing detection line)

## Development Phases

### Phase 1 (2-7 days): Pipeline-first
- End-to-end pipeline with baseline detector
- Stable daily series + coverage scores

### Phase 2: Quality lift
- AIS calibration
- Improved SAR detector
- QA dashboards

### Phase 3: Production hardening
- Multi-provider STAC support
- Robust failure handling
- Storage compaction

## License

MIT
