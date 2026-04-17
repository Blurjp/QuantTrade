# Agricultural Signal

> **Detection Method:** NDVI/EVI from Sentinel-2 optical imagery
> **Applied To:** [[brazil_soy_north]], [[brazil_soy_central]], [[brazil_soy_southeast]], [[iowa_corn]]

## How It Works

Vegetation indices (NDVI, EVI) from Sentinel-2 are tracked over cropland. Anomalies in crop health indicate potential yield deviations, which translate to price signals for agricultural commodities.

## Key Metric

- **NDVI anomaly:** deviation from same-period historical average

## Signal Quality Notes

_Production-grade. Uses real data from Planetary Computer when credentials are available._

## Cross-References

- Brazil sub-regions feed into [[brazil_soy]] meta-signal
- Impacts [[Soybeans]], [[Corn]], [[WEAT]]
