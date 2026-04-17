# Retail Parking Signal

> **Detection Method:** Vehicle detection (YOLOv8) on high-res optical imagery
> **Applied To:** [[walmart_hq]], [[costco_hq]]

## How It Works

Parking lot vehicle counts at retail HQ locations serve as a proxy for foot traffic and consumer spending. Deviations from baselines generate signals.

## Key Metric

- **Vehicle count anomaly:** deviation from historical parking occupancy

## Signal Quality Notes

_Currently uses simulated/placeholder data. Needs vehicle detection model deployment._

## Cross-References

- Feeds into [[us_retail]] meta-signal
- Impacts [[WMT]], [[COST]], [[XRT]]
