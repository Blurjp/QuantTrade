# Chokepoint Signal

> **Detection Method:** CFAR ship detection on Sentinel-1 SAR imagery
> **Applied To:** [[hormuz]], [[suez]], [[malacca]], [[bab_el_mandeb]], [[panama_canal]]

## How It Works

Ship traffic through chokepoints is detected using Constant False Alarm Rate (CFAR) algorithm on Sentinel-1 SAR imagery. Throughput deviations from historical baselines generate LONG (disruption = bullish oil) or SHORT (normalized flow = bearish oil) signals.

## Key Metric

- **Throughput anomaly:** deviation of daily ship count from rolling mean

## Signal Quality Notes

_Observations about signal reliability, seasonal patterns, false positives._

## Cross-References

- Feeds into [[global_oil]] meta-signal
- Impacts [[WTI]], [[Brent]], [[XLE]]
