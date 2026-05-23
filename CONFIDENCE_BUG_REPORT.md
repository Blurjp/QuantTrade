# QuantTrade Confidence Generation Bug Report

**Audit Date:** 2026-05-23  
**Scope:** All confidence level generation code across 14 signal types  
**Total Files Audited:** 12 Python modules + JSON signal outputs

---

## CRITICAL FINDINGS

### BUG #1 — DEAD CODE: `combined_z < -2.0` Can Never Be True (SHORT Signals Impossible)
**Severity:** HIGH  
**Files Affected:**
- `pipeline/atmospheric.py` line 674
- `pipeline/thermal_infrared.py` lines 432, 448, 463
- `pipeline/solar_irradiance.py` lines 407, 439
- `pipeline/sea_surface_temperature.py` line 472

**Issue:** `combined_z` is computed as an average of **absolute** z-scores, so it's always ≥ 0. The `elif combined_z < -2.0` branch is dead code. These modules can **never** generate SHORT signals through the default signal path. 

Examples:
```python
# atmospheric.py line 584 — always >= 0
combined_z = (abs(no2_z) * 0.4 + abs(so2_z) * 0.3 + abs(co2_z) * 0.2 + abs(ch4_z) * 0.1)

# thermal_infrared.py line 356 — always >= 0
combined_z = (abs(temp_z_score) + abs(coverage_z_score)) / 2

# Then later:
elif combined_z < -2.0:   # IMPOSSIBLE — can never be True
    direction = "short"
```

**Impact:** 5 out of 14 signal types (atmospheric, thermal_infrared, solar_irradiance, sea_surface_temperature, vegetation_health) can never produce SHORT signals via the combined_z path. This causes massive confidence = 50 clustering and eliminates half the trading signal space.

**Fix:** Use signed z-scores for direction determination. Keep absolute values only for confidence magnitude:
```python
# Correct approach:
combined_z_signed = (no2_z * 0.4 + so2_z * 0.3 + co2_z * 0.2 + ch4_z * 0.1)
combined_z_magnitude = abs(combined_z_signed)

if combined_z_signed > 2.0:
    direction = "long"
    confidence = min(100, 60 + combined_z_magnitude * 10)
elif combined_z_signed < -2.0:
    direction = "short"
    confidence = min(100, 60 + combined_z_magnitude * 10)
```

---

### BUG #2 — SST Confidence Artificially Capped at 55 (Makes Confidence Meaningless)
**Severity:** HIGH  
**File:** `pipeline/sea_surface_temperature.py` line 490  
**Code:**
```python
"confidence": min(confidence, 55),  # Cap SST confidence to avoid overriding direct ag signals
```

**Issue:** ALL SST signals have confidence capped to 55 maximum. Even a strong El Niño with calculated confidence of 90+ becomes 55. The SST module can only ever produce confidence values of 50 (neutral) or 55 (non-neutral). The dynamic range is destroyed.

**Evidence:** All 9 SST output signals have confidence = 55 — every single one.

**Impact:** SST signals provide no useful confidence information. A mild and a severe anomaly look identical.

**Fix:** Cap at a higher value (e.g., 75) or use a proportional dampening instead of a hard cap.

---

### BUG #3 — Massive Clustering at Confidence = 50 (Neutral Bias)
**Severity:** HIGH  
**Files Affected:** All standalone pipeline modules (thermal_infrared, atmospheric, nighttime_lights, solar_irradiance, sea_surface_temperature, soil_moisture, vegetation_health, precipitation)

**Issue:** Every module uses the same pattern: when the z-score is between -2.0 and +2.0 (the "normal" range), confidence is hardcoded to exactly 50. Since simulated data with static baselines rarely produces z-scores > 2.0, the vast majority of signals are neutral with confidence = 50.

**Evidence from actual output data:**
| Module | Total Signals | Confidence = 50 | Percentage |
|--------|--------------|-----------------|------------|
| atmospheric | 9 | 9 | 100% |
| thermal_infrared | 13 | 12 | 92% |
| sea_surface_temperature | 9 | 0 (all 55 due to cap) | N/A |
| vegetation_health | 16 | 0 (all 35 due to penalty) | N/A |

Across all signal JSON files: **124 out of ~200 signals** have confidence exactly 50.

**Root Cause:** Two factors compound:
1. Hardcoded confidence = 50 for all neutral signals (no gradation)
2. Simulated data with static baselines and small noise rarely exceeds z=2.0 threshold

**Fix:** Use a continuous confidence mapping even for neutral signals, e.g., `confidence = 50 + abs(combined_z) * 5` for the -2.0 to +2.0 range, giving values from 50-60 instead of always 50.

---

### BUG #4 — Vegetation Health Confidence Collapses to 35.0 for Simulated Data
**Severity:** MEDIUM  
**File:** `pipeline/vegetation_health.py` lines 716-718  
**Code:**
```python
if not is_real_data:
    confidence = round(max(35.0, confidence * 0.7), 1)
    confidence_penalty = 30
```

**Issue:** For neutral signals (confidence = 50), the penalty computes: `max(35.0, 50 * 0.7) = max(35.0, 35.0) = 35.0`. This creates a massive cluster at exactly 35.0. For signals that were 60, the penalty gives `max(35.0, 42.0) = 42.0` — only marginally different from a neutral signal.

**Evidence:** 11 out of 16 vegetation_health signals have confidence exactly 35.0.

**Fix:** Apply the penalty before the neutral assignment, or use a different penalty formula that preserves more dynamic range, e.g., `confidence = round(max(35.0, confidence - 15), 1)`.

---

## MEDIUM SEVERITY FINDINGS

### BUG #5 — Incompatible Confidence Formats Across Modules
**Severity:** MEDIUM  
**Files Affected:** All pipeline modules vs. `pipeline/signals.py` vs. `pipeline/signals_multi.py`

**Issue:** Three incompatible confidence representation systems coexist:

| System | Modules | Format | Range |
|--------|---------|--------|-------|
| String labels | signals.py, signals_multi.py, signal_generator_optimized.py | "High"/"Medium"/"Low" | Categorical |
| Numeric 0-100 + label | vegetation_health.py, precipitation.py, cattle_feedlot.py | 35.0, "Low" | 0-100 + string |
| Numeric 0-100 only | thermal_infrared.py, atmospheric.py, nighttime_lights.py, solar_irradiance.py, sea_surface_temperature.py, soil_moisture.py | 50, 82.7 | 0-100 |

The daily pipeline runner (`pipeline/run_daily.py` line 71) expects string labels:
```python
confidence = CONFIDENCE_WEIGHTS.get(signal.get("confidence", "Low"), 0.25)
```
But standalone modules output numeric values. When `confidence = 50` (integer), `CONFIDENCE_WEIGHTS.get(50, 0.25)` returns 0.25 (the default), treating numeric 50 the same as "Low". This means the portfolio manager **ignores the actual confidence magnitude** of standalone module signals.

**Fix:** Normalize all modules to output both numeric confidence (0-100) AND string label, or make the pipeline runner handle both formats.

---

### BUG #6 — `_confidence_label()` Duplicated 3 Times with Different Logic
**Severity:** MEDIUM  
**Files:**
- `pipeline/signals.py` lines 17-24: Input 0-1 scale, ≥0.75=High, ≥0.55=Medium
- `pipeline/vegetation_health.py` lines 28-39: Auto-detects 0-100 vs 0-1, same thresholds
- `pipeline/precipitation.py` lines 30-41: Same as vegetation_health.py

**Issue:** The auto-detection (`if score > 1.0: score = score / 100.0`) is fragile. A confidence value of exactly 1.0 (1% on 0-100 scale, or 100% on 0-1 scale) would be treated as 0-1 scale, yielding "Medium" (≥0.55). But 1% confidence should be "Low".

**Fix:** Extract to a shared utility. Remove auto-detection; require explicit scale parameter.

---

### BUG #7 — `signals.py` Neutral Signals Get "Medium" Confidence (Not "Low")
**Severity:** MEDIUM  
**File:** `pipeline/signals.py` lines 481, 555, 620, 686  
**Code (4 instances):**
```python
# When signal is "Neutral crop conditions" / "Neutral inventory" / etc:
confidence = "Medium"
```

**Issue:** In `signals_multi.py`, "No trade" signals always get `confidence = "Low"`. But in `signals.py`, neutral/no-trade signals get `confidence = "Medium"`. This inconsistency means the same underlying condition ("no actionable signal") produces different confidence levels depending on which code path is used.

**Impact:** The daily pipeline uses `signals.py` via `generate_signal()`, so neutral signals get "Medium" confidence, which maps to weight 0.6 in the portfolio. In `signals_multi.py`, they'd get "Low" (weight 0.25). This causes the portfolio to over-weight neutral signals.

---

### BUG #8 — Duplicate Signal Logic with Different Thresholds (signals.py vs signals_multi.py)
**Severity:** MEDIUM  
**Files:** `pipeline/signals.py` vs `pipeline/signals_multi.py`

**Chokepoint thresholds:**
- `signals.py` line 675: `throughput_change < -threshold * baseline_mean` (threshold=0.1, i.e., 10% deviation)
- `signals_multi.py` line 88: `current_throughput < 0.5 * baseline_mean` (50% deviation!)

**Oil storage thresholds:**
- `signals.py` line 607-616: threshold=5.0 (absolute), confidence by % change
- `signals_multi.py` line 223-240: fill_pct > 80 AND trend > 0, hardcoded "High" confidence

**Impact:** The same data produces completely different signals depending on which module processes it. The daily pipeline uses `signals.py`, while the UI/backtesting may use `signals_multi.py`.

---

### BUG #9 — Division by Zero Risks in Deviation Calculations
**Severity:** MEDIUM  
**Files:**
- `pipeline/sea_surface_temperature.py` line 329-331: `baseline["sst"]["mean"]` not guarded
- `pipeline/solar_irradiance.py` lines 340-342, 345-347, 350-352: 3 unguarded divisions
- `pipeline/precipitation.py` lines 490-492: `baseline["precipitation"]["mean"]` not guarded

**Issue:** While z-score calculations have `if std > 0` guards, the deviation_pct calculations do not. If baseline mean is 0 (e.g., from failed baseline computation), these will throw ZeroDivisionError.

---

### BUG #10 — Inconsistent Confidence Formulas Across Modules
**Severity:** MEDIUM  
**Files:** All standalone modules

| Module | Confidence Formula | Range |
|--------|--------------------|-------|
| atmospheric | `min(100, 60 + combined_z * 10)` | 60-100 |
| thermal_infrared | `min(100, 60 + combined_z * 10)` | 60-100 |
| nighttime_lights | `min(100, 60 + abs(z_score) * 10)` | 60-100 |
| solar_irradiance | `min(100, 60 + combined_z * 10)` | 60-100 |
| vegetation_health | `min(100, 60 + impact_score * 0.5)` | 60-100 |
| soil_moisture (drought) | `min(100, 65 + impact_score * 0.5)` | 65-100 |
| cattle_feedlot | `int(abs(supply_score) * 25 + 10)` | 10-100 |

The base confidence varies from 10 to 65, and the multiplier varies from 0.3 to 25. A z-score of 2.0 produces confidence=80 in atmospheric but would need a completely different input to achieve the same in cattle_feedlot.

---

### BUG #11 — Precipitation "Normal" = SHORT (Asymmetric Default)
**Severity:** MEDIUM  
**File:** `pipeline/precipitation.py` lines 580-583  
**Code:**
```python
else:  # normal
    direction = "short"
    confidence = 55
```

**Issue:** Normal precipitation produces a SHORT signal at confidence 55. This means the default state generates a directional trading signal rather than neutral. Combined with the 70% simulated-data penalty (confidence becomes `max(35, 55*0.7) = max(35, 38.5) = 38.5`), normal precipitation still produces a short signal.

---

## LOW SEVERITY FINDINGS

### BUG #12 — SST Hurricane Zone Hardcoded Confidence
**Severity:** LOW  
**File:** `pipeline/sea_surface_temperature.py` lines 402-413  
**Issue:** Hurricane zone confidence is hardcoded (75, 50, 50) regardless of SST anomaly magnitude. A 28.1°C reading and a 32.0°C reading both get confidence=75 if thermal_stress="high".

---

### BUG #13 — Confidence Multiplication Bug in Critical Season
**Severity:** LOW  
**Files:** `pipeline/vegetation_health.py` line 645, `pipeline/precipitation.py` line 549, `pipeline/soil_moisture.py` line 436  
**Code:**
```python
confidence = min(100, confidence * 1.2)
```

**Issue:** When initial confidence is already capped at 100 (from `min(100, ...)`), the 1.2x boost has no effect: `min(100, 100 * 1.2) = 100`. When initial confidence is 80, boost gives 96. When initial is 90, boost gives 100. The boost effect is inconsistent.

**Fix:** Apply the multiplier before the min(100, ...) cap, or use an additive boost like `confidence = min(100, confidence + 10)`.

---

### BUG #14 — Cattle Feedlot Neutral Confidence Divider Inconsistency
**Severity:** LOW  
**File:** `pipeline/cattle_feedlot.py` line 237  
**Code:**
```python
confidence = max(MIN_CONFIDENCE, confidence // 2)
```

**Issue:** Uses integer division (`//`) while the rest of the codebase uses floating-point division. This can produce slightly different results.

---

### BUG #15 — Static Baselines Make All Signals Nearly Identical
**Severity:** LOW (design issue)  
**Files:** All modules with `calculate_baseline()` methods

**Issue:** Every module uses hardcoded static baselines instead of computing from historical data. This was done intentionally to prevent download loops, but it means:
- Baseline std is always a fixed ratio of baseline mean (e.g., 10%, 20%, 25%)
- Z-scores are deterministic given the same date seed
- Confidence values have very little day-to-day variation

This is the root cause of the confidence clustering problem.

---

## SUMMARY STATISTICS FROM ACTUAL OUTPUT DATA

| Confidence Value | Count | Source |
|-----------------|-------|--------|
| 50 | 124 | Most modules (neutral default) |
| 35.0 | 11 | vegetation_health (simulated penalty) |
| 55 | 10 | sea_surface_temperature (capped) + precipitation |
| 100 | 9 | soil_moisture (drought cap hit) |
| 10 | 9 | cattle_feedlot (MIN_CONFIDENCE) |
| 95.0 | 8 | soil_moisture |
| 66.5 | 6 | precipitation (drought formula) |
| 70.0 | 4 | soil_moisture |

**Key observation:** 67% of all signals have confidence of exactly 50, 55, or 35 — values that come from hardcoded constants rather than data-driven calculations.

---

## RECOMMENDED PRIORITY FIXES

1. **Fix combined_z sign issue (BUG #1)** — This is the highest-impact bug. 5 signal types can never produce SHORT signals.
2. **Remove SST confidence cap of 55 (BUG #2)** — Destroys SST signal utility.
3. **Replace hardcoded confidence=50 with graduated formula (BUG #3)** — Will eliminate the massive clustering.
4. **Unify confidence format (BUG #5)** — Portfolio manager currently ignores numeric confidence values.
5. **Fix signals.py "Medium" for neutral (BUG #7)** — Over-weights non-actionable signals.
