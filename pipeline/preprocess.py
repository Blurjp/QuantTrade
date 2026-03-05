"""
SAR Preprocessing

Minimal preprocessing for Sentinel-1 SAR data:
- Water mask (restrict to water pixels)
- Intensity normalization
- Metadata preservation
"""

import numpy as np
import xarray as xr
from typing import Optional, Tuple

def apply_water_mask(
    ds: xr.Dataset,
    land_threshold_db: float = -10.0
) -> xr.Dataset:
    """
    Apply water mask based on backscatter threshold.

    Water typically has lower backscatter than land.

    Args:
        ds: Input dataset
        land_threshold_db: Threshold in dB (below = likely water)

    Returns:
        Masked dataset
    """
    # Convert to dB if needed
    if ds["vv"].max() > 0:
        vv_db = 10 * np.log10(ds["vv"].clip(min=1e-10))
    else:
        vv_db = ds["vv"]

    # Create water mask
    water_mask = vv_db < land_threshold_db

    # Apply mask
    ds_masked = ds.where(water_mask)

    return ds_masked

def normalize_intensity(
    ds: xr.Dataset,
    method: str = "percentile"
) -> xr.Dataset:
    """
    Normalize SAR intensity values.

    Args:
        ds: Input dataset
        method: Normalization method (percentile, minmax)

    Returns:
        Normalized dataset
    """
    if method == "percentile":
        # Percentile-based normalization (robust)
        p2 = ds["vv"].quantile(0.02)
        p98 = ds["vv"].quantile(0.98)

        ds_norm = (ds["vv"] - p2) / (p98 - p2)
        ds_norm = ds_norm.clip(0, 1)

    elif method == "minmax":
        # Min-max normalization
        vmin = ds["vv"].min()
        vmax = ds["vv"].max()

        ds_norm = (ds["vv"] - vmin) / (vmax - vmin)

    ds["vv_normalized"] = ds_norm

    return ds

def compute_qc_metrics(ds: xr.Dataset) -> dict:
    """
    Compute quality control metrics for a scene.

    Args:
        ds: Input dataset

    Returns:
        QC metrics dict
    """
    metrics = {}

    # Backscatter statistics
    if "vv" in ds:
        metrics["vv_mean"] = float(ds["vv"].mean())
        metrics["vv_median"] = float(ds["vv"].median())
        metrics["vv_std"] = float(ds["vv"].std())

    if "vh" in ds:
        metrics["vh_mean"] = float(ds["vh"].mean())
        metrics["vh_median"] = float(ds["vh"].median())
        metrics["vh_std"] = float(ds["vh"].std())

    # Coverage
    if "vv" in ds:
        valid_pixels = ds["vv"].notnull().sum()
        total_pixels = ds["vv"].size
        metrics["coverage_fraction"] = float(valid_pixels / total_pixels)

    return metrics

def preprocess_scene(
    ds: xr.Dataset,
    apply_mask: bool = True,
    normalize: bool = True
) -> Tuple[xr.Dataset, dict]:
    """
    Full preprocessing pipeline for a single scene.

    Args:
        ds: Input dataset
        apply_mask: Apply water mask
        normalize: Normalize intensity

    Returns:
        Preprocessed dataset and QC metrics
    """
    if apply_mask:
        ds = apply_water_mask(ds)

    if normalize:
        ds = normalize_intensity(ds)

    # Compute QC metrics
    qc = compute_qc_metrics(ds)

    return ds, qc

if __name__ == "__main__":
    print("SAR Preprocessing module")
    print("Usage: python -m pipeline.preprocess --input <scene_path>")
