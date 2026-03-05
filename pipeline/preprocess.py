"""
SAR Preprocessing

Minimal preprocessing for Sentinel-1 SAR data:
- Water mask (restrict to water pixels)
- QC metrics computation
"""

import numpy as np
import xarray as xr
from typing import Optional, Tuple


def apply_water_mask(
    ds: xr.Dataset,
    land_threshold_db: float = 35.0
) -> xr.Dataset:
    """
    Apply water mask based on backscatter threshold.

    Land typically has higher backscatter than water.
    For Sentinel-1 IW GRD, water is typically < 20-25 dB.

    Args:
        ds: Input dataset
        land_threshold_db: Threshold in dB (above = likely land, will be masked)

    Returns:
        Masked dataset with land pixels set to NaN
    """
    if "vv" not in ds:
        return ds

    # Convert to dB - handle dask arrays by computing first
    vv_data = ds["vv"]
    if hasattr(vv_data, 'compute'):
        vv_data = vv_data.compute()

    # Convert to dB if needed (positive values > 1 = linear scale)
    if float(vv_data.max()) > 1:
        vv_db = 10 * np.log10(np.clip(vv_data.values, 1e-10, None))
    else:
        vv_db = vv_data.values

    # Create water mask (keep pixels below threshold - these are water)
    # Mask out land (high backscatter)
    land_mask = vv_db > land_threshold_db

    # Apply mask (set land to NaN)
    ds_masked = ds.where(~land_mask)

    return ds_masked


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
        vv_data = ds["vv"]
        if hasattr(vv_data, 'compute'):
            vv_data = vv_data.compute()
        vv_values = vv_data.values

        metrics["vv_mean"] = float(np.nanmean(vv_values))
        metrics["vv_median"] = float(np.nanmedian(vv_values))
        metrics["vv_std"] = float(np.nanstd(vv_values))

    if "vh" in ds:
        vh_data = ds["vh"]
        if hasattr(vh_data, 'compute'):
            vh_data = vh_data.compute()
        vh_values = vh_data.values

        metrics["vh_mean"] = float(np.nanmean(vh_values))
        metrics["vh_median"] = float(np.nanmedian(vh_values))
        metrics["vh_std"] = float(np.nanstd(vh_values))

    # Coverage
    if "vv" in ds:
        vv_data = ds["vv"]
        if hasattr(vv_data, 'compute'):
            vv_data = vv_data.compute()
        vv_values = vv_data.values

        valid_pixels = np.sum(~np.isnan(vv_values))
        total_pixels = vv_values.size
        metrics["coverage_fraction"] = float(valid_pixels / total_pixels) if total_pixels > 0 else 0.0

    return metrics


def preprocess_scene(
    ds: xr.Dataset,
    apply_mask: bool = True,
    compute_qc: bool = True
) -> Tuple[xr.Dataset, dict]:
    """
    Full preprocessing pipeline for a single scene.

    Args:
        ds: Input dataset
        apply_mask: Apply water mask
        compute_qc: Compute QC metrics

    Returns:
        Preprocessed dataset and QC metrics
    """
    if apply_mask:
        ds = apply_water_mask(ds)

    if compute_qc:
        qc = compute_qc_metrics(ds)

    return ds, qc


if __name__ == "__main__":
    print("SAR Preprocessing module")
    print("Usage: python -m pipeline.preprocess --input <scene_path>")
