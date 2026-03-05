"""
QuantTrade - Hormuz SAR Throughput Pipeline

Production pipeline for estimating Hormuz chokepoint throughput
using Sentinel-1 SAR detections, calibrated against AIS data.
"""

__version__ = "0.1.0"
__author__ = "blurjp"

from pipeline.manifest import run_manifest_builder
from pipeline.loader import load_scenes_from_manifest, iter_scenes, compute_coverage_score
from pipeline.preprocess import preprocess_scene
from pipeline.detection import run_detection_pipeline
from pipeline.tracking import link_detections
from pipeline.crossings import infer_crossings
from pipeline.metrics import aggregate_daily_metrics
from pipeline.calibration import fit_bias_model, apply_bias_correction

__all__ = [
    "run_manifest_builder",
    "load_scenes_from_manifest",
    "iter_scenes",
    "compute_coverage_score",
    "preprocess_scene",
    "run_detection_pipeline",
    "link_detections",
    "infer_crossings",
    "aggregate_daily_metrics",
    "fit_bias_model",
    "apply_bias_correction"
]
