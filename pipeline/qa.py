"""
QA artifact generation for one-day SAR runs.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from shapely.geometry import LineString, shape

_FALLBACK_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc`\x00\x00"
    b"\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _extract_plot_data(ds, band: str = "vv"):
    data = ds[band].values
    if data.ndim > 2:
        data = data[0]
    data = np.clip(data, 1e-10, None)
    data = 10 * np.log10(data)

    x_name = "longitude" if "longitude" in ds[band].coords else "x"
    y_name = "latitude" if "latitude" in ds[band].coords else "y"
    x_coords = ds[band].coords[x_name].values
    y_coords = ds[band].coords[y_name].values

    return data, x_coords, y_coords


def save_scene_preview(
    ds,
    detections_df: pd.DataFrame,
    aoi_geom: dict,
    gate_line: LineString,
    output_path: str,
    title: str
) -> str:
    """Save a scene preview PNG with AOI, gate line, and detections."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        output_file.write_bytes(_FALLBACK_PNG)
        return str(output_file)

    data, x_coords, y_coords = _extract_plot_data(ds)
    extent = [float(np.min(x_coords)), float(np.max(x_coords)), float(np.min(y_coords)), float(np.max(y_coords))]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(data, cmap="gray", extent=extent, origin="lower", aspect="auto")

    aoi_polygon = shape(aoi_geom)
    aoi_x, aoi_y = aoi_polygon.exterior.xy
    ax.plot(aoi_x, aoi_y, color="#00b894", linewidth=1.5, label="AOI")

    gate_x, gate_y = gate_line.xy
    ax.plot(gate_x, gate_y, color="#d63031", linewidth=2, label="Gate")

    if len(detections_df) > 0:
        ax.scatter(
            detections_df["centroid_lon"],
            detections_df["centroid_lat"],
            s=25,
            c="#fdcb6e",
            edgecolors="#2d3436",
            linewidths=0.5,
            label="Detections",
        )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    return str(output_file)


def write_scene_load_log(
    records: Iterable[dict],
    output_path: str
) -> str:
    """Write scene load log to parquet."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(records))
    df.to_parquet(output_file, index=False)
    return str(output_file)


def write_daily_html(
    target_date: str,
    output_path: str,
    metrics_df: pd.DataFrame,
    load_log_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    preview_paths: list[str],
    manifest_path: str,
    detections_geojson_path: str,
    metrics_path: str,
    load_log_path: str,
) -> str:
    """Write a static HTML summary for a day's outputs."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    metrics_table = metrics_df.to_html(index=False, classes="table") if len(metrics_df) > 0 else "<p>No metrics.</p>"
    load_table = load_log_df.to_html(index=False, classes="table") if len(load_log_df) > 0 else "<p>No load records.</p>"
    detection_preview = (
        detections_df[["scene_id", "datetime", "score", "centroid_lon", "centroid_lat"]]
        .head(50)
        .to_html(index=False, classes="table")
        if len(detections_df) > 0
        else "<p>No detections.</p>"
    )

    preview_html = "\n".join(
        f'<figure><img src="{html.escape(Path(path).name)}" alt="{html.escape(Path(path).stem)}"><figcaption>{html.escape(Path(path).stem)}</figcaption></figure>'
        for path in preview_paths
    ) or "<p>No previews generated.</p>"

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>QuantTrade QA {html.escape(target_date)}</title>
  <style>
    body {{ font-family: Helvetica, Arial, sans-serif; margin: 24px; color: #1f2933; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .links a {{ margin-right: 16px; }}
    .table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    .table th, .table td {{ border: 1px solid #d2d6dc; padding: 6px 8px; text-align: left; }}
    .previews {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; border: 1px solid #d2d6dc; padding: 8px; }}
    img {{ width: 100%; height: auto; display: block; }}
  </style>
</head>
<body>
  <h1>QuantTrade QA: {html.escape(target_date)}</h1>
  <div class="links">
    <a href="../manifests/{html.escape(Path(manifest_path).name)}">Manifest</a>
    <a href="../logs/{html.escape(Path(load_log_path).name)}">Scene Load Log</a>
    <a href="../detections/{html.escape(Path(detections_geojson_path).name)}">Detections GeoJSON</a>
    <a href="../metrics/{html.escape(Path(metrics_path).name)}">Daily Metrics</a>
  </div>
  <h2>Daily Metrics</h2>
  {metrics_table}
  <h2>Scene Load Log</h2>
  {load_table}
  <h2>Detections</h2>
  {detection_preview}
  <h2>Scene Previews</h2>
  <div class="previews">{preview_html}</div>
</body>
</html>"""

    output_file.write_text(page, encoding="utf-8")
    return str(output_file)
