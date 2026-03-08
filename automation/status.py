"""
Persisted status helpers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

def status_path(output_base: str, region_id: str) -> Path:
    return Path(output_base) / "regions" / region_id / "status" / "latest_status.json"


def load_region_status(output_base: str, region_id: str) -> dict:
    path = status_path(output_base, region_id)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_region_status(output_base: str, region_id: str, status: dict) -> str:
    path = status_path(output_base, region_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    status = dict(status)
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(status, indent=2, default=str))
    return str(path)
