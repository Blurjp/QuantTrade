"""
Region registry helpers for multi-asset monitoring.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


REGISTRY_V2_PATH = Path("configs/regions/registry_v2.json")
REGISTRY_PATH = Path("configs/regions/registry.json")


def _default_registry_path() -> Path:
    if REGISTRY_V2_PATH.exists():
        return REGISTRY_V2_PATH
    return REGISTRY_PATH


def load_registry(registry_path: Optional[Union[str, Path]] = None) -> Dict:
    path = Path(registry_path) if registry_path else _default_registry_path()
    try:
        if not path.exists():
            raise FileNotFoundError(f"Registry not found: {path}")
        raw = path.read_text().strip()
        if not raw:
            raise ValueError(f"Registry file is empty: {path}")
        return json.loads(raw)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        # Fallback: derive regions from scheduler API signals
        import os
        api_url = os.environ.get("SCHEDULER_API_URL", "")
        if api_url:
            import requests
            try:
                resp = requests.get(f"{api_url}/api/all-signals", timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    signals = data.get("signals", {})
                    regions = {}
                    for region_id, sig in signals.items():
                        regions[region_id] = {
                            "name": sig.get("region_name", region_id),
                            "instruments": sig.get("instruments", []),
                            "active": True,
                        }
                    return {"version": 2, "regions": regions}
            except Exception:
                pass
        raise


def load_region_registry(registry_path: Optional[Union[str, Path]] = None) -> Dict:
    data = load_registry(registry_path)
    return data.get("regions", {})


def list_regions(registry_path: Optional[Union[str, Path]] = None) -> List[Dict]:
    regions = load_region_registry(registry_path)
    return [
        {
            "id": region_id,
            "name": config.get("name", region_id),
            "description": config.get("description", ""),
        }
        for region_id, config in regions.items()
    ]


def get_region_config(region_id: str, registry_path: Optional[Union[str, Path]] = None) -> Dict:
    regions = load_region_registry(registry_path)
    if region_id not in regions:
        raise KeyError(f"Unknown region: {region_id}")

    config = dict(regions[region_id])
    config["id"] = region_id
    return config


def resolve_region_paths(region_id: str, registry_path: Optional[Union[str, Path]] = None) -> Tuple[Optional[str], Optional[str]]:
    config = get_region_config(region_id, registry_path)
    return config.get("aoi_path", config.get("aoi_file")), config.get("gate_path", config.get("gate_file"))


def get_active_regions(registry_path: Optional[Union[str, Path]] = None) -> Dict:
    regions = load_region_registry(registry_path)
    return {
        region_id: config
        for region_id, config in regions.items()
        if config.get("active", False)
    }


def resolve_region_output_base(output_base: str = "outputs", region_id: str = "hormuz") -> str:
    base = Path(output_base)
    region_base = base / "regions" / region_id

    # Backward compatibility for the original single-region layout.
    if region_id == "hormuz":
        return str(base)

    return str(region_base)


__all__ = [
    "get_active_regions",
    "get_region_config",
    "list_regions",
    "load_region_registry",
    "load_registry",
    "resolve_region_output_base",
    "resolve_region_paths",
]
