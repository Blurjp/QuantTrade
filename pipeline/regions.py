"""
Region registry and path resolution for multi-chokepoint monitoring.
"""

import json
from pathlib import Path


REGISTRY_PATH = Path("configs/regions/registry.json")


def load_region_registry(registry_path: str | Path = REGISTRY_PATH) -> dict:
    path = Path(registry_path)
    with open(path) as f:
        data = json.load(f)

    return data.get("regions", {})


def list_regions(registry_path: str | Path = REGISTRY_PATH) -> list[dict]:
    regions = load_region_registry(registry_path)
    return [
        {
            "id": region_id,
            "name": config.get("name", region_id),
            "description": config.get("description", ""),
        }
        for region_id, config in regions.items()
    ]


def get_region_config(region_id: str, registry_path: str | Path = REGISTRY_PATH) -> dict:
    regions = load_region_registry(registry_path)
    if region_id not in regions:
        raise KeyError(f"Unknown region: {region_id}")

    config = dict(regions[region_id])
    config["id"] = region_id
    return config


def resolve_region_paths(region_id: str, registry_path: str | Path = REGISTRY_PATH) -> tuple[str, str]:
    config = get_region_config(region_id, registry_path)
    return config["aoi_path"], config["gate_path"]


def resolve_region_output_base(output_base: str = "outputs", region_id: str = "hormuz") -> str:
    base = Path(output_base)
    region_base = base / "regions" / region_id

    # Backward compatibility for the original single-region layout.
    if region_id == "hormuz":
        return str(base)

    return str(region_base)
