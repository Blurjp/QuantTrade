"""
Instrument mapping helpers for region-aware trading recommendations.
"""

import json
from pathlib import Path


INSTRUMENTS_PATH = Path("configs/instruments.json")


def load_instrument_registry(path: str | Path = INSTRUMENTS_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def list_region_instruments(
    region_id: str,
    enabled_for_backtest: bool | None = None,
    enabled_for_alerts: bool | None = None,
    path: str | Path = INSTRUMENTS_PATH,
) -> list[dict]:
    registry = load_instrument_registry(path)
    instruments = list(registry.get("regions", {}).get(region_id, []))

    if enabled_for_backtest is not None:
        instruments = [
            instrument
            for instrument in instruments
            if bool(instrument.get("enabled_for_backtest")) == enabled_for_backtest
        ]

    if enabled_for_alerts is not None:
        instruments = [
            instrument
            for instrument in instruments
            if bool(instrument.get("enabled_for_alerts")) == enabled_for_alerts
        ]

    return instruments


def get_primary_instrument(region_id: str, path: str | Path = INSTRUMENTS_PATH) -> dict | None:
    instruments = list_region_instruments(region_id, path=path)
    for instrument in instruments:
        if instrument.get("primary"):
            return instrument
    return instruments[0] if instruments else None
