"""
Alert helpers for Telegram and persisted alert logs.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


def _alert_log_path(output_base: str, region_id: str) -> Path:
    return Path(output_base) / "regions" / region_id / "alerts" / "alert_log.parquet"


def append_alert_log(output_base: str, region_id: str, alert_type: str, payload: dict) -> str:
    log_path = _alert_log_path(output_base, region_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "region": region_id,
        "alert_type": alert_type,
        "payload": json.dumps(payload, default=str),
    }
    if log_path.exists():
        existing = pd.read_parquet(log_path)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row])
    updated.to_parquet(log_path, index=False)
    return str(log_path)


def send_telegram_message(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required")

    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=30,
    )
    response.raise_for_status()


def send_signal_alert(payload: dict, dry_run: bool = False) -> dict:
    message = (
        f"[Signal] {payload['region']} {payload['date']}\n"
        f"Signal: {payload['signal']} ({payload['confidence']})\n"
        f"Throughput: {payload['throughput_value']}\n"
        f"7d baseline: {payload['baseline_value']}\n"
        f"Coverage: {payload['coverage']}\n"
        f"Instruments: {', '.join(payload.get('instruments', []))}\n"
        f"UI: {payload.get('ui_path', 'n/a')}"
    )
    if not dry_run:
        send_telegram_message(message)
    return {"message": message, "dry_run": dry_run}


def send_failure_alert(payload: dict, dry_run: bool = False) -> dict:
    message = (
        f"[Failure] {payload['region']}\n"
        f"Stage: {payload['stage']}\n"
        f"Error: {payload['error']}\n"
        f"Last success: {payload.get('last_success_at', 'n/a')}"
    )
    if not dry_run:
        send_telegram_message(message)
    return {"message": message, "dry_run": dry_run}
