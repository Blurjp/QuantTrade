"""
Unified daily pipeline runner for all monitoring types.

Processes all active regions and generates signals for the multi-asset portfolio.
"""

import argparse
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

from pipeline.regions import get_active_regions, load_registry
from pipeline.detection_dispatcher import run_detection
from pipeline.signals import generate_signal
from paper_trading.multi_asset_portfolio import MultiAssetPortfolio


CONFIDENCE_WEIGHTS = {
    "High": 1.0,
    "Medium": 0.6,
    "Low": 0.25,
}
DEFAULT_CONFIRMATIONS = 2


def _normalize_confidence(confidence) -> str:
    """Convert any confidence format to a standard label (High/Medium/Low).
    
    Handles:
    - String labels: "High", "Medium", "Low" → pass through
    - Numeric 0-100: ≥70 → "High", ≥40 → "Medium", <40 → "Low"
    - Numeric 0-1: ≥0.7 → "High", ≥0.4 → "Medium", <0.4 → "Low"
    """
    if isinstance(confidence, str):
        return confidence if confidence in CONFIDENCE_WEIGHTS else "Low"
    if isinstance(confidence, (int, float)):
        # Auto-detect 0-1 vs 0-100 scale
        if confidence <= 1.0:
            if confidence >= 0.7:
                return "High"
            elif confidence >= 0.4:
                return "Medium"
            else:
                return "Low"
        else:
            # 0-100 scale
            if confidence >= 70:
                return "High"
            elif confidence >= 40:
                return "Medium"
            else:
                return "Low"
    return "Low"


def _meta_signal_text(group_config: Dict, direction: str) -> str:
    label = group_config.get("label", "Meta signal")
    if direction == "LONG":
        return f"{label} meta-long"
    if direction == "SHORT":
        return f"{label} meta-short"
    return f"{label} meta-neutral"


def _meta_bias(group_config: Dict, direction: str) -> str:
    bullish = group_config.get("bullish_bias", "Bullish prices")
    bearish = group_config.get("bearish_bias", "Bearish prices")
    neutral = group_config.get("neutral_bias", "Mixed regional signal")
    if direction == "LONG":
        return bullish
    if direction == "SHORT":
        return bearish
    return neutral


def _persistence_state_file(output_base: str) -> Path:
    return Path(output_base) / "signal_persistence_state.json"


def load_persistence_state(output_base: str) -> Dict:
    state_file = _persistence_state_file(output_base)
    if not state_file.exists():
        return {}
    return json.loads(state_file.read_text())


def save_persistence_state(output_base: str, state: Dict) -> None:
    state_file = _persistence_state_file(output_base)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2))


def _signal_vote(signal: Dict) -> float:
    action = signal.get("trading_action", "FLAT")
    direction = 1.0 if action == "LONG" else -1.0 if action == "SHORT" else 0.0
    confidence_label = _normalize_confidence(signal.get("confidence", "Low"))
    confidence = CONFIDENCE_WEIGHTS.get(confidence_label, 0.25)
    return direction * confidence


def apply_signal_persistence(raw_signal: Dict, previous_state: Dict, confirmations_required: int = DEFAULT_CONFIRMATIONS) -> tuple[Dict, Dict]:
    raw_action = raw_signal.get("trading_action", "FLAT")
    live_action = previous_state.get("live_action", "FLAT")
    pending_action = previous_state.get("pending_action", raw_action)
    pending_count = int(previous_state.get("pending_count", 0))

    if raw_action == live_action:
        pending_action = raw_action
        pending_count = 0
        applied_action = live_action
    elif raw_action == "FLAT":
        pending_action = raw_action
        pending_count = pending_count + 1 if pending_action == raw_action else 1
        applied_action = "FLAT" if pending_count >= confirmations_required else live_action
    else:
        if pending_action == raw_action:
            pending_count += 1
        else:
            pending_action = raw_action
            pending_count = 1
        applied_action = raw_action if pending_count >= confirmations_required else live_action

    persisted = dict(raw_signal)
    if applied_action != raw_action:
        persisted["signal"] = persisted.get("signal", "Signal") + " (pending confirmation)"
        persisted["actionability"] = "Ignore" if applied_action == "FLAT" else persisted.get("actionability", "Ignore")
        persisted["confidence"] = "Low"
    persisted["raw_trading_action"] = raw_action
    persisted["trading_action"] = applied_action
    persisted["pending_action"] = pending_action
    persisted["pending_count"] = pending_count
    persisted["confirmations_required"] = confirmations_required

    next_state = {
        "live_action": applied_action,
        "pending_action": pending_action,
        "pending_count": pending_count,
    }
    return persisted, next_state


def build_meta_signals(signals: Dict, region_configs: Dict, meta_groups: Dict, persistence_state: Dict) -> tuple[Dict, Dict]:
    grouped_regions = {}
    for region_id, config in region_configs.items():
        group = config.get("meta_group")
        if not group:
            continue
        grouped_regions.setdefault(group, []).append((region_id, config))

    meta_signals = {}
    next_meta_state = {}
    for group, members in grouped_regions.items():
        group_config = meta_groups.get(group, {})
        weighted_votes = []
        constituents = []

        for region_id, config in members:
            signal = signals.get(region_id)
            if not signal:
                continue
            weight = float(config.get("meta_weight", 1.0))
            vote = _signal_vote(signal)
            weighted_votes.append((weight, vote))
            constituents.append({
                "region": region_id,
                "weight": weight,
                "action": signal.get("trading_action", "FLAT"),
                "confidence": _normalize_confidence(signal.get("confidence", "Low")),
                "signal": signal.get("signal", "No data"),
            })

        if not weighted_votes:
            continue

        total_weight = sum(weight for weight, _ in weighted_votes)
        vote_score = sum(weight * vote for weight, vote in weighted_votes) / total_weight if total_weight else 0.0

        if vote_score >= 0.2:
            raw_action = "LONG"
            actionability = "Actionable"
        elif vote_score <= -0.2:
            raw_action = "SHORT"
            actionability = "Actionable"
        else:
            raw_action = "FLAT"
            actionability = "Ignore"

        abs_score = abs(vote_score)
        if abs_score >= 0.6:
            confidence = "High"
        elif abs_score >= 0.3:
            confidence = "Medium"
        else:
            confidence = "Low"

        if confidence == "Low":
            actionability = "Ignore"

        raw_signal = {
            "signal": _meta_signal_text(group_config, raw_action),
            "confidence": confidence,
            "actionability": actionability,
            "trading_action": raw_action,
            "type": group_config.get("type", "meta_signal"),
            "instruments": group_config.get("instruments", []),
            "bias": _meta_bias(group_config, raw_action),
            "meta_group": group,
            "vote_score": vote_score,
            "constituents": constituents,
            "portfolio_trade": group_config.get("portfolio_trade", True),
        }
        persisted_signal, persisted_state = apply_signal_persistence(
            raw_signal,
            persistence_state.get(f"meta:{group}", {}),
            confirmations_required=int(group_config.get("confirmations_required", DEFAULT_CONFIRMATIONS)),
        )
        meta_signals[f"{group}_meta"] = persisted_signal
        next_meta_state[f"meta:{group}"] = persisted_state

    return meta_signals, next_meta_state


def _extract_signal_frame(region_type: str, detection: dict, region_id: str, output_base: str) -> pd.DataFrame:
    live_frame = pd.DataFrame()
    metadata = detection.get("metadata", {}) if isinstance(detection, dict) else {}
    details = detection.get("details", []) if isinstance(detection, dict) else []
    live_detection_ok = metadata.get("status", "success") == "success"
    valid_pixels = details[0].get("valid_pixels", 0) if details else 0

    # Handle oil_storage type specially
    if region_type == "oil_storage":
        if details and metadata.get("status") == "success":
            row = dict(details[0])
            row.setdefault("date", detection.get("date"))
            live_frame = pd.DataFrame([row])

        backfill_file = Path(output_base) / "backfill" / f"{region_id}_backfill.json"
        if backfill_file.exists():
            history = json.loads(backfill_file.read_text())
            stats = history.get("daily_stats") or history.get("weekly_stats") or []
            frame = pd.DataFrame(stats)
            if not frame.empty and 'date' in frame.columns:
                if not live_frame.empty and 'date' in live_frame.columns:
                    merged = pd.concat([frame, live_frame], ignore_index=True, sort=False)
                    merged = merged.sort_values('date').drop_duplicates(subset=['date'], keep='last')
                    return merged
                return frame.sort_values('date')
        
        return live_frame

    if details and (region_type not in {"agriculture", "agricultural"} or (live_detection_ok and valid_pixels > 0)):
        row = dict(details[0])
        row.setdefault("date", detection.get("date"))
        live_frame = pd.DataFrame([row])

    backfill_file = Path(output_base) / "backfill" / f"{region_id}_backfill.json"
    if backfill_file.exists():
        history = json.loads(backfill_file.read_text())
        stats = history.get("daily_stats") or history.get("weekly_stats") or []
        if not stats:
            logger.debug(f"Empty backfill stats for {region_id}")
        frame = pd.DataFrame(stats)
        if not frame.empty and 'date' in frame.columns:
            if not live_frame.empty and 'date' in live_frame.columns:
                merged = pd.concat([frame, live_frame], ignore_index=True, sort=False)
                merged = merged.sort_values('date').drop_duplicates(subset=['date'], keep='last')
                return merged
            return frame.sort_values('date')

    if not live_frame.empty:
        return live_frame

    count = detection.get("count") if isinstance(detection, dict) else None
    target_date = detection.get("date") if isinstance(detection, dict) else None
    if region_type in {"chokepoint", "port_logistics"} and count is not None and target_date:
        return pd.DataFrame([{"date": target_date, "detections": count}])

    return pd.DataFrame()


def process_region(
    region_id: str,
    region_config: dict,
    target_date: str,
    output_base: str = "outputs",
) -> Dict:
    """
    Process a single region for a given date.
    
    Args:
        region_id: Region identifier
        region_config: Region configuration from registry
        target_date: Date to process
        output_base: Output directory
    
    Returns:
        Processing result with detections and signal
    """
    region_type = region_config.get("type")
    aoi_file = region_config.get("aoi_file")

    if not isinstance(region_type, str):
        return {
            "region": region_id,
            "date": target_date,
            "status": "error",
            "message": f"Unknown monitoring type for {region_id}",
        }
    
    if not aoi_file or not Path(aoi_file).exists():
        return {
            "region": region_id,
            "date": target_date,
            "status": "error",
            "message": f"AOI file not found: {aoi_file}",
        }
    
    try:
        # Run detection
        detection_result = run_detection(
            monitoring_type=region_type,
            aoi_path=aoi_file,
            target_date=target_date,
            output_base=output_base,
        )
        
        return {
            "region": region_id,
            "name": region_config.get("name", region_id),
            "type": region_type,
            "date": target_date,
            "status": "success",
            "detection": detection_result.to_dict() if hasattr(detection_result, 'to_dict') else detection_result,
            "instruments": region_config.get("instruments", []),
        }
    
    except Exception as e:
        return {
            "region": region_id,
            "date": target_date,
            "status": "error",
            "message": str(e),
        }


def run_daily_pipeline(
    target_date: Optional[str] = None,
    output_base: str = "outputs",
    regions_filter: Optional[List[str]] = None,
) -> Dict:
    """
    Run daily pipeline for all active regions.
    
    Args:
        target_date: Date to process (default: today)
        output_base: Output directory
        regions_filter: Optional list of region IDs to process
    
    Returns:
        Summary of all processing results
    """
    if target_date is None:
        target_date = date.today().isoformat()

    registry = load_registry()
    active_regions = get_active_regions()
    meta_groups = registry.get("meta_groups", {})
    persistence_state = load_persistence_state(output_base)
    next_persistence_state = {
        key: value for key, value in persistence_state.items()
        if key.startswith("meta:")
    }
    
    if regions_filter:
        active_regions = {
            rid: rconfig for rid, rconfig in active_regions.items()
            if rid in regions_filter
        }
    
    print(f"Processing {len(active_regions)} active regions for {target_date}")
    print("=" * 60)
    
    results = []
    signals = {}
    
    for region_id, region_config in active_regions.items():
        print(f"\n[{region_config.get('name', region_id)}]")
        
        try:
            result = process_region(
                region_id=region_id,
                region_config=region_config,
                target_date=target_date,
                output_base=output_base,
            )
        except Exception as e:
            logger.error("Failed to process region %s: %s", region_id, e)
            result = {"status": "error", "message": str(e), "detection": {}}
        
        results.append(result)
        
        signal_payload = {
            "signal": "No data",
            "confidence": "Low",
            "actionability": "Ignore",
            "trading_action": "FLAT",
            "type": region_config.get("type"),
            "instruments": region_config.get("instruments", []),
        }

        try:
            if result["status"] == "success":
                frame = _extract_signal_frame(
                    region_config.get("type"),
                    result.get("detection", {}),
                    region_id,
                    output_base,
                )
                if not frame.empty:
                    generated_signal = generate_signal(region_config.get("type"), frame)
                    signal_payload.update(generated_signal)
        except Exception as e:
            logger.warning("Failed to generate signal for %s: %s", region_id, e)

        persisted_signal, persisted_state = apply_signal_persistence(
            signal_payload,
            persistence_state.get(f"region:{region_id}", {}),
            confirmations_required=int(region_config.get("confirmations_required", DEFAULT_CONFIRMATIONS)),
        )
        signals[region_id] = persisted_signal
        next_persistence_state[f"region:{region_id}"] = persisted_state

        status = "✅" if result["status"] == "success" else "❌"
        print(f"  Status: {status} {result['status']}")

    try:
        from pipeline.agriculture_signal import build_agriculture_signals

        agriculture_signals = build_agriculture_signals(target_date, output_base=output_base)
        for region_id, signal_payload in agriculture_signals.items():
            persisted_signal, persisted_state = apply_signal_persistence(
                signal_payload,
                persistence_state.get(f"region:{region_id}", {}),
                confirmations_required=int(signal_payload.get("confirmations_required", DEFAULT_CONFIRMATIONS)),
            )
            signals[region_id] = persisted_signal
            next_persistence_state[f"region:{region_id}"] = persisted_state
    except Exception as e:
        print(f"Warning: agriculture combined signals failed: {e}")

    meta_signals, next_meta_state = build_meta_signals(
        signals,
        active_regions,
        meta_groups,
        persistence_state,
    )
    signals.update(meta_signals)
    next_persistence_state.update(next_meta_state)
    save_persistence_state(output_base, next_persistence_state)
    
    # Build summary
    summary = {
        "date": target_date,
        "regions_processed": len(results),
        "regions_successful": sum(1 for r in results if r["status"] == "success"),
        "regions_failed": sum(1 for r in results if r["status"] != "success"),
        "signals": signals,
        "results": results,
    }
    
    # Save summary
    output_path = Path(output_base) / target_date
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_path / "daily_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, default=str))
    
    print(f"\n{'=' * 60}")
    print(f"Summary saved to: {summary_file}")
    print(f"Regions processed: {summary['regions_processed']}")
    print(f"Successful: {summary['regions_successful']}")
    
    # Track total assets for equity curve using mark-to-market valuation
    from pipeline.asset_tracker import record_daily_assets
    from pipeline.price_feed import get_prices_for_portfolio

    try:
        portfolio = MultiAssetPortfolio(output_base=output_base)
        prices = get_prices_for_portfolio(portfolio)
        total_value = portfolio.get_total_value(prices)
        record_daily_assets(total_value, target_date, output_base=output_base)
    except Exception as e:
        logger.warning(f"Failed to track daily assets: {e}")

    try:
        from pipeline.wiki_ingest import ingest_daily_summary
        ingest_daily_summary(target_date, summary, output_base=output_base)
    except Exception as e:
        print(f"  Warning: wiki ingest failed: {e}")
    
    return summary


def update_portfolio_with_signals(
    portfolio: MultiAssetPortfolio,
    signals: Dict,
    prices: Dict[str, float],
) -> List[Dict]:
    """
    Update portfolio based on signals.
    
    Args:
        portfolio: MultiAssetPortfolio instance
        signals: Dictionary of signals by region
        prices: Current prices for instruments
    
    Returns:
        Trading actions taken
    """
    actions = []
    
    for region_id, signal in signals.items():
        if signal.get("portfolio_trade") is False:
            continue
        if signal.get("actionability") != "Actionable":
            continue
        
        instruments = signal.get("instruments", [])
        trading_action = signal.get("trading_action", "FLAT")
        
        for instrument in instruments:
            ticker = instrument if isinstance(instrument, str) else instrument.get("ticker")
            
            if ticker not in prices:
                continue
            
            # Check if we should trade
            if trading_action == "LONG":
                # Check if already long
                if ticker not in portfolio.positions or portfolio.positions[ticker].direction != "long":
                    action = {
                        "ticker": ticker,
                        "action": "OPEN_LONG",
                        "price": prices[ticker],
                        "signal": signal["signal"],
                        "region": region_id,
                    }
                    actions.append(action)
            
            elif trading_action == "SHORT":
                # Check if already short
                if ticker not in portfolio.positions or portfolio.positions[ticker].direction != "short":
                    action = {
                        "ticker": ticker,
                        "action": "OPEN_SHORT",
                        "price": prices[ticker],
                        "signal": signal["signal"],
                        "region": region_id,
                    }
                    actions.append(action)
            
            elif trading_action == "FLAT":
                # Close position if exists
                if ticker in portfolio.positions:
                    action = {
                        "ticker": ticker,
                        "action": "CLOSE",
                        "price": prices[ticker],
                        "signal": signal["signal"],
                        "region": region_id,
                    }
                    actions.append(action)
    
    return actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run daily QuantTrade pipeline")
    parser.add_argument("--date", default=None, help="Date to process (YYYY-MM-DD)")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--regions", nargs="*", help="Specific regions to process")
    args = parser.parse_args()
    
    summary = run_daily_pipeline(
        target_date=args.date,
        output_base=args.output,
        regions_filter=args.regions,
    )
    
    actionable = sum(1 for signal in summary["signals"].values() if signal.get("actionability") == "Actionable")
    print(f"\nPipeline complete. {summary['regions_successful']}/{summary['regions_processed']} regions processed successfully.")
    print(f"Actionable signals: {actionable}")
