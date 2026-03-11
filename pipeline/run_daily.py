"""
Unified daily pipeline runner for all monitoring types.

Processes all active regions and generates signals for the multi-asset portfolio.
"""

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Optional
import pandas as pd

from pipeline.regions import get_active_regions
from pipeline.detection_multi import run_detection
from pipeline.signals_multi import generate_signal
from paper_trading.multi_asset_portfolio import MultiAssetPortfolio


CONFIDENCE_WEIGHTS = {
    "High": 1.0,
    "Medium": 0.6,
    "Low": 0.25,
}


def _signal_vote(signal: Dict) -> float:
    action = signal.get("trading_action", "FLAT")
    direction = 1.0 if action == "LONG" else -1.0 if action == "SHORT" else 0.0
    confidence = CONFIDENCE_WEIGHTS.get(signal.get("confidence", "Low"), 0.25)
    return direction * confidence


def build_meta_signals(signals: Dict, region_configs: Dict) -> Dict:
    grouped_regions = {}
    for region_id, config in region_configs.items():
        group = config.get("meta_group")
        if not group:
            continue
        grouped_regions.setdefault(group, []).append((region_id, config))

    meta_signals = {}
    for group, members in grouped_regions.items():
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
                "confidence": signal.get("confidence", "Low"),
                "signal": signal.get("signal", "No data"),
            })

        if not weighted_votes:
            continue

        total_weight = sum(weight for weight, _ in weighted_votes)
        vote_score = sum(weight * vote for weight, vote in weighted_votes) / total_weight if total_weight else 0.0

        if vote_score >= 0.2:
            trading_action = "LONG"
            signal_text = "Brazil soy meta-long"
            bias = "Bullish soybean prices"
            actionability = "Actionable"
        elif vote_score <= -0.2:
            trading_action = "SHORT"
            signal_text = "Brazil soy meta-short"
            bias = "Bearish soybean prices"
            actionability = "Actionable"
        else:
            trading_action = "FLAT"
            signal_text = "Brazil soy meta-neutral"
            bias = "Mixed regional soybean signal"
            actionability = "Ignore"

        abs_score = abs(vote_score)
        if abs_score >= 0.6:
            confidence = "High"
        elif abs_score >= 0.3:
            confidence = "Medium"
        else:
            confidence = "Low"

        meta_signals[f"{group}_meta"] = {
            "signal": signal_text,
            "confidence": confidence,
            "actionability": actionability,
            "trading_action": trading_action,
            "type": "meta_agriculture",
            "instruments": ["Soybeans"],
            "bias": bias,
            "meta_group": group,
            "vote_score": vote_score,
            "constituents": constituents,
        }

    return meta_signals


def _extract_signal_frame(region_type: str, detection: dict, region_id: str, output_base: str) -> pd.DataFrame:
    live_frame = pd.DataFrame()
    metadata = detection.get("metadata", {}) if isinstance(detection, dict) else {}
    details = detection.get("details", []) if isinstance(detection, dict) else []
    live_detection_ok = metadata.get("status", "success") == "success"
    valid_pixels = details[0].get("valid_pixels", 0) if details else 0

    if details and (region_type not in {"agriculture", "agricultural"} or (live_detection_ok and valid_pixels > 0)):
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
    
    active_regions = get_active_regions()
    
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
        
        result = process_region(
            region_id=region_id,
            region_config=region_config,
            target_date=target_date,
            output_base=output_base,
        )
        
        results.append(result)
        
        signal_payload = {
            "signal": "No data",
            "confidence": "Low",
            "actionability": "Ignore",
            "trading_action": "FLAT",
            "type": region_config.get("type"),
            "instruments": region_config.get("instruments", []),
        }

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

        signals[region_id] = signal_payload
        
        status = "✅" if result["status"] == "success" else "❌"
        print(f"  Status: {status} {result['status']}")
    
    # Save summary
    signals.update(build_meta_signals(signals, active_regions))

    summary = {
        "date": target_date,
        "regions_processed": len(results),
        "regions_successful": len([r for r in results if r["status"] == "success"]),
        "results": results,
        "signals": signals,
        "generated_at": datetime.now().isoformat(),
    }
    
    output_path = Path(output_base) / target_date
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_path / "daily_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, default=str))
    
    print(f"\n{'=' * 60}")
    print(f"Summary saved to: {summary_file}")
    print(f"Regions processed: {len(results)}")
    print(f"Successful: {summary['regions_successful']}")
    
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
