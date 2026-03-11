"""
Unified daily pipeline runner for all monitoring types.

Processes all active regions and generates signals for the multi-asset portfolio.
"""

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Optional

from pipeline.regions import load_registry, get_active_regions
from pipeline.detection_multi import run_detection
from pipeline.signals_multi import generate_signal
from paper_trading.multi_asset_portfolio import MultiAssetPortfolio


def process_region(
    region_id: str,
    region_config: dict,
    target_date: str,
    output_base: str = "outputs",
) -> dict:
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
    target_date: str = None,
    output_base: str = "outputs",
    regions_filter: List[str] = None,
) -> dict:
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
    
    # Load registry
    registry_path = Path("configs/regions/registry_v2.json")
    if not registry_path.exists():
        registry_path = Path("configs/regions/registry.json")
    
    with open(registry_path) as f:
        registry = json.load(f)
    
    regions = registry.get("regions", {})
    
    # Filter to active regions
    active_regions = {
        rid: rconfig for rid, rconfig in regions.items()
        if rconfig.get("active", False)
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
        
        result = process_region(
            region_id=region_id,
            region_config=region_config,
            target_date=target_date,
            output_base=output_base,
        )
        
        results.append(result)
        
        # Generate signal placeholder (would use actual data in production)
        signals[region_id] = {
            "signal": "Pending data",
            "confidence": "Low",
            "actionability": "Ignore",
            "type": region_config.get("type"),
            "instruments": region_config.get("instruments", []),
        }
        
        status = "✅" if result["status"] == "success" else "❌"
        print(f"  Status: {status} {result['status']}")
    
    # Save summary
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
) -> dict:
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
    
    print(f"\nPipeline complete. {summary['regions_successful']}/{summary['regions_processed']} regions processed successfully.")
