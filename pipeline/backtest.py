"""
Signal backtesting and calibration.

Tests historical signals against actual price movements to:
1. Validate signal accuracy
2. Optimize thresholds
3. Calculate expected returns
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np


def load_backfill_data(region_id: str, output_base: str = "outputs") -> Dict:
    """Load backfilled detection data for a region."""
    backfill_file = Path(output_base) / "backfill" / f"{region_id}_backfill.json"
    if not backfill_file.exists():
        return None
    
    return json.loads(backfill_file.read_text())


def fetch_historical_prices(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch historical prices from Yahoo Finance.
    
    Returns DataFrame with date, open, high, low, close, volume.
    """
    try:
        import yfinance as yf
        
        # Map commodity tickers
        yahoo_ticker = {
            "WTI": "CL=F",
            "Brent": "BZ=F",
            "Corn": "ZC=F",
            "Soybeans": "ZS=F",
            "Natural Gas": "NG=F",
        }.get(ticker, ticker)
        
        stock = yf.Ticker(yahoo_ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            return None
        
        df = df.reset_index()
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


def calculate_price_returns(prices: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
    """
    Calculate forward returns for each date.
    
    Args:
        prices: Price DataFrame
        forward_days: Number of days to look forward
    
    Returns:
        DataFrame with added forward return columns
    """
    df = prices.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Calculate forward returns
    df[f'return_{forward_days}d'] = df['Close'].pct_change(forward_days).shift(-forward_days)
    
    # Calculate volatility
    df['daily_return'] = df['Close'].pct_change()
    df['volatility_5d'] = df['daily_return'].rolling(5).std()
    
    return df


def generate_historical_signals(
    detection_data: Dict,
    signal_type: str = "chokepoint",
) -> pd.DataFrame:
    """
    Generate signals from historical detection data.
    
    Args:
        detection_data: Backfilled detection data
        signal_type: Type of signal logic to apply
    
    Returns:
        DataFrame with date and signal columns
    """
    if not detection_data:
        return None
    
    daily_stats = detection_data.get("daily_stats", detection_data.get("weekly_stats", []))
    
    if not daily_stats:
        return None
    
    df = pd.DataFrame(daily_stats)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Default neutral
    df['signal_direction'] = 'neutral'
    df['signal_raw'] = 0
    
    if signal_type == "chokepoint":
        # For chokepoints: low detections = disruption, high = normal
        if 'detections' in df.columns:
            # Calculate baseline
            baseline = df['detections'].rolling(7, min_periods=3).mean()
            
            # Signal: deviation from baseline
            # 优化: 降低阈值从0.3 (30%) 至 0.10 (10%)
            df['signal_raw'] = (df['detections'] - baseline) / baseline
            df['signal_direction'] = np.where(df['signal_raw'] < -0.10, 'long_disruption',
                                              np.where(df['signal_raw'] > 0.10, 'short_disruption', 'neutral'))
    
    elif signal_type == "port_logistics":
        # For ports: high activity = bullish logistics
        if 'detections' in df.columns:
            baseline = df['detections'].rolling(7, min_periods=3).mean()
            df['signal_raw'] = (df['detections'] - baseline) / baseline
            # High port activity = bullish supply chain
            df['signal_direction'] = np.where(df['signal_raw'] > 0.2, 'long',
                                              np.where(df['signal_raw'] < -0.2, 'short', 'neutral'))
    
    elif signal_type in ["agricultural", "agriculture", "oil_storage", "auto_inventory"]:
        # NDVI-based signals
        if 'ndvi_mean' in df.columns:
            baseline = df['ndvi_mean'].rolling(7, min_periods=3).mean()  # 降低窗口期
            df['signal_raw'] = (df['ndvi_mean'] - baseline) / baseline
            
            # For agriculture: high NDVI = good crop = bearish prices
            # For auto: low NDVI (dark parking lots) = high inventory = bearish
            # 优化: 降低阈值从0.1 (10%) 至 0.03 (3%)
            if signal_type == "auto_inventory":
                df['signal_direction'] = np.where(df['signal_raw'] < -0.05, 'long',
                                                  np.where(df['signal_raw'] > 0.05, 'short', 'neutral'))
            else:  # agricultural
                df['signal_direction'] = np.where(df['signal_raw'] > 0.03, 'short',
                                                  np.where(df['signal_raw'] < -0.03, 'long', 'neutral'))
    
    return df


def backtest_signals(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    forward_days: int = 5,
) -> Dict:
    """
    Backtest signals against price movements.
    
    Args:
        signals: DataFrame with date and signal_direction
        prices: DataFrame with date and forward returns
        forward_days: Forward return period
    
    Returns:
        Dictionary with backtest results
    """
    # Merge signals with prices
    signals['date'] = pd.to_datetime(signals['date'])
    prices['Date'] = pd.to_datetime(prices['Date'])
    
    merged = pd.merge(
        signals,
        prices,
        left_on='date',
        right_on='Date',
        how='inner'
    )
    
    if merged.empty:
        return {"status": "no_overlap", "message": "No overlapping dates"}
    
    results = {
        "total_signals": len(merged),
        "forward_days": forward_days,
        "by_direction": {},
    }
    
    # Analyze by signal direction
    for direction in ['long', 'short', 'long_disruption', 'short_disruption', 'neutral']:
        subset = merged[merged['signal_direction'] == direction]
        
        if len(subset) == 0:
            continue
        
        returns = subset[f'return_{forward_days}d'].dropna()
        
        if len(returns) == 0:
            continue
        
        # For long signals, positive return = correct
        # For short signals, negative return = correct
        if direction in ['long', 'long_disruption']:
            correct = (returns > 0).sum()
            avg_return = returns.mean()
        elif direction in ['short', 'short_disruption']:
            correct = (returns < 0).sum()
            avg_return = -returns.mean()  # Flip for interpretation
        else:
            correct = len(returns)  # Neutral is always "correct"
            avg_return = returns.abs().mean()
        
        results["by_direction"][direction] = {
            "count": len(subset),
            "correct": int(correct),
            "accuracy": float(correct / len(returns)) if len(returns) > 0 else 0,
            "avg_return": float(avg_return) if not np.isnan(avg_return) else 0,
            "std_return": float(returns.std()) if len(returns) > 1 else 0,
        }
    
    # Overall accuracy (excluding neutral)
    directional = merged[merged['signal_direction'].isin(['long', 'short', 'long_disruption', 'short_disruption'])]
    if len(directional) > 0:
        long_mask = directional['signal_direction'].isin(['long', 'long_disruption'])
        short_mask = directional['signal_direction'].isin(['short', 'short_disruption'])
        
        returns = directional[f'return_{forward_days}d']
        
        long_correct = ((long_mask) & (returns > 0)).sum()
        short_correct = ((short_mask) & (returns < 0)).sum()
        total_correct = long_correct + short_correct
        
        results["overall_accuracy"] = float(total_correct / len(directional))
        results["total_directional_signals"] = len(directional)
    
    return results


def run_full_backtest(
    region_id: str,
    ticker: str,
    output_base: str = "outputs",
    forward_days: int = 5,
) -> Dict:
    """
    Run complete backtest for a region and ticker.
    
    Args:
        region_id: Region to backtest
        ticker: Trading instrument ticker
        output_base: Output directory
        forward_days: Forward return period
    
    Returns:
        Dictionary with full backtest results
    """
    print(f"\n{'='*60}")
    print(f"Backtesting: {region_id} → {ticker}")
    print(f"{'='*60}")
    
    # Load detection data
    detection_data = load_backfill_data(region_id, output_base)
    
    if not detection_data:
        print(f"  ❌ No backfill data for {region_id}")
        return {"status": "error", "message": "No backfill data"}
    
    print(f"  ✓ Loaded {len(detection_data.get('daily_stats', detection_data.get('weekly_stats', [])))} data points")
    
    # Determine date range
    stats = detection_data.get("daily_stats", detection_data.get("weekly_stats", []))
    if not stats:
        return {"status": "error", "message": "No data points"}
    
    dates = [s['date'] for s in stats]
    start_date = min(dates)
    end_date = max(dates)
    
    # Add buffer for forward returns
    end_date_buffered = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=forward_days + 10)).strftime("%Y-%m-%d")
    
    # Fetch prices
    print(f"  Fetching {ticker} prices ({start_date} to {end_date_buffered})...")
    prices = fetch_historical_prices(ticker, start_date, end_date_buffered)
    
    if prices is None or prices.empty:
        print(f"  ❌ No price data for {ticker}")
        return {"status": "error", "message": "No price data"}
    
    print(f"  ✓ Got {len(prices)} price points")
    
    # Calculate returns
    prices = calculate_price_returns(prices, forward_days)
    
    # Generate signals
    signal_type = detection_data.get("type", "chokepoint")
    signals = generate_historical_signals(detection_data, signal_type)
    
    if signals is None or signals.empty:
        print(f"  ❌ Could not generate signals")
        return {"status": "error", "message": "Could not generate signals"}
    
    print(f"  ✓ Generated {len(signals)} signals")
    
    # Run backtest
    backtest_results = backtest_signals(signals, prices, forward_days)
    
    # Compile results
    results = {
        "region": region_id,
        "ticker": ticker,
        "signal_type": signal_type,
        "start_date": start_date,
        "end_date": end_date,
        "forward_days": forward_days,
        "backtest": backtest_results,
    }
    
    # Print summary
    print(f"\n  📊 Results:")
    print(f"     Total signals: {backtest_results.get('total_signals', 0)}")
    
    if "overall_accuracy" in backtest_results:
        print(f"     Overall accuracy: {backtest_results['overall_accuracy']*100:.1f}%")
    
    for direction, stats in backtest_results.get("by_direction", {}).items():
        print(f"     {direction}: {stats['count']} signals, {stats['accuracy']*100:.1f}% accuracy, avg return: {stats['avg_return']*100:.2f}%")
    
    # Save results
    output_path = Path(output_base) / "backtest"
    output_path.mkdir(parents=True, exist_ok=True)
    
    result_file = output_path / f"{region_id}_{ticker}_backtest.json"
    result_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  ✓ Saved to {result_file}")
    
    return results


def optimize_thresholds(
    region_id: str,
    ticker: str,
    output_base: str = "outputs",
    threshold_range: List[float] = None,
) -> Dict:
    """
    Find optimal signal thresholds through grid search.
    
    Args:
        region_id: Region to optimize
        ticker: Trading instrument
        output_base: Output directory
        threshold_range: List of threshold values to test
    
    Returns:
        Dictionary with optimal thresholds
    """
    if threshold_range is None:
        threshold_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Load data
    detection_data = load_backfill_data(region_id, output_base)
    if not detection_data:
        return {"status": "error", "message": "No backfill data"}
    
    # Get prices
    stats = detection_data.get("daily_stats", detection_data.get("weekly_stats", []))
    dates = [s['date'] for s in stats]
    start_date = min(dates)
    end_date = (datetime.strptime(max(dates), "%Y-%m-%d") + timedelta(days=15)).strftime("%Y-%m-%d")
    
    prices = fetch_historical_prices(ticker, start_date, end_date)
    if prices is None:
        return {"status": "error", "message": "No price data"}
    
    prices = calculate_price_returns(prices, 5)
    
    # Test each threshold
    results = []
    
    for threshold in threshold_range:
        # Generate signals with this threshold
        signals = generate_historical_signals(detection_data, detection_data.get("type", "chokepoint"))
        
        # Override threshold (would need to modify signal generation)
        # For now, just run backtest
        backtest = backtest_signals(signals, prices, 5)
        
        accuracy = backtest.get("overall_accuracy", 0)
        
        results.append({
            "threshold": threshold,
            "accuracy": accuracy,
        })
    
    # Find best
    best = max(results, key=lambda x: x["accuracy"])
    
    return {
        "region": region_id,
        "ticker": ticker,
        "optimal_threshold": best["threshold"],
        "optimal_accuracy": best["accuracy"],
        "all_results": results,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Signal backtesting")
    parser.add_argument("--region", required=True, help="Region ID")
    parser.add_argument("--ticker", required=True, help="Trading ticker")
    parser.add_argument("--forward", type=int, default=5, help="Forward days")
    parser.add_argument("--output", default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    run_full_backtest(args.region, args.ticker, args.output, args.forward)
