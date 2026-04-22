#!/usr/bin/env python3
"""
Update unrealized P&L for all positions using latest prices.
Should run periodically during market hours.
"""

import json
import sys
import urllib.request

API_BASE = "https://scheduler-production-b60f.up.railway.app"

def main():
    # Get portfolio
    with urllib.request.urlopen(f"{API_BASE}/api/portfolio") as r:
        portfolio = json.loads(r.read())
    
    positions = portfolio.get("positions", {})
    if not positions:
        print("No positions to update.")
        return
    
    tickers = list(positions.keys())
    print(f"Updating P&L for {len(tickers)} positions...")
    
    # Fetch prices via yfinance
    import yfinance as yf
    data = yf.download(tickers, period="1d", progress=False)
    
    if data.empty:
        print("No price data available (market closed or API error)")
        return
    
    close = data["Close"]
    updated = 0
    for ticker in tickers:
        pos = positions[ticker]
        entry = pos["entry_price"]
        direction = pos["direction"]
        qty = pos.get("quantity", pos.get("position_value", 0) / entry) if entry else 0
        
        if len(tickers) == 1:
            current = float(close.iloc[-1])
        elif ticker in close.columns:
            series = close[ticker].dropna()
            if len(series) == 0:
                continue
            current = float(series.iloc[-1])
        else:
            continue
        
        if direction == "long":
            pnl = (current - entry) * qty
        else:
            pnl = (entry - current) * qty
        
        pnl_pct = ((current - entry) / entry * 100) if direction == "long" else ((entry - current) / entry * 100)
        
        pos["unrealized_pnl"] = round(pnl, 2)
        pos["current_price"] = round(current, 2)
        updated += 1
        
        emoji = "🟢" if pnl >= 0 else "🔴"
        print(f"  {emoji} {ticker:6} ${current:>8.2f}  P&L=${pnl:>+,.2f} ({pnl_pct:+.1f}%)")
    
    # Summary
    total_pnl = sum(p.get("unrealized_pnl", 0) for p in positions.values())
    total_val = sum(p.get("position_value", 0) for p in positions.values())
    cash = portfolio.get("cash", 0)
    
    print(f"\n💰 Cash:       ${cash:>10,.2f}")
    print(f"📈 Holdings:   ${total_val:>10,.2f}")
    print(f"📊 Total:      ${cash + total_val:>10,.2f}")
    print(f"💹 Unrealized: ${total_pnl:>+10,.2f}")
    print(f"\n✅ Updated {updated}/{len(tickers)} positions")

if __name__ == "__main__":
    main()
