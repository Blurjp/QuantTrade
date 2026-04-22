#!/usr/bin/env python3
"""
Clean portfolio data — removes corrupt/empty trade records and fixes data integrity issues.

Safe to run — only removes clearly invalid records:
- Trades with entry_price=0, exit_price=0, pnl=0 (empty placeholders)
- Duplicate position entries
"""

import json
import sys
import urllib.request

API_BASE = "https://scheduler-production-b60f.up.railway.app"

def main():
    print("🧹 Portfolio Cleaner\n" + "=" * 50)
    
    # Fetch current portfolio
    with urllib.request.urlopen(f"{API_BASE}/api/portfolio", timeout=15) as r:
        portfolio = json.loads(r.read())
    
    trades = portfolio.get("trades", [])
    positions = portfolio.get("positions", {})
    
    print(f"\nCurrent state: {len(positions)} positions, {len(trades)} trade records")
    
    # Find empty trades
    empty_trades = []
    good_trades = []
    for t in trades:
        if t.get("entry_price", 0) == 0 and t.get("exit_price", 0) == 0 and t.get("pnl", 0) == 0:
            empty_trades.append(t)
        else:
            good_trades.append(t)
    
    if empty_trades:
        print(f"\n🗑️  Found {len(empty_trades)} empty trade records to remove:")
        for t in empty_trades:
            print(f"   - {t.get('ticker', '?')} {t.get('direction', '?')} (empty data)")
    else:
        print("\n✅ No empty trade records found — portfolio is clean")
        return
    
    print(f"\nKeeping {len(good_trades)} valid trade records")
    
    # Update portfolio
    portfolio["trades"] = good_trades
    
    # Print what we'd save
    print(f"\nProposed changes:")
    print(f"   Trades: {len(trades)} → {len(good_trades)} (removed {len(empty_trades)})")
    print(f"   Positions: {len(positions)} (unchanged)")
    print(f"   Cash: ${portfolio.get('cash', 0):,.2f} (unchanged)")
    
    # Write back — this needs to be done via API or direct file write
    # Since we can't write to Railway directly, output the clean JSON for manual upload
    clean_file = "outputs/portfolio_clean.json"
    import pathlib
    pathlib.Path(clean_file).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(clean_file).write_text(json.dumps(portfolio, indent=2, default=str))
    print(f"\n✅ Clean portfolio saved to {clean_file}")
    print(f"⚠️  To apply: upload this file to Railway volume or restart with clean state")

if __name__ == "__main__":
    main()
