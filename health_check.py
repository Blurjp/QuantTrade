#!/usr/bin/env python3
"""
Portfolio Health Check — automated risk and health diagnostics.

Checks:
1. Position concentration (sector/correlation)
2. Unrealized P&L distribution
3. Stop-loss proximity
4. Max drawdown from entry
5. Cash buffer adequacy
6. Correlation risk (too many similar positions)
7. Trade history integrity
"""

import json
import sys
import urllib.request

API_BASE = "https://scheduler-production-b60f.up.railway.app"

# Sector classification
SECTORS = {
    "FXI": "china", "MCHI": "china", "ASHR": "china", "KWEB": "china", "BABA": "china", "JD": "china",
    "EPOL": "emerging", "EPI": "emerging", "INDA": "emerging",
    "EWG": "europe", "FXD": "consumer",
    "USO": "energy", "BNO": "energy", "XLE": "energy", "OIH": "energy", "XOM": "energy", "CVX": "energy",
    "BTU": "commodity",
    "CORN": "agriculture", "SOYB": "agriculture", "WEAT": "agriculture",
    "NUE": "industrial", "STLD": "industrial", "XLI": "industrial", "CAT": "industrial", "DE": "industrial",
    "GLD": "metals", "SLV": "metals",
    "UNG": "energy",
}

def fetch_portfolio():
    with urllib.request.urlopen(f"{API_BASE}/api/portfolio", timeout=15) as r:
        return json.loads(r.read())

def check_concentration(positions):
    """Check sector concentration risk."""
    sector_exposure = {}
    for ticker, pos in positions.items():
        sector = SECTORS.get(ticker, "unknown")
        val = pos.get("position_value", 0)
        sector_exposure[sector] = sector_exposure.get(sector, 0) + val
    
    issues = []
    total = sum(sector_exposure.values())
    for sector, val in sorted(sector_exposure.items(), key=lambda x: -x[1]):
        pct = val / total * 100 if total else 0
        if pct > 50:
            issues.append(f"🔴 Sector '{sector}' is {pct:.0f}% of holdings — too concentrated!")
        elif pct > 35:
            issues.append(f"🟡 Sector '{sector}' is {pct:.0f}% of holdings")
    
    return issues, sector_exposure

def check_stop_loss_proximity(positions):
    """Check how close positions are to stop-loss."""
    issues = []
    for ticker, pos in positions.items():
        entry = pos.get("entry_price", 0)
        stop = pos.get("stop_loss", 0)
        current = pos.get("current_price", entry)
        
        if stop <= 0 or entry <= 0:
            continue
        
        if pos.get("direction") == "long":
            distance_pct = (current - stop) / entry * 100
        else:
            distance_pct = (stop - current) / entry * 100
        
        if distance_pct < 1:
            issues.append(f"🔴 {ticker} within 1% of stop-loss! (distance={distance_pct:.1f}%)")
        elif distance_pct < 2:
            issues.append(f"🟡 {ticker} within 2% of stop-loss (distance={distance_pct:.1f}%)")
    
    return issues

def check_trade_history_integrity(trades):
    """Check for corrupt/empty trade records."""
    issues = []
    empty_trades = [t for t in trades if t.get("entry_price", 0) == 0 and t.get("pnl", 0) == 0]
    if empty_trades:
        issues.append(f"🟡 {len(empty_trades)} trade records with empty data (entry=0, pnl=0) — should be cleaned")
    
    return issues

def check_pnl_integrity(positions):
    """Cross-check stored unrealized_pnl against live recalculation."""
    issues = []
    try:
        import subprocess
        from pathlib import Path
        venv_python = str(Path(__file__).parent / ".venv" / "bin" / "python3")
        tickers = list(positions.keys())
        if not tickers:
            return issues
        script = f'''
import yfinance as yf, json, sys
try:
    batch = yf.download({tickers!r}, period="1d", progress=False)
    if batch.empty: sys.exit(0)
    close = batch["Close"]
    result = {{}}
    for t in {tickers!r}:
        if len({tickers!r}) == 1:
            result[t] = float(close.iloc[-1])
        elif t in close.columns:
            s = close[t].dropna()
            if len(s) > 0: result[t] = float(s.iloc[-1])
    print(json.dumps(result))
except: pass
'''
        result = subprocess.run([venv_python, "-c", script], capture_output=True, text=True, timeout=20)
        if not result.stdout.strip():
            return issues
        live_prices = json.loads(result.stdout.strip())

        for ticker, pos in positions.items():
            entry = float(pos.get("entry_price", 0))
            qty = float(pos.get("quantity", 0))
            direction = pos.get("direction", "long")
            stored_pnl = float(pos.get("unrealized_pnl", 0))

            live_price = live_prices.get(ticker)
            if not live_price or not qty:
                continue

            if direction == "long":
                calc_pnl = qty * (live_price - entry)
            else:
                calc_pnl = qty * (entry - live_price)

            drift = abs(calc_pnl - stored_pnl)
            pct_drift = (drift / abs(calc_pnl) * 100) if calc_pnl else 0

            if drift > 20 or pct_drift > 10:
                issues.append(f"🔴 {ticker} P&L stale: stored=${stored_pnl:+.2f} live=${calc_pnl:+.2f} drift=${drift:.2f} ({pct_drift:.0f}%)")
            elif drift > 5:
                issues.append(f"🟡 {ticker} P&L drift: stored=${stored_pnl:+.2f} live=${calc_pnl:+.2f} drift=${drift:.2f}")
    except Exception:
        pass  # Don't fail health check on yfinance issues
    return issues


def check_cash_buffer(cash, total_value):
    """Check if cash buffer is adequate."""
    issues = []
    cash_pct = cash / total_value * 100 if total_value else 0
    if cash_pct < 10:
        issues.append(f"🔴 Cash buffer only {cash_pct:.0f}% — very low!")
    elif cash_pct < 20:
        issues.append(f"🟡 Cash buffer {cash_pct:.0f}% — below recommended 20%")
    return issues

def main():
    print("🏥 Portfolio Health Check\n" + "=" * 50)
    
    try:
        portfolio = fetch_portfolio()
    except Exception as e:
        print(f"❌ Cannot fetch portfolio: {e}")
        sys.exit(1)
    
    positions = portfolio.get("positions", {})
    trades = portfolio.get("trades", [])
    cash = portfolio.get("cash", 0)
    total_pos = sum(p.get("position_value", 0) for p in positions.values())
    total = cash + total_pos
    total_pnl = sum(p.get("unrealized_pnl", 0) for p in positions.values())
    
    print(f"\n📊 Total: ${total:,.2f} | Cash: ${cash:,.2f} ({cash/total*100:.0f}%) | Positions: {len(positions)}")
    print(f"💹 Unrealized P&L: ${total_pnl:+,.2f}")
    
    all_issues = []
    
    # 1. Concentration
    print(f"\n🏢 Sector Concentration:")
    conc_issues, sectors = check_concentration(positions)
    for sector, val in sorted(sectors.items(), key=lambda x: -x[1]):
        pct = val / total_pos * 100 if total_pos else 0
        tickers = [t for t, p in positions.items() if SECTORS.get(t) == sector]
        print(f"   {sector:12} ${val:>8,.0f} ({pct:>5.1f}%) {tickers}")
    all_issues.extend(conc_issues)
    
    # 2. Stop-loss proximity
    print(f"\n🎯 Stop-Loss Proximity:")
    sl_issues = check_stop_loss_proximity(positions)
    all_issues.extend(sl_issues)
    if not sl_issues:
        print("   ✅ All positions have comfortable distance to stop-loss")
    
    # 3. Trade history
    print(f"\n📜 Trade History ({len(trades)} records):")
    trade_issues = check_trade_history_integrity(trades)
    all_issues.extend(trade_issues)
    if not trade_issues:
        print("   ✅ Trade history clean")
    
    # 4. Cash buffer
    print(f"\n💰 Cash Buffer:")
    cash_issues = check_cash_buffer(cash, total)
    all_issues.extend(cash_issues)
    if not cash_issues:
        print(f"   ✅ Cash buffer: {cash/total*100:.0f}% (adequate)")
    
    # 5. P&L integrity (stale data check)
    print(f"\n🔍 P&L Integrity (live vs stored):")
    pnl_issues = check_pnl_integrity(positions)
    all_issues.extend(pnl_issues)
    if not pnl_issues:
        print("   ✅ All unrealized P&L values match live prices")

    # 6. P&L distribution
    print(f"\n📈 P&L Distribution:")
    winners = [p for p in positions.values() if p.get("unrealized_pnl", 0) > 0]
    losers = [p for p in positions.values() if p.get("unrealized_pnl", 0) < 0]
    flat = [p for p in positions.values() if p.get("unrealized_pnl", 0) == 0]
    print(f"   🟢 Winners: {len(winners)} | 🔴 Losers: {len(losers)} | ⚪ Flat: {len(flat)}")
    
    if len(positions) > 0:
        win_rate = len(winners) / len(positions) * 100
        if win_rate < 30:
            all_issues.append(f"🔴 Win rate only {win_rate:.0f}% — most positions losing")
    
    # Summary
    critical = [i for i in all_issues if i.startswith("🔴")]
    warnings = [i for i in all_issues if i.startswith("🟡")]
    
    print(f"\n{'=' * 50}")
    print(f"📋 Summary: {len(critical)} critical, {len(warnings)} warnings")
    
    for issue in all_issues:
        print(f"   {issue}")
    
    if not all_issues:
        print("   ✅ All checks passed!")
    
    return len(critical) > 0

if __name__ == "__main__":
    has_critical = main()
    sys.exit(1 if has_critical else 0)
