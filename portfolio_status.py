#!/usr/bin/env python3
"""
Portfolio status — always reads from source of truth (API/portfolio JSON).
Never estimates. Never assumes fixed position sizes.
"""

import json
import sys
import urllib.request
from pathlib import Path

API_BASE = "https://scheduler-production-b60f.up.railway.app"


def get_status():
    """Get accurate portfolio status from API."""
    try:
        with urllib.request.urlopen(f"{API_BASE}/api/portfolio", timeout=15) as r:
            d = json.loads(r.read())
    except Exception as e:
        print(f"ERROR: Failed to fetch portfolio: {e}")
        sys.exit(1)

    cash = d.get("cash", 0)
    positions = d.get("positions", {})
    trades = d.get("trades", [])

    # ─── Positions (RECALCULATE from live prices, never trust stale data) ───
    total_pos_val = 0
    total_unrealized_pnl = 0

    if positions:
        # Fetch live prices for all tickers
        live_prices = {}
        try:
            import subprocess
            tickers = list(positions.keys())
            # Use venv python with yfinance installed
            venv_python = str(Path(__file__).parent / ".venv" / "bin" / "python3")
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
            if result.stdout.strip():
                live_prices = json.loads(result.stdout.strip())
        except Exception:
            pass  # Fallback to stale data

        print("📊 Holdings:")
        for t, p in sorted(positions.items()):
            entry = p.get("entry_price", 0)
            qty = p.get("quantity", 0)
            direction = p.get("direction", "long").upper()

            # Recalculate from live price if available
            live_price = live_prices.get(t)
            if live_price and qty:
                if direction == "LONG":
                    unrealized = qty * (live_price - entry)
                    pos_val = qty * live_price
                else:
                    unrealized = qty * (entry - live_price)
                    pos_val = p.get("position_value", 0) + unrealized
                current_price = live_price
            else:
                # Fallback to stale data
                unrealized = p.get("unrealized_pnl", 0)
                pos_val = p.get("position_value", 0)
                current_price = p.get("current_price", 0)

            total_pos_val += pos_val
            total_unrealized_pnl += unrealized

            emoji = "🟢" if unrealized >= 0 else "🔴"
            pnl_pct = (unrealized / pos_val * 100) if pos_val else 0
            stale_flag = " ⚠️stale" if not live_price else ""
            print(f"  {emoji} {direction:5} {t:6} cur=${current_price:>8.2f}  "
                  f"entry=${entry:.2f}  P&L=${unrealized:>+,.2f} ({pnl_pct:+.1f}%){stale_flag}")

    # ─── Summary ───
    total_value = cash + total_pos_val
    print(f"\n💰 Cash:        ${cash:>10,.2f}")
    print(f"📈 Holdings:    ${total_pos_val:>10,.2f}")
    print(f"📊 Total:       ${total_value:>10,.2f}")
    print(f"💹 Unrealized:  ${total_unrealized_pnl:>+10,.2f}")

    # ─── Realized trades ───
    if trades:
        realized = [t for t in trades if t.get("pnl", 0) != 0]
        if realized:
            total_realized = sum(t.get("pnl", 0) for t in realized)
            wins = sum(1 for t in realized if t.get("pnl", 0) > 0)
            print(f"📜 Realized:    ${total_realized:>+10,.2f} ({wins}/{len(realized)} wins)")

    # ─── Learning ───
    try:
        with urllib.request.urlopen(f"{API_BASE}/api/learning", timeout=10) as r:
            learn = json.loads(r.read())
        closed = learn.get("total_closed_trades", 0)
        wr = learn.get("overall_win_rate", "N/A")
        regions = learn.get("regions_learned", 0)
        if closed > 0:
            print(f"🧠 Learning:    {closed} trades, win rate: {wr}, {regions} regions learned")
    except Exception:
        pass

    return {
        "cash": cash,
        "holdings_value": total_pos_val,
        "total_value": total_value,
        "unrealized_pnl": total_unrealized_pnl,
        "num_positions": len(positions),
    }


if __name__ == "__main__":
    get_status()
