#!/usr/bin/env python3
"""
Portfolio status — always reads from source of truth (API/portfolio JSON).
Never estimates. Never assumes fixed position sizes.
"""

import json
import sys
import urllib.request

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

    # ─── Positions (use REAL position_value, never estimate) ───
    total_pos_val = 0
    total_unrealized_pnl = 0

    if positions:
        print("📊 Holdings:")
        for t, p in sorted(positions.items()):
            pos_val = p.get("position_value", 0)
            unrealized = p.get("unrealized_pnl", 0)
            entry = p.get("entry_price", 0)
            direction = p.get("direction", "long").upper()
            total_pos_val += pos_val
            total_unrealized_pnl += unrealized

            emoji = "🟢" if unrealized >= 0 else "🔴"
            pnl_pct = (unrealized / pos_val * 100) if pos_val else 0
            print(f"  {emoji} {direction:5} {t:6} val=${pos_val:>9,.2f}  "
                  f"entry=${entry:.2f}  P&L=${unrealized:>+,.2f} ({pnl_pct:+.1f}%)")

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
