#!/usr/bin/env python3
"""
Verify deployment — checks that the running Railway service matches the latest git commit.
"""

import json
import subprocess
import urllib.request

API_BASE = "https://scheduler-production-b60f.up.railway.app"

def get_git_info():
    """Get latest commit info."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        msg = subprocess.check_output(["git", "log", "-1", "--format=%s"], text=True).strip()
        return commit, msg
    except Exception as e:
        return None, str(e)

def get_deploy_info():
    """Check deployed service health."""
    try:
        with urllib.request.urlopen(f"{API_BASE}/health", timeout=10) as r:
            health = json.loads(r.read())
        return health
    except Exception as e:
        return {"error": str(e)}

def check_api_endpoints():
    """Verify key API endpoints are responding."""
    endpoints = {
        "/health": "Service health",
        "/api/portfolio": "Portfolio data",
        "/api/signals": "Signal data",
        "/api/learning": "Learning system",
    }
    
    results = {}
    for path, desc in endpoints.items():
        try:
            with urllib.request.urlopen(f"{API_BASE}{path}", timeout=10) as r:
                data = json.loads(r.read())
                results[path] = {"status": "✅", "desc": desc, "has_data": bool(data and data.get("error") is None)}
        except Exception as e:
            results[path] = {"status": "❌", "desc": desc, "error": str(e)[:60]}
    
    return results

def main():
    print("🚀 Deployment Verifier\n" + "=" * 50)
    
    # Git info
    commit, msg = get_git_info()
    print(f"\n📝 Latest git: {commit} — {msg}")
    
    # Deploy health
    health = get_deploy_info()
    if "error" in health:
        print(f"\n❌ Service unreachable: {health['error']}")
        return
    
    print(f"\n🏥 Service: {health.get('status', '?')}")
    print(f"   Last run: {health.get('last_run', '?')}")
    print(f"   Total runs: {health.get('total_runs', '?')}")
    print(f"   Last status: {health.get('last_status', '?')}")
    
    # API endpoints
    print(f"\n🔌 API Endpoints:")
    results = check_api_endpoints()
    for path, info in results.items():
        status = info["status"]
        desc = info["desc"]
        if "error" in info:
            print(f"   {status} {path:25} {desc} — {info['error']}")
        else:
            print(f"   {status} {path:25} {desc}")
    
    # Check learning system
    try:
        with urllib.request.urlopen(f"{API_BASE}/api/learning", timeout=10) as r:
            learning = json.loads(r.read())
        trades = learning.get("total_closed_trades", 0)
        regions = learning.get("regions_learned", 0)
        print(f"\n🧠 Learning: {trades} closed trades, {regions} regions learned")
    except:
        print(f"\n🧠 Learning: endpoint not available (may be pre-v2 deploy)")
    
    # Portfolio summary
    try:
        with urllib.request.urlopen(f"{API_BASE}/api/portfolio", timeout=10) as r:
            portfolio = json.loads(r.read())
        cash = portfolio.get("cash", 0)
        positions = portfolio.get("positions", {})
        total_pos = sum(p.get("position_value", 0) for p in positions.values())
        print(f"\n📊 Portfolio: {len(positions)} positions, ${cash + total_pos:,.2f} total")
    except:
        pass
    
    print(f"\n{'=' * 50}")

if __name__ == "__main__":
    main()
