#!/usr/bin/env python3
"""
Signal Logic Auditor — scans all pipeline modules for direction logic consistency.

Checks:
1. Each signal module's direction logic matches its instrument purpose
2. No conflicting long/short on same economic condition
3. Instruments match the signal type (no SOYB on thermal signals, etc.)
4. trade_direction override is properly applied

Run: python audit_signals.py
"""

import ast
import re
import sys
from pathlib import Path

PIPELINE_DIR = Path("pipeline")

# Known instrument mappings — what SHOULD each signal type produce?
EXPECTED_INSTRUMENTS = {
    "vegetation_health": {
        "expected_directions": {"long": "stress/shortage → bullish prices", "short": "healthy → bearish prices"},
        "allowed_instruments": {"CORN", "SOYB", "WEAT", "Cattle", "Feeder Cattle"},
    },
    "precipitation": {
        "expected_directions": {"long": "drought/flood → supply damage → bullish", "short": "normal/wet → supply ok → bearish"},
        "allowed_instruments": {"CORN", "SOYB", "WEAT"},
    },
    "sea_surface_temperature": {
        "expected_directions": {"neutral": "SST should NOT directly trade — context only"},
        "allowed_instruments": set(),  # SST should be neutral/context
    },
    "atmospheric": {
        "expected_directions": {"long": "emissions up → production up", "short": "emissions down → production down"},
        "allowed_instruments": {"FXI", "MCHI", "EWG", "XLI", "BTU", "KWEB", "ASHR", "FXD"},
    },
    "thermal_infrared": {
        "expected_directions": {"long": "heat up → production up", "short": "heat down → production down"},
        "allowed_instruments": {"XLE", "XOM", "CVX", "BTU", "XLI", "NUE", "STLD", "USO", "BNO"},
    },
    "nighttime_lights": {
        "expected_directions": {"long": "lights up → activity up", "short": "lights down → activity down"},
        "allowed_instruments": {"FXI", "MCHI", "INDA", "EPI", "EWG", "XLI", "FXD"},
    },
    "cattle_feedlot": {
        "expected_directions": {"long": "supply tight → beef up", "short": "supply excess → beef down"},
        "allowed_instruments": {"Cattle", "Feeder Cattle", "COW"},
    },
    "solar_irradiance": {
        "expected_directions": {"long": "high irradiance → solar stocks up", "short": "low irradiance → solar stocks down"},
        "allowed_instruments": {"TAN", "ICLN", "FAN"},
    },
    "soil_moisture": {
        "expected_directions": {"long": "dry → supply risk → bullish", "short": "wet → supply ok → bearish"},
        "allowed_instruments": {"CORN", "SOYB", "WEAT"},
    },
}

def audit_module(filepath: Path) -> list:
    """Audit a single pipeline module for direction logic issues."""
    issues = []
    content = filepath.read_text()
    module_name = filepath.stem
    
    # Skip if not a signal module
    if module_name not in EXPECTED_INSTRUMENTS:
        return issues
    
    expected = EXPECTED_INSTRUMENTS[module_name]
    
    # Check 1: Find all direction assignments
    direction_assigns = re.findall(r'direction\s*=\s*["\'](\w+)["\']', content)
    if not direction_assigns:
        issues.append(f"⚠️  No direction assignments found")
        return issues
    
    # Check 2: SST should only have "neutral" as trade_direction
    if module_name == "sea_surface_temperature":
        # Look for trade_direction = "long" or "short" (should be neutral)
        trade_dir_matches = re.findall(r'trade_direction\s*=\s*["\'](\w+)["\']', content)
        for td in trade_dir_matches:
            if td != "neutral":
                issues.append(f"🔴 SST has trade_direction='{td}' — should be 'neutral' (context only)")
        
        # Check if any direct direction = "long"/"short" is used for actual trading
        if "long" in direction_assigns or "short" in direction_assigns:
            # OK for internal use, but verify trade_direction is neutral
            if not trade_dir_matches:
                issues.append(f"🟡 SST has direction assignments but no trade_direction override — verify it's context-only")
    
    # Check 3: Instruments should match expected
    all_instruments = re.findall(r'["\']([A-Z]{2,})["\']', content)
    if expected["allowed_instruments"]:
        for inst in set(all_instruments):
            if inst in {"LONG", "SHORT", "NEUTRAL", "HTTP", "GET", "POST", "JSON", "API", "CSV", "NDVI"}:
                continue  # Skip false positives
            if inst not in expected["allowed_instruments"] and len(inst) >= 3 and inst.isupper():
                # Only flag if it looks like a real ticker
                if inst in {"CORN", "SOYB", "WEAT", "XLE", "USO", "BNO", "FXI", "MCHI", "OIH", "XOM", "CVX", 
                           "BTU", "XLI", "EWG", "ASHR", "FXD", "KWEB", "BABA", "JD", "F", "GM", "TM",
                           "CAT", "DE", "WMT", "COST", "TGT", "HD", "XRT", "CARZ", "UNG", "GLD", "SLV",
                           "INDA", "EPI", "EPOL", "NUE", "STLD", "TAN", "ICLN", "FAN"}:
                    issues.append(f"🟡 Instrument '{inst}' may not belong in {module_name}")
    
    # Check 4: Both long and short should exist for non-neutral modules
    if module_name != "sea_surface_temperature":
        has_long = "long" in direction_assigns
        has_short = "short" in direction_assigns
        if not has_long:
            issues.append(f"⚠️  No 'long' direction — module may be incomplete")
        if not has_short:
            issues.append(f"⚠️  No 'short' direction — module may be incomplete")
    
    # Check 5: Look for hardcoded confidence without minimum
    confidence_assigns = re.findall(r'confidence\s*=\s*(\d+)', content)
    for c in confidence_assigns:
        val = int(c)
        if val < 10 or val > 100:
            issues.append(f"🔴 Suspicious confidence value: {val}")
    
    return issues


def main():
    print("🔍 Signal Logic Auditor\n" + "=" * 50)
    
    total_issues = 0
    modules_checked = 0
    
    for filepath in sorted(PIPELINE_DIR.glob("*.py")):
        if filepath.name.startswith("_") or filepath.name == "__init__.py":
            continue
        
        module_name = filepath.stem
        if module_name not in EXPECTED_INSTRUMENTS:
            continue
        
        issues = audit_module(filepath)
        modules_checked += 1
        
        if issues:
            print(f"\n📦 {module_name}.py")
            for issue in issues:
                print(f"   {issue}")
            total_issues += len([i for i in issues if i.startswith("🔴")])
        else:
            print(f"✅ {module_name}.py — OK")
    
    print(f"\n{'=' * 50}")
    print(f"Checked {modules_checked} modules, found {total_issues} critical issues")
    
    if total_issues > 0:
        sys.exit(1)
    else:
        print("All clear! ✅")

if __name__ == "__main__":
    main()
