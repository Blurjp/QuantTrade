#!/usr/bin/env python3
"""
Auto-trade executor — runs as Railway cron or on-demand.

Pipeline:
  1. Collect actionable signals from all monitoring modules
  2. Map signals → TradeCandidates via TradeMapper
  3. For each candidate: fetch live price, size position, open via MultiAssetPortfolio
  4. Upload updated portfolio to Railway scheduler API
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_BASE = Path(os.environ.get("OUTPUT_BASE", "outputs"))
PORTFOLIO_FILE = OUTPUT_BASE / "paper_trading" / "multi_asset_portfolio.json"
SCHEDULER_API = os.environ.get(
    "SCHEDULER_API_URL",
    "https://scheduler-production-b60f.up.railway.app",
)

# ── position sizing for $10k account ──────────────────────────────
MAX_POSITION_PCT = 0.10  # max 10% of capital per trade
MIN_CONFIDENCE = 0.5     # only trade medium+ confidence signals


def fetch_price(ticker: str) -> float | None:
    """Fetch latest price for a ticker."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.warning("Price fetch failed for %s: %s", ticker, e)
        return None


def collect_actionable_signals() -> list[dict]:
    """Run signal monitor and return actionable signals."""
    from notifications.signal_monitor import SignalMonitor
    monitor = SignalMonitor()
    summary = monitor.check_all_modules()
    logger.info("Signals: total=%d actionable=%d",
                summary["total_signals"], summary["actionable_signals"])

    # Return all signals that are actionable
    actionable = []
    for sig in monitor._get_recent_signals() if hasattr(monitor, '_get_recent_signals') else []:
        if monitor.is_actionable(sig):
            actionable.append(sig)
    return actionable


def collect_signals_from_outputs() -> list[dict]:
    """
    Read latest signal files from outputs/ and extract actionable ones.
    Signal format: direction=long|short|neutral, confidence=0-100.
    """
    import glob
    from datetime import date

    today = date.today().isoformat()
    signals = []
    seen = set()

    # Scan today's signal files
    for path in sorted(glob.glob(str(OUTPUT_BASE / f"**/signal_*_{today}.json"), recursive=True)):
        try:
            data = json.loads(Path(path).read_text())
            if not isinstance(data, dict):
                continue
            direction = data.get("direction", "").lower()
            confidence = float(data.get("confidence", 0))
            region_id = data.get("region_id", "")

            if direction not in ("long", "short"):
                continue
            if confidence < 85:  # Only trade 85%+ confidence
                continue
            if region_id in seen:
                continue

            seen.add(region_id)
            # Normalize to internal format
            data["action"] = direction.upper()
            data["asset_class"] = _infer_asset_class(data.get("signal_type", ""), region_id)
            signals.append(data)
        except Exception:
            continue

    # Sort by confidence desc
    signals.sort(key=lambda s: float(s.get("confidence", 0)), reverse=True)
    return signals


def _infer_asset_class(signal_type: str, region_id: str) -> str:
    """Map signal type / region to asset class for position sizing."""
    if "oil" in region_id or "energy" in signal_type or "hormuz" in region_id or "cushing" in region_id:
        return "energy"
    if "corn" in region_id or "soy" in region_id or "wheat" in region_id or "agri" in signal_type:
        return "agriculture"
    if "auto" in region_id or "detroit" in region_id:
        return "auto"
    if "retail" in region_id or "walmart" in region_id or "costco" in region_id:
        return "retail"
    return "etf"


def map_signal_to_trade(signal: dict, capital: float) -> dict | None:
    """
    Convert a signal dict to a trade plan.
    Returns dict with ticker, direction, value, rationale, etc. or None.
    """
    action = signal.get("action", "").upper()
    if action not in ("LONG", "SHORT"):
        # Also check direction field
        direction = signal.get("direction", "").upper()
        if direction in ("LONG", "SHORT"):
            action = direction
        else:
            return None

    confidence = signal.get("confidence", 0)
    if isinstance(confidence, str):
        if confidence == "Low":
            return None
    elif isinstance(confidence, (int, float)):
        if confidence < 70:
            return None

    # Get region / signal source
    region = signal.get("region_id", signal.get("region", ""))
    signal_type = signal.get("signal_type", signal.get("strategy", "unknown"))
    rationale = signal.get("rationale", signal.get("thesis", f"{signal_type} signal for {region}"))

    # Map region to tickers
    ticker_map = {
        # Energy / Oil
        "hormuz": ["USO"],
        "suez": ["USO"],
        "malacca": ["USO"],
        "cushing": ["USO"],
        "gulf_mexico": ["USO"],
        "la_longbeach": ["USO"],
        "panama_canal": ["USO"],
        "global_oil_meta": ["USO"],
        "atlantic_hurricane": ["USO"],
        "refinery": ["XLE"],
        "permian": ["XLE"],
        "middle_east_oil": ["XLE"],
        "power_plant": ["XLU"],
        # Agriculture
        "corn": ["CORN"],
        "soy": ["SOYB"],
        "wheat": ["WEAT"],
        "cotton": ["BAL"],
        "cocoa": ["NIB"],
        "argentina_pampas": ["DBA"],
        "brazil": ["SOYB"],
        "great_plains": ["DBA"],
        "midwest": ["DBA"],
        "corn_belt": ["CORN"],
        "flint_hills": ["CORN"],
        "sandhills": ["CORN"],
        # Retail
        "walmart": ["WMT"],
        "costco": ["COST"],
        "retail": ["XRT"],
        # Auto
        "detroit": ["CARZ"],
        "auto": ["CARZ"],
        # Industrial
        "steel": ["X"],
        "semiconductor": ["SMH"],
        # China / EM
        "china": ["FXI"],
        "india": ["INDA"],
        # Climate / El Nino
        "nino": ["DBA"],
        "peru": ["EPU"],
        "benguela": ["DBA"],
        "pacific_warm": ["DBA"],
        "indian_ocean": ["DBA"],
        # Solar / Energy
        "solar": ["TAN"],
        "datacenter": ["SMH"],
    }

    # Find matching tickers
    tickers = []
    region_lower = region.lower()
    # Try specific matches first
    for key, mapped in sorted(ticker_map.items(), key=lambda x: -len(x[0])):
        if key in region_lower:
            tickers = mapped
            break

    if not tickers:
        # Try signal_type based mapping
        st = signal_type.lower()
        if "oil" in st or "energy" in st:
            tickers = ["USO"]
        elif "precip" in st or "soil" in st or "vegetation" in st:
            tickers = ["DBA"]
        elif "thermal" in st:
            tickers = ["XLE"]
        elif "solar" in st:
            tickers = ["TAN"]
        elif "nighttime" in st:
            tickers = ["FXI"]  # default to EM
        else:
            return None

    # Pick primary ticker (first)
    ticker = tickers[0]

    # Size based on confidence (numeric 0-100)
    conf_val = float(confidence) if isinstance(confidence, (int, float)) else 75.0
    if conf_val >= 85:
        size_pct = 0.08
    elif conf_val >= 75:
        size_pct = 0.06
    else:
        size_pct = 0.05
    size_pct = min(size_pct, MAX_POSITION_PCT)
    position_value = capital * size_pct

    # Adjust risk by signal type
    if signal_type in ("sea_surface_temperature",):
        sl, tp = 0.04, 0.08  # tighter for macro signals
    else:
        sl, tp = 0.05, 0.10

    return {
        "ticker": ticker,
        "direction": action.lower(),
        "position_value": position_value,
        "asset_class": signal.get("asset_class", "etf"),
        "rationale": f"[{signal_type}] {rationale} (conf={conf_val:.0f}%)",
        "region_id": region,
        "stop_loss_pct": sl,
        "take_profit_pct": tp,
        "confidence": conf_val,
    }


def upload_portfolio_to_railway(portfolio_data: dict):
    """Push portfolio to Railway scheduler API."""
    import urllib.request
    url = f"{SCHEDULER_API}/api/portfolio/upload"
    body = json.dumps(portfolio_data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            logger.info("Portfolio uploaded to Railway: %s", result)
    except Exception as e:
        logger.error("Failed to upload portfolio to Railway: %s", e)


def run_auto_trade():
    """Main auto-trade execution."""
    logger.info("=" * 60)
    logger.info("Auto-trade started at %s", datetime.now().isoformat())
    logger.info("=" * 60)

    # Load current portfolio
    if not PORTFOLIO_FILE.exists():
        logger.error("Portfolio file not found: %s", PORTFOLIO_FILE)
        return

    portfolio = json.loads(PORTFOLIO_FILE.read_text())
    cash = float(portfolio.get("cash", 0))
    positions = portfolio.get("positions", {})
    capital = cash + sum(
        float(p.get("position_value", 0)) for p in positions.values()
    )

    logger.info("Capital: $%.2f | Cash: $%.2f | Open positions: %d",
                capital, cash, len(positions))

    if len(positions) >= 10:
        logger.info("Max positions reached (10). Skipping.")
        return

    # Step 1: Collect signals
    try:
        signals = collect_actionable_signals()
    except Exception as e:
        logger.warning("Signal monitor failed, falling back to output files: %s", e)
        signals = collect_signals_from_outputs()

    if not signals:
        logger.info("Trying output file fallback...")
        signals = collect_signals_from_outputs()

    if not signals:
        logger.info("No actionable signals found. Nothing to trade.")
        return

    logger.info("Found %d actionable signals", len(signals))

    # Step 2: Map signals to trades
    trades_placed = 0
    for signal in signals:
        trade_plan = map_signal_to_trade(signal, capital)
        if not trade_plan:
            continue

        ticker = trade_plan["ticker"]
        if ticker in positions:
            logger.info("Already have position in %s, skipping", ticker)
            continue

        # Fetch live price
        price = fetch_price(ticker)
        if not price or price <= 0:
            logger.warning("No price for %s, skipping", ticker)
            continue

        direction = trade_plan["direction"]
        position_value = min(trade_plan["position_value"], cash * 0.9)  # don't blow all cash
        if position_value < 100:
            logger.warning("Position value too small ($%.2f) for %s", position_value, ticker)
            continue

        quantity = position_value / price

        # Calculate stop loss / take profit
        if direction == "long":
            stop_loss = price * (1 - trade_plan["stop_loss_pct"])
            take_profit = price * (1 + trade_plan["take_profit_pct"])
        else:
            stop_loss = price * (1 + trade_plan["stop_loss_pct"])
            take_profit = price * (1 - trade_plan["take_profit_pct"])

        # Open position
        pos = {
            "ticker": ticker,
            "asset_class": trade_plan["asset_class"],
            "direction": direction,
            "quantity": quantity,
            "entry_price": price,
            "entry_date": datetime.now().strftime("%Y-%m-%d"),
            "position_value": position_value,
            "rationale": trade_plan["rationale"],
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "unrealized_pnl": 0,
            "region_id": trade_plan["region_id"],
        }

        portfolio["positions"][ticker] = pos
        # Both long and short deduct position_value as margin
        portfolio["cash"] = float(portfolio["cash"]) - position_value

        # Record trade
        trade_record = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "ticker": ticker,
            "action": f"OPEN_{direction.upper()}",
            "price": price,
            "quantity": quantity,
            "value": position_value,
            "rationale": trade_plan["rationale"],
        }
        portfolio.setdefault("trades", []).append(trade_record)

        logger.info("✅ OPENED %s %s | Qty: %.2f @ $%.2f | Value: $%.2f | SL: $%.2f | TP: $%.2f",
                     direction.upper(), ticker, quantity, price, position_value,
                     stop_loss, take_profit)
        logger.info("   Reason: %s", trade_plan["rationale"])

        trades_placed += 1
        cash = float(portfolio["cash"])

        if trades_placed >= 3:
            logger.info("Max 3 new trades per run. Stopping.")
            break

    # Save
    if trades_placed > 0:
        portfolio["last_updated"] = datetime.now().isoformat()
        PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)
        PORTFOLIO_FILE.write_text(json.dumps(portfolio, indent=2))
        logger.info("Portfolio saved locally")

        # Upload to Railway
        upload_portfolio_to_railway(portfolio)

        remaining_cash = float(portfolio["cash"])
        open_count = len(portfolio["positions"])
        logger.info("Summary: %d trades placed | Cash: $%.2f | Positions: %d",
                     trades_placed, remaining_cash, open_count)
    else:
        logger.info("No trades placed this run.")

    logger.info("=" * 60)
    logger.info("Auto-trade complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_auto_trade()
