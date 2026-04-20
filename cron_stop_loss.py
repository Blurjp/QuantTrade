#!/usr/bin/env python3
"""
Stop-loss monitor — runs every 5 minutes via Railway cron.
Checks current prices against stop-loss/take-profit levels and auto-closes positions.
"""

import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import yfinance as yf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_BASE = Path(os.environ.get("OUTPUT_BASE", "outputs"))
PORTFOLIO_FILE = OUTPUT_BASE / "paper_trading" / "multi_asset_portfolio.json"


def load_portfolio() -> dict:
    if not PORTFOLIO_FILE.exists():
        logger.error("Portfolio file not found: %s", PORTFOLIO_FILE)
        return {}
    return json.loads(PORTFOLIO_FILE.read_text())


def save_portfolio(data: dict):
    PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_FILE.write_text(json.dumps(data, indent=2, default=str))


def fetch_prices_batch(tickers: list) -> dict:
    """Fetch current prices via batch yfinance download."""
    if not tickers:
        return {}
    try:
        data = yf.download(tickers, period="1d", progress=False)
        if data.empty:
            return {}
        close = data["Close"]
        prices = {}
        if len(tickers) == 1:
            val = float(close.iloc[-1])
            if val == val:  # not NaN
                prices[tickers[0]] = val
        else:
            for t in tickers:
                if t in close.columns:
                    val = close[t].iloc[-1]
                    if val == val:
                        prices[t] = float(val)
        return prices
    except Exception as e:
        logger.error("Failed to fetch prices: %s", e)
        return {}


def check_stop_loss():
    """Main stop-loss check logic.""""

    portfolio = load_portfolio()
    if not portfolio:
        logger.info("No portfolio data found. Skipping.")
        return

    positions = portfolio.get("positions", {})
    if not positions:
        logger.info("No open positions. Skipping.")
        return

    tickers = list(positions.keys())
    logger.info("Checking %d positions: %s", len(tickers), tickers)

    # Fetch prices
    prices = fetch_prices_batch(tickers)
    if not prices:
        logger.warning("Could not fetch any prices. Skipping.")
        return

    logger.info("Fetched %d prices: %s", len(prices), 
                {k: f"${v:.2f}" for k, v in prices.items()})

    closed_any = False

    for ticker, pos in list(positions.items()):
        if ticker not in prices:
            logger.warning("No price for %s, skipping.", ticker)
            continue

        current_price = prices[ticker]
        entry_price = float(pos.get("entry_price", 0))
        stop_loss = float(pos.get("stop_loss", 0))
        take_profit = float(pos.get("take_profit", 0))
        direction = pos.get("direction", "long")
        quantity = float(pos.get("quantity", 0))

        # Update unrealized P&L
        if direction == "long":
            pos["unrealized_pnl"] = quantity * (current_price - entry_price)
        else:
            pos["unrealized_pnl"] = quantity * (entry_price - current_price)

        triggered = False
        reason = ""

        if stop_loss > 0:
            if direction == "long" and current_price <= stop_loss:
                triggered = True
                reason = f"STOP LOSS: {ticker} LONG @ ${current_price:.2f} <= stop ${stop_loss:.2f}"
            elif direction == "short" and current_price >= stop_loss:
                triggered = True
                reason = f"STOP LOSS: {ticker} SHORT @ ${current_price:.2f} >= stop ${stop_loss:.2f}"

        if not triggered and take_profit > 0:
            if direction == "long" and current_price >= take_profit:
                triggered = True
                reason = f"TAKE PROFIT: {ticker} LONG @ ${current_price:.2f} >= target ${take_profit:.2f}"
            elif direction == "short" and current_price <= take_profit:
                triggered = True
                reason = f"TAKE PROFIT: {ticker} SHORT @ ${current_price:.2f} <= target ${take_profit:.2f}"

        if triggered:
            logger.info("🚨 %s", reason)

            # Calculate close value
            position_value = float(pos.get("position_value", 0))
            if direction == "long":
                close_value = current_price * quantity
                pnl = (current_price - entry_price) * quantity
            else:
                close_value = position_value + (entry_price - current_price) * quantity
                pnl = (entry_price - current_price) * quantity

            # Return cash
            portfolio["cash"] = float(portfolio.get("cash", 0)) + close_value

            # Record trade
            trade = {
                "ticker": ticker,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": current_price,
                "quantity": quantity,
                "pnl": pnl,
                "reason": reason,
                "closed_at": datetime.now().isoformat(),
            }
            portfolio.setdefault("trades", []).append(trade)

            # Remove position
            del portfolio["positions"][ticker]
            closed_any = True

            logger.info("Closed %s %s: P&L=$%.2f, cash returned=$%.2f",
                        direction.upper(), ticker, pnl, close_value)

    if closed_any:
        save_portfolio(portfolio)
        logger.info("Portfolio saved after stop-loss closures.")
        
        # Log summary
        remaining = len(portfolio.get("positions", {}))
        cash = portfolio.get("cash", 0)
        logger.info("Remaining: %d positions, $%.2f cash", remaining, cash)
    else:
        logger.info("No stop-loss/take-profit triggered.")
        # Still save updated unrealized P&L
        save_portfolio(portfolio)


if __name__ == "__main__":
    check_stop_loss()
