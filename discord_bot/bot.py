"""
Discord Bot Commands via Webhook + Flask Endpoint

Provides command handling for portfolio management via Discord.
Uses the existing Discord webhook for responses and exposes a
Flask endpoint for Discord interactions.

Commands:
    !status / !portfolio  - Show current positions and P&L
    !close <ticker>       - Close a position
    !signals              - Show latest signals
    !help                 - List commands
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DiscordCommandHandler:
    """Handles Discord bot commands and sends responses via webhook."""

    def __init__(self, output_base: str = "outputs"):
        self.output_base = Path(output_base)
        self.webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")

    def _send_webhook(self, content: str = "", embed: Optional[Dict] = None) -> bool:
        """Send a message via Discord webhook."""
        if not self.webhook_url:
            logger.warning("Discord webhook URL not configured")
            return False

        try:
            import requests
            data = {}
            if content:
                data["content"] = content
            if embed:
                data["embeds"] = [embed]
            if not data:
                return False

            resp = requests.post(self.webhook_url, json=data, timeout=10)
            return resp.status_code == 204
        except Exception as e:
            logger.error(f"Discord webhook failed: {e}")
            return False

    def handle_command(self, command: str) -> Dict:
        """
        Parse and handle a Discord command.

        Args:
            command: Raw command string (e.g., "!status" or "!close F")

        Returns:
            Dict with "response" text and optional "embed"
        """
        parts = command.strip().split()
        if not parts:
            return {"response": "No command provided"}

        cmd = parts[0].lower().lstrip("!")
        args = parts[1:]

        handlers = {
            "status": self._cmd_status,
            "portfolio": self._cmd_status,
            "close": self._cmd_close,
            "signals": self._cmd_signals,
            "help": self._cmd_help,
        }

        handler = handlers.get(cmd)
        if not handler:
            return {"response": f"Unknown command: `!{cmd}`. Use `!help` for available commands."}

        return handler(args)

    def _load_portfolio(self):
        """Load portfolio instance."""
        from paper_trading.multi_asset_portfolio import MultiAssetPortfolio
        return MultiAssetPortfolio(output_base=str(self.output_base))

    def _cmd_status(self, args: list) -> Dict:
        """Show current portfolio status."""
        try:
            portfolio = self._load_portfolio()
            from pipeline.price_feed import get_prices_for_portfolio
            prices = get_prices_for_portfolio(portfolio)
            summary = portfolio.get_summary(prices)

            fields = [
                {"name": "Total Value", "value": f"${summary['total_value']:,.2f}", "inline": True},
                {"name": "Cash", "value": f"${summary['cash']:,.2f}", "inline": True},
                {"name": "Return", "value": f"{summary['total_return_pct']:+.2f}%", "inline": True},
                {"name": "Positions", "value": str(summary['num_positions']), "inline": True},
                {"name": "Closed Trades", "value": str(summary['num_trades']), "inline": True},
                {"name": "Winning", "value": str(summary['winning_trades']), "inline": True},
            ]

            # Add each position
            if portfolio.positions:
                pos_lines = []
                for ticker, pos in portfolio.positions.items():
                    current = prices.get(ticker, pos.entry_price)
                    if pos.direction == "long":
                        pnl = pos.quantity * (current - pos.entry_price)
                    else:
                        pnl = pos.quantity * (pos.entry_price - current)
                    emoji = "+" if pnl >= 0 else ""
                    pos_lines.append(
                        f"`{ticker}` {pos.direction.upper()} "
                        f"Entry: ${pos.entry_price:.2f} Now: ${current:.2f} "
                        f"P&L: {emoji}${pnl:.2f}"
                    )
                fields.append({
                    "name": "Open Positions",
                    "value": "\n".join(pos_lines),
                    "inline": False,
                })

            color = 3066993 if summary['total_return_pct'] >= 0 else 15158332
            embed = {
                "title": "Portfolio Status",
                "color": color,
                "fields": fields,
                "footer": {"text": "QuantTrade"},
                "timestamp": datetime.utcnow().isoformat(),
            }

            return {"response": "", "embed": embed}

        except Exception as e:
            logger.error(f"Status command failed: {e}")
            return {"response": f"Error fetching portfolio status: {e}"}

    def _cmd_close(self, args: list) -> Dict:
        """Close a position by ticker."""
        if not args:
            return {"response": "Usage: `!close <TICKER>` (e.g., `!close F`)"}

        ticker = args[0].upper()

        try:
            portfolio = self._load_portfolio()
            if ticker not in portfolio.positions:
                available = ", ".join(portfolio.positions.keys()) or "none"
                return {"response": f"No open position for `{ticker}`. Open positions: {available}"}

            from pipeline.price_feed import fetch_all_prices
            prices = fetch_all_prices([ticker])
            current_price = prices.get(ticker)

            if not current_price:
                return {"response": f"Could not fetch price for `{ticker}`. Try again later."}

            trade = portfolio.close_position(
                ticker, current_price, "Closed via Discord command"
            )

            if trade:
                # Track in signal history
                try:
                    from tracking.signal_tracker import SignalTracker
                    tracker = SignalTracker(output_base=str(self.output_base))
                    tracker.record_position_closed(ticker, current_price, trade.pnl)
                except Exception:
                    pass

                pnl_emoji = "+" if trade.pnl >= 0 else ""
                return {
                    "response": "",
                    "embed": {
                        "title": f"Position Closed: {ticker}",
                        "color": 3066993 if trade.pnl >= 0 else 15158332,
                        "fields": [
                            {"name": "Price", "value": f"${current_price:.2f}", "inline": True},
                            {"name": "P&L", "value": f"{pnl_emoji}${trade.pnl:.2f}", "inline": True},
                            {"name": "Quantity", "value": f"{trade.quantity:.2f}", "inline": True},
                        ],
                        "footer": {"text": "QuantTrade"},
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                }
            else:
                return {"response": f"Failed to close position for `{ticker}`."}

        except Exception as e:
            logger.error(f"Close command failed: {e}")
            return {"response": f"Error closing position: {e}"}

    def _cmd_signals(self, args: list) -> Dict:
        """Show latest signals."""
        try:
            output_root = Path(self.output_base)
            candidates = sorted(
                p.parent.name for p in output_root.glob("*/daily_summary.json")
                if p.parent.name[:4].isdigit()
            )

            if not candidates:
                return {"response": "No signals available yet."}

            latest_date = candidates[-1]
            summary_file = output_root / latest_date / "daily_summary.json"
            summary = json.loads(summary_file.read_text())

            signals = summary.get("signals", {})
            if not signals:
                return {"response": f"No signals for {latest_date}."}

            lines = []
            for region_id, sig in list(signals.items())[:10]:
                direction = sig.get("direction", "neutral")
                emoji = {"long": "+", "short": "-"}.get(direction, "~")
                conf = sig.get("confidence", "?")
                instruments = ", ".join(sig.get("instruments", []))
                lines.append(
                    f"`{emoji}` **{region_id}** {direction.upper()} "
                    f"| Conf: {conf} | {instruments}"
                )

            embed = {
                "title": f"Latest Signals - {latest_date}",
                "description": "\n".join(lines),
                "color": 5814783,
                "footer": {"text": f"QuantTrade | {len(signals)} total signals"},
                "timestamp": datetime.utcnow().isoformat(),
            }

            return {"response": "", "embed": embed}

        except Exception as e:
            logger.error(f"Signals command failed: {e}")
            return {"response": f"Error fetching signals: {e}"}

    def _cmd_help(self, args: list) -> Dict:
        """Show help message."""
        embed = {
            "title": "QuantTrade Bot Commands",
            "color": 5814783,
            "fields": [
                {"name": "!status / !portfolio", "value": "Show current positions and P&L", "inline": False},
                {"name": "!close <ticker>", "value": "Close a position (e.g., `!close F`)", "inline": False},
                {"name": "!signals", "value": "Show latest trading signals", "inline": False},
                {"name": "!help", "value": "Show this help message", "inline": False},
            ],
            "footer": {"text": "QuantTrade Satellite Trading System"},
        }
        return {"response": "", "embed": embed}

    def process_and_respond(self, command: str) -> bool:
        """Process a command and send the response via webhook."""
        result = self.handle_command(command)
        return self._send_webhook(
            content=result.get("response", ""),
            embed=result.get("embed"),
        )
