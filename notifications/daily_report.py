"""
Daily P&L Email Report

Generates and sends a daily portfolio summary with all positions,
P&L details, total assets, and cash. Sent via email and Discord.
"""

import json
import logging
import os
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DailyReportGenerator:
    """Generates and sends daily P&L reports."""

    def __init__(self, output_base: str = "outputs"):
        self.output_base = Path(output_base)
        self.state_file = self.output_base / "daily_report_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> Dict:
        """Load report tracking state."""
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except Exception:
                pass
        return {"last_sent_date": None, "yesterday_value": None}

    def _save_state(self, state: Dict):
        """Save report tracking state."""
        self.state_file.write_text(json.dumps(state, indent=2))

    def should_send_today(self) -> bool:
        """Check if daily report should be sent today."""
        state = self._load_state()
        today = date.today().isoformat()
        return state.get("last_sent_date") != today

    def generate_and_send(self, portfolio, prices: Dict[str, float]) -> bool:
        """
        Generate daily report and send via email + Discord.

        Args:
            portfolio: MultiAssetPortfolio instance
            prices: Current prices dict {ticker: price}

        Returns:
            True if report was sent successfully
        """
        if not self.should_send_today():
            logger.info("Daily report already sent today, skipping")
            return False

        state = self._load_state()
        yesterday_value = state.get("yesterday_value")

        summary = portfolio.get_summary(prices)
        total_value = summary["total_value"]
        today_str = date.today().isoformat()

        # Calculate daily change
        daily_change = None
        daily_change_pct = None
        if yesterday_value and yesterday_value > 0:
            daily_change = total_value - yesterday_value
            daily_change_pct = (daily_change / yesterday_value) * 100

        # Build report data
        report = self._build_report(portfolio, prices, summary, daily_change, daily_change_pct)

        # Send via notification channels
        sent = False
        try:
            from notifications.notification_manager import NotificationManager
            nm = NotificationManager()

            # Send email
            if nm.config.email_enabled:
                html = self._build_html_report(report)
                subject = self._build_subject(report)
                email_sent = nm.send_email(subject, report["text"], html)
                if email_sent:
                    sent = True
                    logger.info("Daily report sent via email")

            # Send Discord
            if nm.config.discord_enabled:
                embed = self._build_discord_embed(report)
                discord_sent = nm.send_discord("", embed)
                if discord_sent:
                    sent = True
                    logger.info("Daily report sent via Discord")

        except Exception as e:
            logger.error(f"Failed to send daily report: {e}")

        # Update state
        state["last_sent_date"] = today_str
        state["yesterday_value"] = total_value
        self._save_state(state)

        return sent

    def _build_report(
        self,
        portfolio,
        prices: Dict[str, float],
        summary: Dict,
        daily_change: Optional[float],
        daily_change_pct: Optional[float],
    ) -> Dict:
        """Build report data structure."""
        positions = []
        for ticker, pos in portfolio.positions.items():
            current_price = prices.get(ticker, pos.entry_price)
            if pos.direction == "long":
                pnl = pos.quantity * (current_price - pos.entry_price)
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
            else:
                pnl = pos.quantity * (pos.entry_price - current_price)
                pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100

            positions.append({
                "ticker": ticker,
                "direction": pos.direction.upper(),
                "entry_price": pos.entry_price,
                "current_price": current_price,
                "quantity": pos.quantity,
                "position_value": pos.position_value,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "entry_date": pos.entry_date,
            })

        # Build text version
        text_lines = [
            f"QuantTrade Daily P&L Report - {date.today().isoformat()}",
            "=" * 50,
            "",
            f"Total Value:    ${summary['total_value']:,.2f}",
            f"Cash:           ${summary['cash']:,.2f}",
            f"Positions:      {summary['num_positions']}",
            f"Total Return:   {summary['total_return_pct']:+.2f}%",
        ]

        if daily_change is not None:
            text_lines.append(f"Daily Change:   ${daily_change:+,.2f} ({daily_change_pct:+.2f}%)")

        text_lines.extend(["", "Positions:", "-" * 50])

        for p in positions:
            text_lines.append(
                f"  {p['ticker']:8s} {p['direction']:5s}  "
                f"Entry: ${p['entry_price']:.2f}  "
                f"Now: ${p['current_price']:.2f}  "
                f"P&L: ${p['pnl']:+.2f} ({p['pnl_pct']:+.1f}%)"
            )

        if not positions:
            text_lines.append("  No open positions")

        # Closed trade stats
        closed_trades = len([t for t in portfolio.trades if t.action == "CLOSE"])
        winning = len([t for t in portfolio.trades if t.action == "CLOSE" and t.pnl > 0])
        text_lines.extend([
            "",
            f"Closed Trades: {closed_trades}  Winning: {winning}",
        ])

        return {
            "date": date.today().isoformat(),
            "total_value": summary["total_value"],
            "cash": summary["cash"],
            "total_return_pct": summary["total_return_pct"],
            "daily_change": daily_change,
            "daily_change_pct": daily_change_pct,
            "num_positions": summary["num_positions"],
            "positions": positions,
            "closed_trades": closed_trades,
            "winning_trades": winning,
            "text": "\n".join(text_lines),
        }

    def _build_subject(self, report: Dict) -> str:
        """Build email subject line."""
        val = report["total_value"]
        ret = report["total_return_pct"]
        arrow = "+" if ret >= 0 else ""
        daily = ""
        if report.get("daily_change_pct") is not None:
            d = report["daily_change_pct"]
            daily = f" | Day: {'+' if d >= 0 else ''}{d:.2f}%"
        return f"QuantTrade Daily Report | ${val:,.0f} ({arrow}{ret:.2f}%){daily}"

    def _build_html_report(self, report: Dict) -> str:
        """Build HTML email body."""
        # Header color based on daily performance
        header_color = "#198754" if (report.get("daily_change", 0) or 0) >= 0 else "#DC3545"

        position_rows = ""
        for p in report["positions"]:
            pnl_color = "#198754" if p["pnl"] >= 0 else "#DC3545"
            position_rows += f"""
            <tr>
                <td style="padding:8px;border-bottom:1px solid #333;">{p['ticker']}</td>
                <td style="padding:8px;border-bottom:1px solid #333;">{p['direction']}</td>
                <td style="padding:8px;border-bottom:1px solid #333;">${p['entry_price']:.2f}</td>
                <td style="padding:8px;border-bottom:1px solid #333;">${p['current_price']:.2f}</td>
                <td style="padding:8px;border-bottom:1px solid #333;color:{pnl_color};font-weight:bold;">
                    ${p['pnl']:+.2f} ({p['pnl_pct']:+.1f}%)
                </td>
                <td style="padding:8px;border-bottom:1px solid #333;">{p['entry_date']}</td>
            </tr>"""

        if not report["positions"]:
            position_rows = """
            <tr>
                <td colspan="6" style="padding:12px;text-align:center;color:#888;">No open positions</td>
            </tr>"""

        daily_section = ""
        if report.get("daily_change") is not None:
            dc = report["daily_change"]
            dcp = report["daily_change_pct"]
            dc_color = "#198754" if dc >= 0 else "#DC3545"
            daily_section = f"""
            <tr>
                <td style="padding:10px;border-bottom:1px solid #333;"><strong>Daily Change:</strong></td>
                <td style="padding:10px;border-bottom:1px solid #333;color:{dc_color};font-weight:bold;">
                    ${dc:+,.2f} ({dcp:+.2f}%)
                </td>
            </tr>"""

        return f"""
<html>
<body style="font-family:Arial,sans-serif;max-width:700px;margin:0 auto;">
    <div style="background-color:#1a1a2e;color:#fff;padding:20px;border-radius:10px;">
        <h1 style="color:{header_color};margin-top:0;">Daily P&L Report</h1>
        <p style="color:#aaa;">{report['date']}</p>

        <table style="width:100%;border-collapse:collapse;margin-bottom:20px;">
            <tr>
                <td style="padding:10px;border-bottom:1px solid #333;"><strong>Total Value:</strong></td>
                <td style="padding:10px;border-bottom:1px solid #333;font-size:18px;font-weight:bold;">
                    ${report['total_value']:,.2f}
                </td>
            </tr>
            <tr>
                <td style="padding:10px;border-bottom:1px solid #333;"><strong>Cash:</strong></td>
                <td style="padding:10px;border-bottom:1px solid #333;">${report['cash']:,.2f}</td>
            </tr>
            <tr>
                <td style="padding:10px;border-bottom:1px solid #333;"><strong>Total Return:</strong></td>
                <td style="padding:10px;border-bottom:1px solid #333;color:{header_color};font-weight:bold;">
                    {report['total_return_pct']:+.2f}%
                </td>
            </tr>
            {daily_section}
        </table>

        <h2 style="color:#0D6EFD;">Open Positions ({report['num_positions']})</h2>
        <table style="width:100%;border-collapse:collapse;">
            <tr style="background-color:#2a2a4e;">
                <th style="padding:8px;text-align:left;">Ticker</th>
                <th style="padding:8px;text-align:left;">Dir</th>
                <th style="padding:8px;text-align:left;">Entry</th>
                <th style="padding:8px;text-align:left;">Current</th>
                <th style="padding:8px;text-align:left;">P&L</th>
                <th style="padding:8px;text-align:left;">Opened</th>
            </tr>
            {position_rows}
        </table>

        <p style="margin-top:20px;color:#aaa;">
            Closed Trades: {report['closed_trades']} |
            Winning: {report['winning_trades']}
        </p>

        <p style="margin-top:20px;font-size:12px;color:#666;">
            Generated by QuantTrade Satellite Trading System<br>
            {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </p>
    </div>
</body>
</html>"""

    def _build_discord_embed(self, report: Dict) -> Dict:
        """Build Discord embed for daily report."""
        color = 1611076 if (report.get("daily_change", 0) or 0) >= 0 else 14370848

        fields = [
            {"name": "Total Value", "value": f"${report['total_value']:,.2f}", "inline": True},
            {"name": "Cash", "value": f"${report['cash']:,.2f}", "inline": True},
            {"name": "Return", "value": f"{report['total_return_pct']:+.2f}%", "inline": True},
        ]

        if report.get("daily_change") is not None:
            fields.append({
                "name": "Daily Change",
                "value": f"${report['daily_change']:+,.2f} ({report['daily_change_pct']:+.2f}%)",
                "inline": True,
            })

        # Add position summary
        if report["positions"]:
            pos_lines = []
            for p in report["positions"]:
                emoji = "+" if p["pnl"] >= 0 else ""
                pos_lines.append(
                    f"`{p['ticker']:5s}` {p['direction']} ${p['current_price']:.2f} "
                    f"P&L: {emoji}{p['pnl_pct']:.1f}%"
                )
            fields.append({
                "name": f"Positions ({report['num_positions']})",
                "value": "\n".join(pos_lines),
                "inline": False,
            })

        return {
            "title": f"Daily P&L Report - {report['date']}",
            "color": color,
            "fields": fields,
            "footer": {"text": "QuantTrade Satellite Trading System"},
            "timestamp": datetime.utcnow().isoformat(),
        }
