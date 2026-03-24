"""
Trading Signal Notifications

Sends email/SMS notifications when actionable trading signals are detected.
Supports multiple recipients and customizable alert thresholds.
"""

import json
import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class NotificationConfig:
    """Configuration for notifications."""

    def __init__(self):
        self.smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.smtp_username = os.environ.get("SMTP_USERNAME", "")
        self.smtp_password = os.environ.get("SMTP_PASSWORD", "")
        self.from_email = os.environ.get("NOTIFICATION_FROM", self.smtp_username)
        self.to_emails = self._parse_recipients(os.environ.get("NOTIFICATION_TO", ""))
        self.sms_gateway = os.environ.get("SMS_GATEWAY", "")  # e.g., "number@vtext.com"

    def _parse_recipients(self, recipients: str) -> List[str]:
        """Parse comma-separated email list."""
        if not recipients:
            return []
        return [e.strip() for e in recipients.split(",") if e.strip()]

    def is_configured(self) -> bool:
        """Check if email is configured."""
        return bool(self.smtp_username and self.smtp_password and self.to_emails)


class SignalNotifier:
    """Send notifications for trading signals."""

    def __init__(self, config: Optional[NotificationConfig] = None):
        self.config = config or NotificationConfig()
        self.output_dir = Path("outputs")

    def get_actionable_signals(self, date: Optional[str] = None) -> List[Dict]:
        """
        Get actionable trading signals from daily summary.

        Args:
            date: Date string (YYYY-MM-DD), defaults to today

        Returns:
            List of actionable signal dictionaries
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        summary_file = self.output_dir / date / "daily_summary.json"

        if not summary_file.exists():
            logger.warning(f"No summary file found for {date}")
            return []

        try:
            with open(summary_file) as f:
                summary = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read summary: {e}")
            return []

        actionable = []
        for region_id, signal in summary.get("signals", {}).items():
            if signal.get("actionability") == "Actionable":
                actionable.append({
                    "region_id": region_id,
                    "signal": signal.get("signal"),
                    "direction": signal.get("trading_action"),
                    "confidence": signal.get("confidence"),
                    "instruments": signal.get("instruments", []),
                    "type": signal.get("type"),
                })

        return actionable

    def format_email_body(self, signals: List[Dict], date: str) -> str:
        """Format signals into HTML email body."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .header {{ background: #1a1a1a; color: white; padding: 20px; }}
        .content {{ padding: 20px; }}
        .signal {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .signal.long {{ border-left: 5px solid #22c55e; }}
        .signal.short {{ border-left: 5px solid #ef4444; }}
        .signal h3 {{ margin: 0 0 10px 0; }}
        .instruments {{ background: #f3f4f6; padding: 5px 10px; border-radius: 3px; display: inline-block; }}
        .footer {{ background: #f3f4f6; padding: 15px; text-align: center; font-size: 12px; color: #666; }}
        .emoji {{ font-size: 24px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛰️ QuantTrade 交易信号通知</h1>
        <p>日期: {date} | 可操作信号数量: {len(signals)}</p>
    </div>
    <div class="content">
"""

        if not signals:
            html += "<p>今天没有可操作的交易信号。</p>"
        else:
            for sig in signals:
                direction_emoji = "📈" if sig["direction"] == "LONG" else "📉"
                direction_class = sig["direction"].lower()

                html += f"""
        <div class="signal {direction_class}">
            <h3>{direction_emoji} {sig['region_id'].replace('_', ' ').title()}</h3>
            <p><strong>信号:</strong> {sig['signal']}</p>
            <p><strong>方向:</strong> {sig['direction']}</p>
            <p><strong>信心:</strong> {sig['confidence']}</p>
            <p><strong>类型:</strong> {sig['type']}</p>
            <p><strong>交易品种:</strong> <span class="instruments">{', '.join(sig['instruments'])}</span></p>
        </div>
"""

        html += f"""
    </div>
    <div class="footer">
        <p>此邮件由 QuantTrade 自动生成，请勿回复。</p>
        <p>如需调整通知设置，请检查您的配置文件。</p>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
"""
        return html

    def send_email(self, subject: str, body: str, attach_summary: bool = False) -> bool:
        """
        Send email notification.

        Args:
            subject: Email subject
            body: HTML email body
            attach_summary: Whether to attach the full summary JSON

        Returns:
            True if sent successfully
        """
        if not self.config.is_configured():
            logger.warning("Email not configured - skipping notification")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.config.from_email
            msg["To"] = ", ".join(self.config.to_emails)

            # Attach HTML body
            html_part = MIMEText(body, "html", "utf-8")
            msg.attach(html_part)

            # Optionally attach summary JSON
            if attach_summary:
                date = datetime.now().strftime("%Y-%m-%d")
                summary_file = self.output_dir / date / "daily_summary.json"
                if summary_file.exists():
                    with open(summary_file, "rb") as f:
                        attachment = MIMEApplication(f.read(), Name="daily_summary.json")
                    attachment["Content-Disposition"] = f'attachment; filename="daily_summary_{date}.json"'
                    msg.attach(attachment)

            # Send email
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(msg)

            logger.info(f"Email sent to {len(self.config.to_emails)} recipients")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def send_sms(self, signals: List[Dict], date: str) -> bool:
        """
        Send SMS notification via email-to-SMS gateway.

        Args:
            signals: List of actionable signals
            date: Date string

        Returns:
            True if sent successfully
        """
        if not self.config.sms_gateway:
            logger.debug("SMS gateway not configured - skipping SMS")
            return False

        if not self.config.smtp_username or not self.config.smtp_password:
            logger.warning("SMTP not configured - cannot send SMS")
            return False

        try:
            lines = [f"QuantTrade {date}", f"{len(signals)} signals:"]

            for sig in signals[:5]:
                direction = "LONG" if sig["direction"] == "LONG" else "SHORT" if sig["direction"] == "SHORT" else "FLAT"
                lines.append(f"- {sig.get('region_id', 'unknown')}: {direction}")

            if len(signals) > 5:
                lines.append(f"... and {len(signals) - 5} more")

            body = "\n".join(lines)

            msg = MIMEMultipart()
            msg["Subject"] = f"QuantTrade: {len(signals)} signals"
            msg["From"] = self.config.from_email
            msg["To"] = self.config.sms_gateway

            text_part = MIMEText(body, "plain", "utf-8")
            msg.attach(text_part)

            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(msg)

            logger.info(f"SMS sent to {self.config.sms_gateway}")
            return True

        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return False

    def check_and_notify(self, date: Optional[str] = None) -> Dict:
        """
        Check for actionable signals and send notification if found.

        Args:
            date: Date to check (YYYY-MM-DD), defaults to today

        Returns:
            Dictionary with notification results
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        signals = self.get_actionable_signals(date)

        result = {
            "date": date,
            "actionable_count": len(signals),
            "notified": False,
            "recipients": self.config.to_emails,
            "signals": signals,
        }

        if signals:
            subject = f"🛰️ QuantTrade: {len(signals)} 个可操作信号 ({date})"
            body = self.format_email_body(signals, date)

            email_sent = self.send_email(subject, body, attach_summary=True)
            sms_sent = self.send_sms(signals, date)

            if email_sent or sms_sent:
                result["notified"] = True
                result["email_sent"] = email_sent
                result["sms_sent"] = sms_sent
                result["message"] = f"Sent notification for {len(signals)} signals (email={email_sent}, sms={sms_sent})"
            else:
                result["message"] = "Failed to send notification"
        else:
            result["message"] = "No actionable signals - no notification sent"

        # Save notification log
        log_file = self.output_dir / date / "notification_log.json"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))

        return result


def send_notification_on_signals(date: Optional[str] = None) -> Dict:
    """
    Convenience function to send notifications for actionable signals.

    Args:
        date: Date to check (YYYY-MM-DD), defaults to today

    Returns:
        Notification result dictionary
    """
    notifier = SignalNotifier()
    return notifier.check_and_notify(date)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("🛰️ QuantTrade 信号通知系统")
    print("=" * 50)

    # Test notification
    result = send_notification_on_signals()

    print(f"日期: {result['date']}")
    print(f"可操作信号数量: {result['actionable_count']}")
    print(f"通知状态: {'✅ 已发送' if result['notified'] else '⏭️ 无需发送'}")
    print(f"收件人: {', '.join(result['recipients']) or '未配置'}")
    print(f"消息: {result['message']}")
