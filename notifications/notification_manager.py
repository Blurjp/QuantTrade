"""
Trading Signal Notification System

Sends SMS and email alerts when trading signals are detected.
Supports multiple notification channels:
- Email (via SMTP or SendGrid)
- SMS (via Twilio)
- Discord (via webhook)
- Push notifications (optional)
"""

import json
import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


class NotificationConfig:
    """Notification configuration from environment variables."""
    
    def __init__(self):
        # Email settings (SendGrid preferred, SMTP fallback)
        self.email_enabled = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
        self.sendgrid_api_key = os.getenv("SENDGRID_API_KEY", "")
        self.email_from = os.getenv("EMAIL_FROM", "quanttrade@railway.app")
        self.email_to = os.getenv("EMAIL_TO", "").split(",")

        self.sendgrid_from = os.getenv("SENDGRID_FROM", "quanttrade@noreply.com")
        # SMTP fallback
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        
        # SMS settings (Twilio)
        self.sms_enabled = os.getenv("SMS_ENABLED", "false").lower() == "true"
        self.twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.twilio_phone_from = os.getenv("TWILIO_PHONE_FROM", "")
        self.twilio_phone_to = os.getenv("TWILIO_PHONE_TO", "").split(",")
        
        # Discord webhook
        self.discord_enabled = os.getenv("DISCORD_ENABLED", "false").lower() == "true"
        self.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
        
        # Notification thresholds
        self.min_confidence = float(os.getenv("NOTIFICATION_MIN_CONFIDENCE", "70.0"))
        self.min_impact_score = float(os.getenv("NOTIFICATION_MIN_IMPACT", "50.0"))
        
        # Quiet hours (no notifications)
        self.quiet_hours_start = int(os.getenv("QUIET_HOURS_START", "23"))  # 11 PM
        self.quiet_hours_end = int(os.getenv("QUIET_HOURS_END", "7"))  # 7 AM
        self.notification_timezone = os.getenv("NOTIFICATION_TIMEZONE", "America/New_York")
        
        self._log_status()
    
    def _log_status(self):
        """Log notification configuration status."""
        logger.info("=" * 60)
        logger.info("Notification Configuration")
        logger.info("=" * 60)
        logger.info(f"Email: {'✅ Enabled' if self.email_enabled else '❌ Disabled'}")
        if self.email_enabled:
            if self.sendgrid_api_key:
                logger.info(f"  Provider: SendGrid")
            else:
                logger.info(f"  Provider: SMTP ({self.smtp_host}:{self.smtp_port})")
            logger.info(f"  From: {self.email_from or self.sendgrid_from}")
            logger.info(f"  To: {len(self.email_to)} recipients")
        
        logger.info(f"SMS: {'✅ Enabled' if self.sms_enabled else '❌ Disabled'}")
        if self.sms_enabled:
            logger.info(f"  Twilio SID: {self.twilio_account_sid[:10]}...")
            logger.info(f"  To: {len(self.twilio_phone_to)} numbers")
        
        logger.info(f"Discord: {'✅ Enabled' if self.discord_enabled else '❌ Disabled'}")
        
        logger.info(f"Min Confidence: {self.min_confidence}%")
        logger.info(f"Min Impact Score: {self.min_impact_score}")
        logger.info(f"Quiet Hours: {self.quiet_hours_start}:00 - {self.quiet_hours_end}:00")
        logger.info(f"Notification Timezone: {self.notification_timezone}")
        logger.info("=" * 60)
    
    def is_quiet_hours(self) -> bool:
        """Check if current time is in quiet hours."""
        try:
            hour = datetime.now(ZoneInfo(self.notification_timezone)).hour
        except Exception:
            hour = datetime.now().hour
        if self.quiet_hours_start > self.quiet_hours_end:
            # Spans midnight (e.g., 23:00 - 07:00)
            return hour >= self.quiet_hours_start or hour < self.quiet_hours_end
        else:
            return self.quiet_hours_start <= hour < self.quiet_hours_end


class NotificationManager:
    """Manages all notification channels."""
    
    def __init__(self, config: Optional[NotificationConfig] = None):
        self.config = config or NotificationConfig()
        self.notification_history_file = Path("outputs/notification_history.json")
        self.notification_history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load notification history
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """Load notification history from file."""
        if self.notification_history_file.exists():
            try:
                with open(self.notification_history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load notification history: {e}")
        return []
    
    def _save_history(self):
        """Save notification history to file."""
        try:
            with open(self.notification_history_file, 'w') as f:
                json.dump(self.history[-100:], f, indent=2)  # Keep last 100
        except Exception as e:
            logger.warning(f"Failed to save notification history: {e}")
    
    def should_notify(self, signal: Dict) -> bool:
        """Check if signal meets notification criteria."""
        # Check confidence
        confidence = float(signal.get("confidence", 0))
        if confidence < self.config.min_confidence:
            logger.info(f"Signal confidence {confidence}% below threshold {self.config.min_confidence}%")
            return False
        
        # Check impact score if available
        impact_score = signal.get("impact_score", 0)
        if impact_score > 0 and impact_score < self.config.min_impact_score:
            logger.info(f"Signal impact {impact_score} below threshold {self.config.min_impact_score}")
            return False
        
        # Check quiet hours
        if self.config.is_quiet_hours():
            logger.info("Current time is in quiet hours - skipping notification")
            return False
        
        # Check if already notified for this signal
        signal_id = f"{signal.get('signal_type')}_{signal.get('region_id', signal.get('facility_id', ''))}_{signal.get('date', '')}"
        recent_notifications = [n for n in self.history[-20:] if n.get("signal_id") == signal_id]
        if recent_notifications:
            logger.info(f"Already notified for signal {signal_id}")
            return False
        
        return True
    
    def send_email(self, subject: str, body: str, html_body: Optional[str] = None) -> bool:
        """Send email notification. Prefers SendGrid, falls back to SMTP."""
        if not self.config.email_enabled:
            logger.debug("Email notifications disabled")
            return False
        
        if not self.config.email_to:
            logger.warning("No email recipients configured")
            return False
        
        # Try SendGrid first
        if self.config.sendgrid_api_key:
            try:
                from sendgrid import SendGridAPIClient
                from sendgrid.helpers.mail import Mail
                
                message = Mail(
                    from_email=self.config.sendgrid_from,
                    to_emails=self.config.email_to,
                    subject=subject,
                    html_content=html_body or f"<pre>{body}</pre>",
                )
                
                sg = SendGridAPIClient(self.config.sendgrid_api_key)
                response = sg.send(message)
                
                if response.status_code in (200, 201, 202):
                    logger.info(f"✅ Email sent via SendGrid: {subject}")
                    return True
                else:
                    logger.warning(f"SendGrid returned {response.status_code}, falling back to SMTP")
                    
            except ImportError:
                logger.warning("sendgrid package not installed, falling back to SMTP")
            except Exception as e:
                logger.warning(f"SendGrid failed: {e}, falling back to SMTP")
        
        # Fallback to SMTP
        try:
            msg = MIMEMultipart("alternative")
            msg["From"] = self.config.email_from
            msg["To"] = ", ".join(self.config.email_to)
            msg["Subject"] = subject
            
            msg.attach(MIMEText(body, "plain"))
            
            if html_body:
                msg.attach(MIMEText(html_body, "html"))
            
            try:
                with smtplib.SMTP_SSL(self.config.smtp_host, self.config.smtp_port, timeout=10) as server:
                    server.login(self.config.smtp_username, self.config.smtp_password)
                    server.sendmail(self.config.email_from, self.config.email_to, msg.as_string())
                logger.info(f"✅ Email sent via SSL: {subject}")
                return True
            except Exception as ssl_error:
                logger.warning(f"SSL failed, trying TLS: {ssl_error}")
                with smtplib.SMTP(self.config.smtp_host, 587, timeout=10) as server:
                    server.starttls()
                    server.login(self.config.smtp_username, self.config.smtp_password)
                    server.sendmail(self.config.email_from, self.config.email_to, msg.as_string())
                logger.info(f"✅ Email sent via TLS: {subject}")
                return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def send_sms(self, message: str) -> bool:
        """Send SMS notification via Twilio."""
        if not self.config.sms_enabled:
            logger.debug("SMS notifications disabled")
            return False
        
        if not self.config.twilio_phone_to:
            logger.warning("No SMS recipients configured")
            return False
        
        try:
            from twilio.rest import Client
            
            client = Client(
                self.config.twilio_account_sid,
                self.config.twilio_auth_token
            )
            
            for phone_to in self.config.twilio_phone_to:
                if phone_to:
                    client.messages.create(
                        body=message,
                        from_=self.config.twilio_phone_from,
                        to=phone_to
                    )
            
            logger.info(f"✅ SMS sent to {len(self.config.twilio_phone_to)} recipients")
            return True
            
        except ImportError:
            logger.error("Twilio library not installed. Run: pip install twilio")
            return False
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return False
    
    def send_discord(self, message: str, embed: Optional[Dict] = None) -> bool:
        """Send Discord notification via webhook."""
        if not self.config.discord_enabled:
            logger.debug("Discord notifications disabled")
            return False
        
        if not self.config.discord_webhook_url:
            logger.warning("Discord webhook URL not configured")
            return False
        
        try:
            import requests
            
            data = {"content": message}
            if embed:
                data["embeds"] = [embed]
            
            response = requests.post(
                self.config.discord_webhook_url,
                json=data,
                timeout=10
            )
            
            if response.status_code == 204:
                logger.info("✅ Discord notification sent")
                return True
            else:
                logger.error(f"Discord webhook failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False
    
    def notify_signal(self, signal: Dict) -> Dict[str, bool]:
        """
        Send notifications for a trading signal.
        
        Returns dict of notification channel -> success status.
        """
        if not self.should_notify(signal):
            logger.info("Signal does not meet notification criteria")
            return {}
        
        # Prepare notification content
        signal_type = signal.get("signal_type", "Unknown")
        region = signal.get("region_name", signal.get("facility_name", "Unknown"))
        direction = signal.get("direction", "NEUTRAL")
        confidence = signal.get("confidence", 0)
        rationale = signal.get("rationale", "No rationale provided")
        instruments = signal.get("instruments", [])
        date = signal.get("date", datetime.now().strftime("%Y-%m-%d"))
        
        # Format instruments
        instruments_str = ", ".join(instruments) if instruments else "N/A"
        
        # Create subject/title
        emoji = "🟢" if direction == "long" else "🔴" if direction == "short" else "⚪"
        subject = f"{emoji} Trading Signal: {direction.upper()} {instruments_str}"
        title = f"{emoji} {signal_type.replace('_', ' ').title()} Signal"
        
        # Create message body
        body = f"""
Trading Signal Alert
====================

Signal Type: {signal_type}
Region/Facility: {region}
Direction: {direction.upper()}
Confidence: {confidence:.1f}%
Date: {date}

Instruments: {instruments_str}

Rationale:
{rationale}

---
Generated by QuantTrade Satellite Monitoring System
""".strip()
        
        # Create HTML body
        direction_color = "#00ff00" if direction == "long" else "#ff0000" if direction == "short" else "#888888"
        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <div style="background-color: #1a1a2e; color: #ffffff; padding: 20px; border-radius: 10px;">
        <h1 style="color: {direction_color}; margin-top: 0;">{title}</h1>
        
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #333;"><strong>Signal Type:</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #333;">{signal_type}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #333;"><strong>Region:</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #333;">{region}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #333;"><strong>Direction:</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #333; color: {direction_color}; font-weight: bold;">{direction.upper()}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #333;"><strong>Confidence:</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #333;">{confidence:.1f}%</td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #333;"><strong>Instruments:</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #333;">{instruments_str}</td>
            </tr>
        </table>
        
        <h3 style="margin-top: 20px;">Rationale:</h3>
        <p style="background-color: #2a2a4e; padding: 15px; border-radius: 5px;">{rationale}</p>
        
        <p style="margin-top: 20px; font-size: 12px; color: #888;">
            Generated by QuantTrade Satellite Monitoring System<br>
            {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </p>
    </div>
</body>
</html>
"""
        
        # Create SMS message (shorter)
        sms_message = f"{emoji} {signal_type.upper()}: {direction.upper()} {instruments_str} ({confidence:.0f}% confidence) - {rationale[:100]}"
        
        # Create Discord embed
        discord_embed = {
            "title": title,
            "description": rationale,
            "color": 3066993 if direction == "long" else 15158332 if direction == "short" else 9807270,
            "fields": [
                {"name": "Region", "value": region, "inline": True},
                {"name": "Direction", "value": direction.upper(), "inline": True},
                {"name": "Confidence", "value": f"{confidence:.1f}%", "inline": True},
                {"name": "Instruments", "value": instruments_str, "inline": False},
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "QuantTrade Satellite Monitoring"}
        }
        
        # Send notifications
        results = {}
        
        # Email
        if self.config.email_enabled:
            results["email"] = self.send_email(subject, body, html_body)
        
        # SMS
        if self.config.sms_enabled:
            results["sms"] = self.send_sms(sms_message)
        
        # Discord
        if self.config.discord_enabled:
            results["discord"] = self.send_discord("", discord_embed)
        
        # Log to history
        if any(results.values()):
            signal_id = f"{signal_type}_{signal.get('region_id', signal.get('facility_id', ''))}_{date}"
            self.history.append({
                "signal_id": signal_id,
                "timestamp": datetime.now().isoformat(),
                "channels": results,
                "signal_summary": {
                    "type": signal_type,
                    "direction": direction,
                    "confidence": confidence,
                    "instruments": instruments
                }
            })
            self._save_history()
        
        return results


def main():
    """Test notification system."""
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 60)
    print("Trading Signal Notification System")
    print("=" * 60)
    
    # Initialize notification manager
    manager = NotificationManager()
    
    # Test signal
    test_signal = {
        "signal_type": "precipitation",
        "region_id": "usa_midwest",
        "region_name": "US Midwest Corn Belt",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "direction": "short",
        "confidence": 78.5,
        "impact_score": 65.0,
        "instruments": ["CORN", "SOYB"],
        "rationale": "Severe precipitation deficit in US Midwest Corn Belt. -45.2% below normal. Crop stress likely. Historical yield impact: -8% to -15%.",
    }
    
    print(f"\nTest Signal:")
    print(f"  Type: {test_signal['signal_type']}")
    print(f"  Region: {test_signal['region_name']}")
    print(f"  Direction: {test_signal['direction']}")
    print(f"  Confidence: {test_signal['confidence']}%")
    print(f"  Instruments: {test_signal['instruments']}")
    
    print(f"\nSending notifications...")
    results = manager.notify_signal(test_signal)
    
    if results:
        print("\nNotification Results:")
        for channel, success in results.items():
            status = "✅ Sent" if success else "❌ Failed"
            print(f"  {channel.title()}: {status}")
    else:
        print("\n⚠️  No notifications sent (check configuration or signal criteria)")
    
    print("\n" + "=" * 60)
    print("Configuration Guide")
    print("=" * 60)
    print("""
To enable notifications, add these to your .env file:

# Email (Gmail example)
EMAIL_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
EMAIL_FROM=your-email@gmail.com
EMAIL_TO=recipient1@gmail.com,recipient2@gmail.com

# SMS (Twilio)
SMS_ENABLED=true
TWILIO_ACCOUNT_SID=your-account-sid
TWILIO_AUTH_TOKEN=your-auth-token
TWILIO_PHONE_FROM=+1234567890
TWILIO_PHONE_TO=+0987654321

# Discord Webhook
DISCORD_ENABLED=true
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Notification Thresholds
NOTIFICATION_MIN_CONFIDENCE=70.0
NOTIFICATION_MIN_IMPACT=50.0

# Quiet Hours (no notifications)
QUIET_HOURS_START=23
QUIET_HOURS_END=7
    """)


if __name__ == "__main__":
    main()
