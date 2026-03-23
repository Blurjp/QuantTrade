"""
Notifications Package

Provides notification capabilities for trading signals:
- Email notifications (SMTP)
- SMS notifications (Twilio)
- Discord webhook notifications
- Signal monitoring and alerting
"""

from notifications.notification_manager import (
    NotificationConfig,
    NotificationManager,
)
from notifications.signal_monitor import SignalMonitor

__all__ = [
    "NotificationConfig",
    "NotificationManager",
    "SignalMonitor",
]
