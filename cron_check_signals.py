#!/usr/bin/env python3
"""
Railway Cron Job Script for Signal Monitoring

This script is designed to run as a Railway cron job.
It checks all signals and sends notifications.

Usage:
  python cron_check_signals.py

Railway Cron Schedule:
  */30 7-23 * * *  # Every 30 minutes between 7 AM and 11 PM
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def is_monitoring_hours() -> bool:
    """Check if current time is within monitoring hours (EST/EDT)."""
    # Get current hour in Eastern Time
    from datetime import timezone
    import pytz
    
    try:
        et = pytz.timezone('America/New_York')
        now_et = datetime.now(et)
        hour = now_et.hour
    except (pytz.exceptions.UnknownTimeZoneError, Exception) as e:
        logger.warning("Failed to get ET timezone: %s, using system time", e)
        hour = datetime.now().hour
    
    start_hour = int(os.getenv("MONITORING_START_HOUR", "7"))
    end_hour = int(os.getenv("MONITORING_END_HOUR", "24"))
    
    return start_hour <= hour < end_hour


def check_signals():
    """Check all signals and send notifications."""
    logger.info("=" * 60)
    logger.info(f"Railway Cron Job - Signal Check Started")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # Check monitoring hours
    if not is_monitoring_hours():
        logger.info("🔕 Outside monitoring hours - skipping check")
        return {"status": "skipped", "reason": "outside_monitoring_hours"}
    
    try:
        from notifications.signal_monitor import SignalMonitor
        
        monitor = SignalMonitor()
        summary = monitor.check_all_modules()
        
        logger.info(f"Total signals: {summary['total_signals']}")
        logger.info(f"Actionable: {summary['actionable_signals']}")
        logger.info(f"Notifications sent: {summary['notified']}")
        
        if summary['notifications']:
            logger.info("\nNotifications sent:")
            for notif in summary['notifications']:
                logger.info(f"  📧 {notif['signal']}")
                for channel, success in notif['channels'].items():
                    status = "✅" if success else "❌"
                    logger.info(f"    {status} {channel}")
        
        logger.info("=" * 60)
        logger.info("Signal Check Complete")
        logger.info("=" * 60)
        
        return summary
        
    except Exception as e:
        logger.error(f"Signal check failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "error": str(e)}


def main():
    """Main entry point for Railway cron job."""
    logger.info("Starting Railway cron job...")
    
    # Log configuration
    logger.info(f"Email enabled: {os.getenv('EMAIL_ENABLED', 'false')}")
    logger.info(f"SMS enabled: {os.getenv('SMS_ENABLED', 'false')}")
    logger.info(f"Monitoring hours: {os.getenv('MONITORING_START_HOUR', '7')}:00 - {os.getenv('MONITORING_END_HOUR', '24')}:00")
    
    # Run check
    result = check_signals()
    
    # Exit with appropriate code
    if result.get("status") == "error":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
