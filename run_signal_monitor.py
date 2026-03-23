#!/usr/bin/env python3
"""
Standalone Signal Monitor with Schedule

Runs signal monitoring every X minutes and sends notifications.
"""

import os
import sys
import time
import logging
import schedule
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / '.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_signals():
    """Check all signals and send notifications."""
    logger.info("=" * 60)
    logger.info(f"Signal Check Started - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    try:
        from notifications.signal_monitor import SignalMonitor
        
        monitor = SignalMonitor()
        summary = monitor.check_all_modules()
        
        logger.info(f"Total signals: {summary['total_signals']}")
        logger.info(f"Actionable: {summary['actionable_signals']}")
        logger.info(f"Notifications sent: {summary['notified']}")
        
        if summary['notifications']:
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
        return None


def is_monitoring_hours():
    """Check if current time is within monitoring hours."""
    hour = datetime.now().hour
    start_hour = int(os.getenv("MONITORING_START_HOUR", "7"))
    end_hour = int(os.getenv("MONITORING_END_HOUR", "24"))
    
    # Handle midnight crossover (e.g., 7:00 - 24:00)
    if start_hour <= end_hour:
        return start_hour <= hour < end_hour
    else:
        return hour >= start_hour or hour < end_hour


def scheduled_check():
    """Scheduled check that respects monitoring hours."""
    if is_monitoring_hours():
        logger.info("✅ Within monitoring hours - running check")
        check_signals()
    else:
        logger.info("🔕 Outside monitoring hours - skipping check (quiet time)")


def main():
    """Run signal monitor on schedule."""
    # Get interval from environment
    interval_minutes = int(os.getenv("PIPELINE_INTERVAL_MINUTES", "30"))
    start_hour = int(os.getenv("MONITORING_START_HOUR", "7"))
    end_hour = int(os.getenv("MONITORING_END_HOUR", "24"))
    
    logger.info("=" * 60)
    logger.info("QuantTrade Signal Monitor")
    logger.info("=" * 60)
    logger.info(f"Check interval: Every {interval_minutes} minutes")
    logger.info(f"Monitoring hours: {start_hour}:00 - {end_hour}:00")
    logger.info(f"Email notifications: {os.getenv('EMAIL_ENABLED', 'false')}")
    logger.info(f"SMS notifications: {os.getenv('SMS_ENABLED', 'false')}")
    logger.info("=" * 60)
    
    # Check if we're in monitoring hours
    if is_monitoring_hours():
        logger.info("\n🔍 Running initial check (within monitoring hours)...")
        check_signals()
    else:
        logger.info(f"\n🔕 Outside monitoring hours - waiting until {start_hour}:00")
    
    # Schedule periodic checks
    schedule.every(interval_minutes).minutes.do(scheduled_check)
    logger.info(f"\n⏰ Scheduled to run every {interval_minutes} minutes")
    logger.info(f"   Active hours: {start_hour}:00 - {end_hour}:00")
    logger.info("Press Ctrl+C to stop\n")
    
    # Run forever
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n👋 Signal monitor stopped by user")
        sys.exit(0)
