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


def main():
    """Run signal monitor on schedule."""
    # Get interval from environment
    interval_minutes = int(os.getenv("PIPELINE_INTERVAL_MINUTES", "30"))
    
    logger.info("=" * 60)
    logger.info("QuantTrade Signal Monitor")
    logger.info("=" * 60)
    logger.info(f"Check interval: Every {interval_minutes} minutes")
    logger.info(f"Email notifications: {os.getenv('EMAIL_ENABLED', 'false')}")
    logger.info(f"SMS notifications: {os.getenv('SMS_ENABLED', 'false')}")
    logger.info(f"Quiet hours: {os.getenv('QUIET_HOURS_START', '23')}:00 - {os.getenv('QUIET_HOURS_END', '7')}:00")
    logger.info("=" * 60)
    
    # Run immediately on startup
    logger.info("\n🔍 Running initial check...")
    check_signals()
    
    # Schedule periodic checks
    schedule.every(interval_minutes).minutes.do(check_signals)
    logger.info(f"\n⏰ Scheduled to run every {interval_minutes} minutes")
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
