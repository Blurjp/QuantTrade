"""
Signal Monitor - Watches for actionable signals and sends notifications

This module integrates with the signal generation system and sends
notifications when high-confidence actionable signals are detected.

Usage:
    from notifications.signal_monitor import SignalMonitor
    
    monitor = SignalMonitor()
    monitor.check_and_notify(latest_signals)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from notifications.notification_manager import NotificationManager, NotificationConfig

logger = logging.getLogger(__name__)


class SignalMonitor:
    """Monitors signals and sends notifications for actionable trades."""
    
    def __init__(
        self,
        notification_config: Optional[NotificationConfig] = None,
        history_file: str = "outputs/notification_state.json"
    ):
        """
        Initialize signal monitor.
        
        Args:
            notification_config: Notification configuration
            history_file: File to track notified signals
        """
        self.notification_manager = NotificationManager(notification_config)
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load notification state
        self.notified_signals = self._load_state()
    
    def _load_state(self) -> Dict:
        """Load notification state from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load notification state: {e}")
        return {"notified": [], "last_check": None}
    
    def _save_state(self):
        """Save notification state to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.notified_signals, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save notification state: {e}")
    
    def _get_signal_key(self, signal: Dict) -> str:
        """Generate unique key for a signal."""
        return f"{signal.get('signal_type', 'unknown')}_{signal.get('region_id', signal.get('facility_id', 'unknown'))}_{signal.get('date', datetime.now().strftime('%Y-%m-%d'))}"
    
    def is_actionable(self, signal: Dict) -> bool:
        """Check if signal is actionable."""
        # Check direction
        direction = signal.get("direction", "").lower()
        if direction not in ["long", "short"]:
            return False
        
        # Check confidence
        confidence = signal.get("confidence", 0)
        if confidence < self.notification_manager.config.min_confidence:
            return False
        
        # Check impact score if available
        impact_score = signal.get("impact_score", 0)
        if impact_score > 0 and impact_score < self.notification_manager.config.min_impact_score:
            return False
        
        # Check if already notified
        signal_key = self._get_signal_key(signal)
        if signal_key in self.notified_signals.get("notified", []):
            logger.debug(f"Signal {signal_key} already notified")
            return False
        
        return True
    
    def check_and_notify(self, signals: List[Dict]) -> Dict:
        """
        Check signals and send notifications for actionable ones.
        
        Args:
            signals: List of signal dictionaries
            
        Returns:
            Summary of notifications sent
        """
        summary = {
            "total_signals": len(signals),
            "actionable_signals": 0,
            "notified": 0,
            "notifications": []
        }
        
        for signal in signals:
            if self.is_actionable(signal):
                summary["actionable_signals"] += 1
                
                # Send notification
                results = self.notification_manager.notify_signal(signal)
                
                if any(results.values()):
                    summary["notified"] += 1
                    summary["notifications"].append({
                        "signal": self._get_signal_key(signal),
                        "channels": results
                    })
                    
                    # Mark as notified
                    signal_key = self._get_signal_key(signal)
                    self.notified_signals["notified"].append(signal_key)
                    
                    logger.info(f"✅ Notified for signal: {signal_key}")
                else:
                    logger.warning(f"⚠️  Failed to notify for signal: {self._get_signal_key(signal)}")
        
        # Update last check time
        self.notified_signals["last_check"] = datetime.now().isoformat()
        
        # Keep only last 1000 notified signals
        self.notified_signals["notified"] = self.notified_signals["notified"][-1000:]
        
        # Save state
        self._save_state()
        
        return summary
    
    def check_all_modules(self) -> Dict:
        """
        Check all monitoring modules for actionable signals.
        
        Returns:
            Summary of all notifications sent
        """
        all_signals = []
        
        # Check precipitation signals
        try:
            from pipeline.precipitation import PrecipitationMonitor
            monitor = PrecipitationMonitor()
            if not hasattr(monitor, "generate_all_signals"):
                raise AttributeError("PrecipitationMonitor missing generate_all_signals()")
            signals = monitor.generate_all_signals()
            all_signals.extend(signals)
            logger.info(f"Checked precipitation: {len(signals)} signals")
        except Exception as e:
            logger.error(f"Failed to check precipitation: {e}")
        
        try:
            from pipeline.sea_surface_temperature import SeaSurfaceTemperatureMonitor
            monitor = SeaSurfaceTemperatureMonitor()
            if not hasattr(monitor, "generate_all_signals"):
                raise AttributeError("SeaSurfaceTemperatureMonitor missing generate_all_signals()")
            signals = monitor.generate_all_signals()
            all_signals.extend(signals)
            logger.info(f"Checked SST: {len(signals)} signals")
        except Exception as e:
            logger.error(f"Failed to check SST: {e}")
        
        try:
            from pipeline.vegetation_health import VegetationHealthMonitor
            monitor = VegetationHealthMonitor()
            if not hasattr(monitor, "generate_all_signals"):
                raise AttributeError("VegetationHealthMonitor missing generate_all_signals()")
            signals = monitor.generate_all_signals()
            all_signals.extend(signals)
            logger.info(f"Checked vegetation: {len(signals)} signals")
        except Exception as e:
            logger.error(f"Failed to check vegetation: {e}")
        
        try:
            from pipeline.soil_moisture import SoilMoistureMonitor
            monitor = SoilMoistureMonitor()
            if not hasattr(monitor, "generate_all_signals"):
                raise AttributeError("SoilMoistureMonitor missing generate_all_signals()")
            signals = monitor.generate_all_signals()
            all_signals.extend(signals)
            logger.info(f"Checked soil moisture: {len(signals)} signals")
        except Exception as e:
            logger.error(f"Failed to check soil moisture: {e}")
        
        try:
            from pipeline.atmospheric import AtmosphericMonitor
            monitor = AtmosphericMonitor()
            if not hasattr(monitor, "generate_all_signals"):
                raise AttributeError("AtmosphericMonitor missing generate_all_signals()")
            signals = monitor.generate_all_signals()
            all_signals.extend(signals)
            logger.info(f"Checked atmospheric: {len(signals)} signals")
        except Exception as e:
            logger.error(f"Failed to check atmospheric: {e}")
        
        try:
            from pipeline.solar_irradiance import SolarIrradianceMonitor
            monitor = SolarIrradianceMonitor()
            if not hasattr(monitor, "generate_all_signals"):
                raise AttributeError("SolarIrradianceMonitor missing generate_all_signals()")
            signals = monitor.generate_all_signals()
            all_signals.extend(signals)
            logger.info(f"Checked solar irradiance: {len(signals)} signals")
        except Exception as e:
            logger.error(f"Failed to check solar irradiance: {e}")
        
        try:
            from pipeline.nighttime_lights import NighttimeLightsMonitor
            monitor = NighttimeLightsMonitor()
            if not hasattr(monitor, "generate_all_signals"):
                raise AttributeError("NighttimeLightsMonitor missing generate_all_signals()")
            signals = monitor.generate_all_signals()
            all_signals.extend(signals)
            logger.info(f"Checked nighttime lights: {len(signals)} signals")
        except Exception as e:
            logger.error(f"Failed to check nighttime lights: {e}")
        
        try:
            from pipeline.thermal_infrared import ThermalInfraredMonitor
            monitor = ThermalInfraredMonitor()
            if not hasattr(monitor, "generate_all_signals"):
                raise AttributeError("ThermalInfraredMonitor missing generate_all_signals()")
            signals = monitor.generate_all_signals()
            all_signals.extend(signals)
            logger.info(f"Checked thermal IR: {len(signals)} signals")
        except Exception as e:
            logger.error(f"Failed to check thermal IR: {e}")
        
        # Check and notify
        logger.info(f"\nTotal signals collected: {len(all_signals)}")
        return self.check_and_notify(all_signals)


def main():
    """Test signal monitor."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 60)
    print("Signal Monitor - Checking for Actionable Signals")
    print("=" * 60)
    
    # Initialize monitor
    monitor = SignalMonitor()
    
    # Check all modules
    summary = monitor.check_all_modules()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total signals checked: {summary['total_signals']}")
    print(f"Actionable signals: {summary['actionable_signals']}")
    print(f"Notifications sent: {summary['notified']}")
    
    if summary['notifications']:
        print("\nNotifications:")
        for notif in summary['notifications']:
            print(f"  • {notif['signal']}")
            for channel, success in notif['channels'].items():
                status = "✅" if success else "❌"
                print(f"    {status} {channel}")
    else:
        print("\nNo actionable signals found or notifications disabled.")
    
    print("\n" + "=" * 60)
    print("Configuration")
    print("=" * 60)
    print(f"Min Confidence: {monitor.notification_manager.config.min_confidence}%")
    print(f"Min Impact Score: {monitor.notification_manager.config.min_impact_score}")
    print(f"Email Enabled: {monitor.notification_manager.config.email_enabled}")
    print(f"SMS Enabled: {monitor.notification_manager.config.sms_enabled}")
    print(f"Discord Enabled: {monitor.notification_manager.config.discord_enabled}")
    print(f"Quiet Hours: {monitor.notification_manager.config.quiet_hours_start}:00 - {monitor.notification_manager.config.quiet_hours_end}:00")
    print("=" * 60)


if __name__ == "__main__":
    main()
