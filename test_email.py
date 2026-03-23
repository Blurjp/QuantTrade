#!/usr/bin/env python3
"""Quick test for email notification."""

import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Load .env manually
from dotenv import load_dotenv
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

print("=" * 60)
print("Environment Variables Check")
print("=" * 60)
print(f"EMAIL_ENABLED: {os.getenv('EMAIL_ENABLED')}")
print(f"SMTP_HOST: {os.getenv('SMTP_HOST')}")
print(f"SMTP_PORT: {os.getenv('SMTP_PORT')}")
print(f"SMTP_USERNAME: {os.getenv('SMTP_USERNAME')}")
print(f"SMTP_PASSWORD: {os.getenv('SMTP_PASSWORD', '')[:10]}...")
print(f"EMAIL_FROM: {os.getenv('EMAIL_FROM')}")
print(f"EMAIL_TO: {os.getenv('EMAIL_TO')}")
print("=" * 60)

# Test email
if os.getenv('EMAIL_ENABLED') == 'true':
    print("\n✅ Email is enabled")
    
    from notifications.notification_manager import NotificationManager
    
    manager = NotificationManager()
    
    # Test signal
    test_signal = {
        "signal_type": "test",
        "region_name": "Test Region",
        "date": "2026-03-22",
        "direction": "long",
        "confidence": 85.0,
        "instruments": ["TEST"],
        "rationale": "This is a test notification to verify email setup."
    }
    
    print("\nSending test email...")
    print(f"Min confidence: {manager.config.min_confidence}")
    print(f"Is quiet hours: {manager.config.is_quiet_hours()}")
    print(f"Should notify: {manager.should_notify(test_signal)}")
    
    results = manager.notify_signal(test_signal)
    
    print("\nResults:")
    if results:
        for channel, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"  {channel}: {status}")
    else:
        print("  No notifications sent (check criteria or config)")
else:
    print("\n❌ Email is disabled in .env")
