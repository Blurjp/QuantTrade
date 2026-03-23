#!/usr/bin/env python3
"""Test SMS notification via Twilio."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env
load_dotenv(Path(__file__).parent / '.env')

print("=" * 60)
print("SMS Test - Twilio")
print("=" * 60)

# Check configuration
print(f"SMS_ENABLED: {os.getenv('SMS_ENABLED')}")
print(f"TWILIO_ACCOUNT_SID: {os.getenv('TWILIO_ACCOUNT_SID', '')[:10]}...")
print(f"TWILIO_AUTH_TOKEN: {os.getenv('TWILIO_AUTH_TOKEN', '')[:10]}...")
print(f"TWILIO_PHONE_FROM: {os.getenv('TWILIO_PHONE_FROM')}")
print(f"TWILIO_PHONE_TO: {os.getenv('TWILIO_PHONE_TO')}")
print("=" * 60)

if os.getenv('SMS_ENABLED') == 'true':
    print("\n✅ SMS is enabled")
    
    try:
        from twilio.rest import Client
        
        # Initialize Twilio client
        client = Client(
            os.getenv('TWILIO_ACCOUNT_SID'),
            os.getenv('TWILIO_AUTH_TOKEN')
        )
        
        # Send test message
        print("\nSending test SMS...")
        message = client.messages.create(
            body="🧪 QuantTrade Test: SMS notifications are working! You will receive trading signals via SMS.",
            from_=os.getenv('TWILIO_PHONE_FROM'),
            to=os.getenv('TWILIO_PHONE_TO')
        )
        
        print(f"\n✅ SMS sent successfully!")
        print(f"   Message SID: {message.sid}")
        print(f"   Status: {message.status}")
        print(f"   To: {message.to}")
        
        print(f"\n📱 Check your phone: {os.getenv('TWILIO_PHONE_TO')}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your Account SID and Auth Token")
        print("2. Verify your Twilio phone number")
        print("3. Make sure your phone number is verified (trial accounts)")
else:
    print("\n❌ SMS is disabled in .env")
