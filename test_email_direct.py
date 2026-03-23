#!/usr/bin/env python3
"""Direct email test."""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from dotenv import load_dotenv

# Load .env
load_dotenv(Path(__file__).parent / '.env')

print("=" * 60)
print("Direct Email Test")
print("=" * 60)

# Get credentials
smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
smtp_port = int(os.getenv('SMTP_PORT', '587'))
smtp_username = os.getenv('SMTP_USERNAME', '')
smtp_password = os.getenv('SMTP_PASSWORD', '')
email_from = os.getenv('EMAIL_FROM', '')
email_to = os.getenv('EMAIL_TO', '').split(',')

print(f"SMTP Host: {smtp_host}")
print(f"SMTP Port: {smtp_port}")
print(f"Username: {smtp_username}")
print(f"Password: {smtp_password[:10]}...")
print(f"From: {email_from}")
print(f"To: {email_to}")
print("=" * 60)

# Create message
msg = MIMEMultipart()
msg['From'] = email_from
msg['To'] = ', '.join(email_to)
msg['Subject'] = '🧪 QuantTrade Email Test'

body = """
This is a test email from QuantTrade.

If you received this email, your notification system is working!

Time: {}
""".format(os.popen('date').read())

msg.attach(MIMEText(body, 'plain'))

# Send email
print("\nSending email...")
try:
    # Try with SSL (port 465)
    print(f"Connecting to {smtp_host}:465 (SSL)...")
    server = smtplib.SMTP_SSL(smtp_host, 465, timeout=10)
    print("✅ Connected via SSL")
    
    server.login(smtp_username, smtp_password)
    print("✅ Logged in")
    
    server.sendmail(email_from, email_to, msg.as_string())
    print("✅ Email sent!")
    
    server.quit()
    print("✅ Connection closed")
    
    print("\n🎉 SUCCESS! Check your inbox: {}".format(email_to[0]))
    
except Exception as e:
    print(f"\n❌ SSL failed: {e}")
    print("\nTrying TLS (port 587)...")
    
    try:
        server = smtplib.SMTP(smtp_host, 587, timeout=10)
        server.starttls()
        print("✅ TLS started")
        
        server.login(smtp_username, smtp_password)
        print("✅ Logged in")
        
        server.sendmail(email_from, email_to, msg.as_string())
        print("✅ Email sent!")
        
        server.quit()
        print("✅ Connection closed")
        
        print("\n🎉 SUCCESS! Check your inbox: {}".format(email_to[0]))
        
    except Exception as e2:
        print(f"\n❌ TLS also failed: {e2}")
        print(f"\nTroubleshooting:")
        print("1. Check your App Password is correct")
        print("2. Make sure 2FA is enabled")
        print("3. Try creating a new App Password")
