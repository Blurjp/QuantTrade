#!/usr/bin/env python3
"""Send a report by SMTP using environment variables."""

import os
import smtplib
import ssl
import sys
from email.message import EmailMessage


def send_email_report(report_text: str, subject: str) -> bool:
    host = os.environ.get("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT", "465"))
    username = os.environ.get("SMTP_USERNAME")
    password = os.environ.get("SMTP_PASSWORD")
    sender = os.environ.get("SMTP_FROM", username or "")
    recipients = os.environ.get("SMTP_TO", "")
    use_starttls = os.environ.get("SMTP_STARTTLS", "false").lower() == "true"

    if not host or not sender or not recipients:
        print("Email skipped: missing SMTP_HOST / SMTP_FROM / SMTP_TO")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipients
    msg.set_content(report_text)

    try:
        if use_starttls:
            with smtplib.SMTP(host, port, timeout=30) as server:
                server.ehlo()
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
                if username and password:
                    server.login(username, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=30) as server:
                if username and password:
                    server.login(username, password)
                server.send_message(msg)

        print("Email sent successfully")
        return True
    except Exception as exc:
        print(f"Email send failed: {exc}")
        return False


if __name__ == "__main__":
    subject = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EMAIL_SUBJECT", "QuantTrade Daily Brief")
    report = sys.stdin.read()
    sys.exit(0 if send_email_report(report, subject) else 1)
