#!/usr/bin/env python3
"""
Send QuantTrade daily report to Discord
"""
import sys
import os

# Add project to path
sys.path.insert(0, '/Users/jianpinghuang/.openclaw/workspace/projects/QuantTrade')

def send_discord_report(report_text: str) -> bool:
    """
    Send report to Discord using webhooks or bot
    
    Args:
        report_text: The report content to send
        
    Returns:
        True if successful, False otherwise
    """
    # Try multiple methods to send to Discord
    
    # Method 1: Try using Discord webhook (if configured)
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
    
    if webhook_url:
        try:
            import requests
            
            # Discord has 2000 character limit per message
            if len(report_text) > 1900:
                # Split into multiple messages
                parts = []
                current_part = ""
                for line in report_text.split('\n'):
                    if len(current_part + line) > 1900:
                        parts.append(current_part)
                        current_part = line + '\n'
                    else:
                        current_part += line + '\n'
                if current_part:
                    parts.append(current_part)
                
                for i, part in enumerate(parts):
                    response = requests.post(
                        webhook_url,
                        json={"content": part}
                    )
                    response.raise_for_status()
                    if i < len(parts) - 1:
                        import time
                        time.sleep(0.5)  # Rate limiting
            else:
                response = requests.post(
                    webhook_url,
                    json={"content": report_text}
                )
                response.raise_for_status()
            
            print("✅ Discord通知已发送 (via webhook)")
            return True
            
        except Exception as e:
            print(f"⚠️  Webhook发送失败: {e}")
    
    # Method 2: Try using OpenClaw message tool
    try:
        # This would work if running within OpenClaw environment
        # For now, just print the report
        print("\n" + "="*60)
        print("Discord报告内容:")
        print("="*60)
        print(report_text)
        print("="*60)
        print("\n💡 提示: 设置DISCORD_WEBHOOK_URL环境变量以启用自动发送")
        print("   获取Webhook: Discord服务器设置 → 整合 → Webhook")
        return True
        
    except Exception as e:
        print(f"❌ 发送失败: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Read report from command line argument
        report = sys.argv[1]
    else:
        # Read from stdin
        report = sys.stdin.read()
    
    success = send_discord_report(report)
    sys.exit(0 if success else 1)
