#!/bin/bash
# QuantTrade Daily Update with Discord Notification
# Runs at 6:00 AM EST daily via LaunchAgent

set -e

PROJECT_DIR="/Users/jianpinghuang/.openclaw/workspace/projects/QuantTrade"
LOG_DIR="$PROJECT_DIR/logs"
DATE=$(date +%Y-%m-%d)
LOG_FILE="$LOG_DIR/daily_${DATE}.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Start logging
{
    echo "========================================"
    echo "QuantTrade Daily Update - $(date)"
    echo "========================================"
    echo ""
} | tee "$LOG_FILE"

# Activate virtual environment
cd "$PROJECT_DIR"
source .venv/bin/activate

# Run the update and send to Discord
python3 << 'PYTHON_SCRIPT'
from datetime import datetime
from paper_trading.multi_asset_portfolio import MultiAssetPortfolio
from pipeline.price_feed import fetch_price_yahoo
import json
from pathlib import Path
import subprocess
import sys

print("📊 生成每日报告...")

# Build report
report_lines = []
report_lines.append("╔══════════════════════════════════════════════════════════════╗")
report_lines.append("║        **QuantTrade 每日更新**                               ║")
report_lines.append("╚══════════════════════════════════════════════════════════════╝")
report_lines.append("")
report_lines.append(f"⏰ **更新时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("")

# Portfolio status
portfolio = MultiAssetPortfolio(100000, 'outputs')

wti = fetch_price_yahoo('WTI') or 86.0
f_price = fetch_price_yahoo('F') or 12.20

report_lines.append("## 📊 组合状态")
report_lines.append("────────────────────────────────────────")
report_lines.append("")

total_pnl = 0

for ticker, pos in portfolio.positions.items():
    price = wti if ticker == 'WTI' else f_price
    if pos.direction == 'short':
        pnl = (pos.entry_price - price) / pos.entry_price * pos.position_value
    else:
        pnl = (price - pos.entry_price) / pos.entry_price * pos.position_value
    total_pnl += pnl
    
    pnl_pct = pnl / pos.position_value * 100
    
    report_lines.append(f"🔴 **{ticker} {pos.direction.upper()}**")
    report_lines.append(f"   入场: `${pos.entry_price:.2f}`")
    report_lines.append(f"   当前: `${price:.2f}`")
    report_lines.append(f"   P&L: `${pnl:+,.2f}` ({pnl_pct:+.2f}%)")
    
    # Check triggers
    if pos.direction == "short":
        if price >= pos.stop_loss:
            report_lines.append(f"   ⚠️ **止损触发!**")
        elif price <= pos.take_profit:
            report_lines.append(f"   ✅ **止盈触发!**")
        else:
            dist_stop = ((pos.stop_loss - price) / price) * 100
            dist_target = ((price - pos.take_profit) / price) * 100
            report_lines.append(f"   止损: `${pos.stop_loss:.2f}` ({dist_stop:+.1f}%)")
            report_lines.append(f"   目标: `${pos.take_profit:.2f}` ({dist_target:+.1f}%)")
    report_lines.append("")

total_value = portfolio.cash + sum(pos.position_value for pos in portfolio.positions.values()) + total_pnl

report_lines.append(f"💰 **总资产:** `${total_value:,.2f}`")
report_lines.append(f"💵 **现金:** `${portfolio.cash:,.2f}`")
report_lines.append(f"📈 **总P&L:** `${total_pnl:+,.2f}`")
report_lines.append("")

# Signal summary
report_lines.append("## 🎯 信号状态")
report_lines.append("────────────────────────────────────────")
report_lines.append("")

# Add signal scoring
try:
    from pipeline.signal_scoring import SignalScorer
    scorer = SignalScorer()
    recommendations = scorer.get_trading_recommendations()
    
    report_lines.append("## 🏆 信号评分 (Top 5)")
    report_lines.append("────────────────────────────────────────")
    report_lines.append("")
    
    top_signals = scorer.get_top_signals(5)
    for i, (signal_id, signal) in enumerate(top_signals, 1):
        if signal['best_direction'] and signal['best_accuracy'] >= 50:
            report_lines.append(f"**{i}. {signal_id}** ({signal['rating']})")
            report_lines.append(f"   方向: {signal['best_direction']} ({signal['best_accuracy']:.0f}%)")
            report_lines.append(f"   评分: {signal['score']:.0f}/100")
        report_lines.append("")
    
    report_lines.append("## 💡 今日建议")
    report_lines.append("────────────────────────────────────────")
    report_lines.append("")
    
    if recommendations["strong_buy"]:
        report_lines.append("**强烈推荐:**")
        for rec in recommendations["strong_buy"]:
            report_lines.append(f"  • {rec['signal']}: {rec['direction']}")
        report_lines.append("")
    
    if recommendations["buy"]:
        report_lines.append("**推荐使用:**")
        for rec in recommendations["buy"]:
            report_lines.append(f"  • {rec['signal']}: {rec['direction']}")
        report_lines.append("")
    
except Exception as e:
    report_lines.append("⚠️ 评分系统暂时不可用")
    report_lines.append("")

backtest_dir = Path("outputs/backtest")
if backtest_dir.exists():
    for file in sorted(backtest_dir.glob("*.json")):
        data = json.loads(file.read_text())
        region = data.get("region", "?")
        ticker = data.get("ticker", "?")
        backtest = data.get("backtest", {})
        
        accuracy = backtest.get("overall_accuracy", 0) * 100
        
        if accuracy >= 70:
            quality = "✅"
        elif accuracy >= 50:
            quality = "⚠️"
        else:
            quality = "❌"
        
        report_lines.append(f"{quality} `{region}` → `{ticker}`: **{accuracy:.1f}%**")

report_lines.append("")

# Recommendations
report_lines.append("## 💡 今日建议")
report_lines.append("────────────────────────────────────────")
report_lines.append("")

for ticker, pos in portfolio.positions.items():
    price = wti if ticker == 'WTI' else f_price
    if pos.direction == 'short':
        pnl = (pos.entry_price - price) / pos.entry_price * pos.position_value
    else:
        pnl = (price - pos.entry_price) / pos.entry_price * pos.position_value
    
    if pnl > 100:
        report_lines.append(f"• **{ticker}:** 继续持有 ✅")
    elif pnl < -100:
        report_lines.append(f"• **{ticker}:** 监控止损 ⚠️")
    else:
        report_lines.append(f"• **{ticker}:** 等待信号 ➡️")

report_lines.append("")
report_lines.append("✅ 系统更新完成")
report_lines.append("")
report_lines.append("_下次更新: 明天 6:00 AM EST_")

report = "\n".join(report_lines)

# Print to console
print(report)
print()

# Try to send to Discord using OpenClaw message tool
print("📤 发送Discord通知...")
try:
    # Use message tool via subprocess (if available)
    # Or directly use Discord webhook if configured
    import os
    
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
    
    if webhook_url:
        import requests
        
        # Split message if too long
        if len(report) > 1900:
            parts = []
            current = ""
            for line in report.split('\n'):
                if len(current + line) > 1800:
                    parts.append(current)
                    current = line + '\n'
                else:
                    current += line + '\n'
            if current:
                parts.append(current)
            
            for i, part in enumerate(parts):
                response = requests.post(
                    webhook_url,
                    json={"content": f"**Part {i+1}/{len(parts)}**\n{part}"}
                )
                response.raise_for_status()
                if i < len(parts) - 1:
                    import time
                    time.sleep(0.5)
        else:
            response = requests.post(
                webhook_url,
                json={"content": report}
            )
            response.raise_for_status()
        
        print("✅ Discord通知已发送 (via webhook)")
    else:
        print("ℹ️  未配置DISCORD_WEBHOOK_URL")
        print("   报告已生成，但未发送到Discord")
        print("   设置方法: export DISCORD_WEBHOOK_URL='your-url'")
        
except Exception as e:
    print(f"⚠️  Discord发送失败: {e}")
    print("   报告已生成，请手动查看")

# Save report to file
report_file = Path("outputs/daily_reports") / f"report_{datetime.now().strftime('%Y-%m-%d')}.md"
report_file.parent.mkdir(parents=True, exist_ok=True)
report_file.write_text(report)
print(f"📄 报告已保存: {report_file}")

PYTHON_SCRIPT

echo "" | tee -a "$LOG_FILE"
echo "Daily update complete at $(date)" | tee -a "$LOG_FILE"
