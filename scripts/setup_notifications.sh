#!/bin/bash
###############################################################################
# QuantTrade Email Notification Setup Script
#
# This script helps you set up email notifications for trading signals.
#
# Usage: ./scripts/setup_notifications.sh
###############################################################################

set -e

PROJECT_DIR="/Users/jianping/projects/QuantTrade"
cd "$PROJECT_DIR"

echo "=============================================="
echo "🛰️ QuantTrade 邮件通知设置"
echo "=============================================="
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ 错误: .env 文件不存在"
    echo "   请先复制 .env.example 到 .env 并填写基本信息"
    exit 1
fi

# Source the .env file
set -a
source .env
set +a

echo "📧 当前邮件配置:"
echo "   SMTP服务器: ${SMTP_SERVER:-未配置}"
echo "   SMTP端口: ${SMTP_PORT:-未配置}"
echo "   发件人: ${SMTP_USERNAME:-未配置}"
echo "   收件人: ${NOTIFICATION_TO:-未配置}"
echo ""

# Ask if user wants to configure email
read -p "是否要配置/更新邮件通知? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "跳过邮件配置"
    exit 0
fi

echo ""
echo "请输入邮件服务器配置:"
echo ""

# Get SMTP server
read -p "SMTP服务器 (例如: smtp.gmail.com): " smtp_server
if [ -n "$smtp_server" ]; then
    sed -i '' "/^SMTP_SERVER=/d" .env 2>/dev/null || true
    echo "SMTP_SERVER=$smtp_server" >> .env
fi

# Get SMTP port
read -p "SMTP端口 (默认: 587): " smtp_port
smtp_port=${smtp_port:-587}
sed -i '' "/^SMTP_PORT=/d" .env 2>/dev/null || true
echo "SMTP_PORT=$smtp_port" >> .env

# Get SMTP username
read -p "SMTP用户名 (邮箱地址): " smtp_username
if [ -n "$smtp_username" ]; then
    sed -i '' "/^SMTP_USERNAME=/d" .env 2>/dev/null || true
    echo "SMTP_USERNAME=$smtp_username" >> .env
    sed -i '' "/^NOTIFICATION_FROM=/d" .env 2>/dev/null || true
    echo "NOTIFICATION_FROM=$smtp_username" >> .env
fi

# Get SMTP password
read -s -p "SMTP密码 (或应用专用密码): " smtp_password
echo
if [ -n "$smtp_password" ]; then
    sed -i '' "/^SMTP_PASSWORD=/d" .env 2>/dev/null || true
    echo "SMTP_PASSWORD=$smtp_password" >> .env
fi

# Get recipients
read -p "收件人邮箱 (多个用逗号分隔): " recipients
if [ -n "$recipients" ]; then
    sed -i '' "/^NOTIFICATION_TO=/d" .env 2>/dev/null || true
    echo "NOTIFICATION_TO=$recipients" >> .env
fi

echo ""
echo "✅ 邮件配置已保存到 .env"
echo ""

# Test email configuration
read -p "是否要发送测试邮件? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "正在发送测试邮件..."
    source .venv/bin/activate
    export PYTHONPATH="$PROJECT_DIR"
    python3 -c "
from pipeline.notifications import SignalNotifier
notifier = SignalNotifier()
if notifier.config.is_configured():
    result = notifier.send_email(
        '🛰️ QuantTrade 测试邮件',
        '<h1>测试成功!</h1><p>您的邮件通知已正确配置。</p>'
    )
    if result:
        print('✅ 测试邮件发送成功!')
    else:
        print('❌ 测试邮件发送失败')
else:
    print('❌ 邮件未正确配置，请检查设置')
"
fi

echo ""
echo "=============================================="
echo "设置cron任务 (可选)"
echo "=============================================="
echo ""
echo "请选择运行频率:"
echo "  1) 每30分钟 (高频监控)"
echo "  2) 每天 8:00 (开盘前)"
echo "  3) 每天 6:30 (晨间)"
echo "  4) 工作日 7:00 (交易日)"
echo "  5) 每小时 (交易时间 9-16点)"
echo "  6) 跳过"
echo ""
read -p "选择 (1-6): " schedule_choice

case $schedule_choice in
    1)
        cron_line="*/30 * * * * cd $PROJECT_DIR && ./scripts/run_daily_automated.sh"
        ;;
    2)
        cron_line="0 8 * * * cd $PROJECT_DIR && ./scripts/run_daily_automated.sh"
        ;;
    3)
        cron_line="30 6 * * * cd $PROJECT_DIR && ./scripts/run_daily_automated.sh"
        ;;
    4)
        cron_line="0 7 * * 1-5 cd $PROJECT_DIR && ./scripts/run_daily_automated.sh"
        ;;
    5)
        cron_line="0 9-16 * * 1-5 cd $PROJECT_DIR && ./scripts/run_daily_automated.sh"
        ;;
    6)
        echo "跳过cron设置"
        exit 0
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "即将添加以下cron任务:"
echo "  $cron_line"
echo ""
read -p "确认添加? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    (crontab -l 2>/dev/null; echo "$cron_line") | crontab -
    echo "✅ Cron任务已添加"
    echo ""
    echo "当前cron任务:"
    crontab -l | grep -v "^#" | grep -v "^$"
else
    echo "跳过cron设置"
    echo ""
    echo "您可以手动添加cron任务:"
    echo "  crontab -e"
    echo ""
    echo "然后添加:"
    echo "  $cron_line"
fi

echo ""
echo "=============================================="
echo "设置完成!"
echo "=============================================="
