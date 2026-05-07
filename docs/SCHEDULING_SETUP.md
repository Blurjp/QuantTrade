# QuantTrade 自动化定时任务设置指南

本文档介绍如何设置 QuantTrade 系统自动运行。

## 🎯 目标

每天自动运行交易信号生成流程，无需手动干预。

## 📋 前置要求

- ✅ Python 3.9+ 已安装
- ✅ 项目依赖已安装 (`pip install -r requirements.txt`)
- ✅ 已成功运行过 `python scripts/run_daily.py`

## 🔧 设置方法

### 方法 1: Cron (推荐用于 macOS/Linux)

#### 步骤 1: 测试启动脚本

```bash
cd /Users/jianping/projects/QuantTrade
./scripts/run_daily_automated.sh
```

#### 步骤 2: 编辑 crontab

```bash
crontab -e
```

#### 步骤 3: 添加定时任务

选择一个适合的时间（建议美股开盘前）：

```bash
# 每天早上 7:00 运行
0 7 * * * cd /Users/jianping/projects/QuantTrade && ./scripts/run_daily_automated.sh
```

#### 步骤 4: 保存并验证

```bash
# 查看已设置的定时任务
crontab -l
```

### 方法 2: Launch Agent (macOS 推荐)

#### 创建 plist 文件

```bash
nano ~/Library/LaunchAgents/com.quanttrade.daily.plist
```

#### 配置内容

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.quanttrade.daily</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/jianping/projects/QuantTrade/scripts/run_daily_automated.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>7</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/Users/jianping/projects/QuantTrade/logs/scheduler.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/jianping/projects/QuantTrade/logs/scheduler.error.log</string>
</dict>
</plist>
```

#### 加载并启动

```bash
launchctl load ~/Library/LaunchAgents/com.quanttrade.daily.plist
launchctl start com.quanttrade.daily
```

### 方法 3: Python Schedule (跨平台)

#### 创建调度器脚本

```python
# scripts/scheduler.py
import schedule
import time
import subprocess
from datetime import datetime

def run_pipeline():
    print(f"[{datetime.now()}] Running daily pipeline...")
    result = subprocess.run(
        ["python", "scripts/run_daily.py"],
        cwd="/Users/jianping/projects/QuantTrade",
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")

# 每天早上7点运行
schedule.every().day.at("07:00").do(run_pipeline)

print("Scheduler started. Press Ctrl+C to stop.")
while True:
    schedule.run_pending()
    time.sleep(60)
```

#### 运行调度器

```bash
python -m scripts.scheduler
```

## 🔍 监控和日志

### 查看运行日志

```bash
# 今天的日志
ls -lt logs/logs/daily_pipeline_*.log | head -1

# 查看最新日志
tail -f logs/logs/daily_pipeline_$(date +%Y%m%d)*.log
```

### 检查运行状态

```bash
# 查看最近的运行结果
cat outputs/$(date +%Y-%m-%d)/daily_summary.json | jq '.signals | to_entries[] | select(.value.actionability == "Actionable") | {signal: .key, action: .value.trading_action}'
```

## ⚠️ 故障排除

### 问题 1: Cron 没有运行

**症状**: 定时时间到了但没有生成输出

**解决**:
```bash
# 检查 cron 服务状态
sudo launchctl list | grep cron

# 查看 cron 日志
log show --predicate 'process == "cron"' --last 1h
```

### 问题 2: Python 环境变量问题

**症状**: "ModuleNotFoundError: No module named 'pipeline'"

**解决**: 在脚本中添加 PYTHONPATH：
```bash
export PYTHONPATH=/Users/jianping/projects/QuantTrade
```

### 问题 3: 权限问题

**症状**: "Permission denied"

**解决**:
```bash
chmod +x scripts/run_daily_automated.sh
```

## 📊 推荐设置

### 最佳实践

1. **运行时间**: 每天早上 6:30-7:00 (美股开盘前)
2. **日志保留**: 30 天
3. **失败通知**: 配置邮件或推送通知
4. **备份策略**: 定期备份 `outputs/` 目录

### 推荐配置

```bash
# 每天早上 6:30 运行，周一到周五
30 6 * * 1-5 cd /Users/jianping/projects/QuantTrade && ./scripts/run_daily_automated.sh
```

## 🎉 验证设置

运行一天后，检查：

```bash
# 1. 检查日志文件是否存在
ls -lh logs/logs/daily_pipeline_*.log | tail -1

# 2. 检查输出文件是否存在
ls -lh outputs/$(date +%Y-%m-%d)/daily_summary.json

# 3. 查看可操作信号数量
python -c "
import json
from pathlib import Path
summary = json.loads(Path('outputs/$(date +%Y-%m-%d)/daily_summary.json').read_text())
print(f'Actionable signals: {sum(1 for s in summary.get(\"signals\", {}).values() if s.get(\"actionability\") == \"Actionable\")}')
"
```

## 📞 支持

如有问题，请查看：
- 项目日志: `logs/logs/`
- 运行日志: `outputs/YYYY-MM-DD/`
- GitHub Issues: [项目地址]
