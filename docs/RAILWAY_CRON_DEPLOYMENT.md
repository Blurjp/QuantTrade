# Railway Cron Job 部署指南

## 📋 概述

将信号监控部署到 Railway 的 cron job，实现 24/7 自动监控，永不停止。

---

## 🚀 快速部署

### 步骤 1: 推送代码到 GitHub

```bash
cd /Users/jianpinghuang/.openclaw/workspace/projects/QuantTrade
git add .
git commit -m "feat: Add Railway cron job support for signal monitoring"
git push origin main
```

### 步骤 2: 在 Railway 上配置环境变量

在 Railway Dashboard 中，添加以下环境变量：

```bash
# Email Notifications
EMAIL_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=465
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
EMAIL_FROM=your-email@gmail.com
EMAIL_TO=recipient@gmail.com

# SMS Notifications
SMS_ENABLED=true
TWILIO_ACCOUNT_SID=your-account-sid
TWILIO_AUTH_TOKEN=your-auth-token
TWILIO_PHONE_FROM=+1234567890
TWILIO_PHONE_TO=+0987654321

# Monitoring Schedule
MONITORING_START_HOUR=7
MONITORING_END_HOUR=24
PIPELINE_INTERVAL_MINUTES=30

# NASA Earthdata (for real satellite data)
NASA_EARTHDATA_USERNAME=your-username
NASA_EARTHDATA_PASSWORD=your-password
```

### 步骤 3: 添加 Cron Job

**方法 1: 通过 Railway CLI**

```bash
# 安装 Railway CLI
npm install -g @railway/cli

# 登录
railway login

# 链接到你的项目
railway link

# 添加 cron job
railway cron add \
  --name signal-monitor \
  --schedule "*/30 7-23 * * *" \
  --command "python cron_check_signals.py" \
  --service scheduler
```

**方法 2: 通过 Railway Dashboard**

1. 打开你的 Railway 项目
2. 选择 `scheduler` 服务
3. 点击 "Settings" → "Cron Jobs"
4. 点击 "Add Cron Job"
5. 填写：
   - **Name:** `signal-monitor`
   - **Schedule:** `*/30 7-23 * * *`
   - **Command:** `python cron_check_signals.py`
   - **Timezone:** `America/New_York`
6. 点击 "Add"

---

## ⏰ Cron Schedule 说明

```
*/30 7-23 * * *
│    │    │ │ │
│    │    │ │ └── 每周每天
│    │    │ └──── 每月每天
│    │    └────── 7 AM 到 11 PM (EST)
│    └─────────── 每 30 分钟
└──────────────── 所有分钟
```

**其他选项：**

```bash
# 每 15 分钟 (7 AM - 11 PM)
*/15 7-23 * * *

# 每小时 (7 AM - 11 PM)
0 7-23 * * *

# 每 30 分钟 (24/7)
*/30 * * * *

# 每小时整点 (24/7)
0 * * * *
```

---

## 🔧 Railway CLI 命令

### 查看 Cron Jobs

```bash
railway cron list
```

### 查看日志

```bash
railway logs --service scheduler
```

### 手动触发

```bash
railway run python cron_check_signals.py
```

### 更新 Cron Job

```bash
railway cron update signal-monitor --schedule "*/15 7-23 * * *"
```

### 删除 Cron Job

```bash
railway cron remove signal-monitor
```

---

## 📊 监控和调试

### 查看实时日志

```bash
railway logs --tail --service scheduler
```

### 查看最近执行

在 Railway Dashboard:
1. 选择 `scheduler` 服务
2. 点击 "Deployments"
3. 查看每次 cron 执行的日志

### 测试 Cron Job

```bash
# 手动运行一次
railway run python cron_check_signals.py
```

---

## 🌍 时区设置

Railway cron jobs 默认使用 UTC 时间。

**配置 Eastern Time (EST/EDT):**

1. 在 Railway Dashboard 中设置时区：
   - Settings → Cron Jobs → Timezone
   - 选择 `America/New_York`

2. 或者在代码中处理时区（已在 `cron_check_signals.py` 中实现）

---

## 💰 费用估算

**Railway 定价:**
- Cron Job 执行: $0.000231/次
- 每天执行: 34 次 (7 AM - 11 PM, 每 30 分钟)
- 每月执行: ~1020 次
- 每月费用: ~$0.24

**通知费用:**
- 邮件: 免费
- SMS: $0.0075/条 (Twilio)
- 如果每天 10 个信号: ~$2.25/月

**总计:** ~$2.50/月

---

## 🔐 安全建议

1. **使用 Railway 环境变量**
   - 不要在代码中硬编码密钥
   - 所有敏感信息都在 Railway 环境变量中

2. **定期更新密码**
   - Gmail App Password
   - Twilio Auth Token

3. **监控使用量**
   - Railway Dashboard → Usage
   - Twilio Console → Usage

---

## 📝 检查清单

部署前确认：

- [ ] 代码已推送到 GitHub
- [ ] Railway 项目已创建
- [ ] 所有环境变量已配置
- [ ] Cron job 已添加
- [ ] 时区设置为 America/New_York
- [ ] 手动测试成功
- [ ] 日志显示正常

---

## 🚨 常见问题

### Q: Cron job 没有执行？

**检查:**
1. Cron schedule 是否正确
2. Timezone 是否设置
3. 查看日志: `railway logs --service scheduler`

### Q: 没有收到通知？

**检查:**
1. 环境变量是否正确配置
2. 是否在监控时段 (7 AM - 11 PM)
3. 信号置信度是否 ≥ 70%
4. 查看 cron 日志

### Q: 如何临时停止监控？

**方法 1:** 禁用 cron job
```bash
railway cron remove signal-monitor
```

**方法 2:** 修改环境变量
```bash
EMAIL_ENABLED=false
SMS_ENABLED=false
```

### Q: 如何恢复本地监控？

```bash
# 停止 Railway cron
railway cron remove signal-monitor

# 启动本地监控
cd /Users/jianpinghuang/.openclaw/workspace/projects/QuantTrade
source venv/bin/activate
nohup python run_signal_monitor.py > logs/signal_monitor.log 2>&1 &
```

---

## 🎯 推荐配置

**生产环境（推荐）:**
```bash
# Railway cron job
Schedule: */30 7-23 * * *
Timezone: America/New_York
Command: python cron_check_signals.py
```

**测试环境:**
```bash
# 每小时执行一次
Schedule: 0 7-23 * * *
```

**高频监控:**
```bash
# 每 15 分钟
Schedule: */15 7-23 * * *
```

---

## 📚 相关文档

- [Railway Cron Jobs](https://docs.railway.app/guides/cron-jobs)
- [Railway CLI](https://docs.railway.app/develop/cli)
- [Railway Environment Variables](https://docs.railway.app/develop/variables)

---

**部署到 Railway 后，你的监控系统将 24/7 运行，永不停止！** 🚀✅
