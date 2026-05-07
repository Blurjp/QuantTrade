# Railway 配置指南

## 1. Cron Job 配置

1. 打开 Railway Dashboard: https://railway.app/dashboard
2. 选择 QuantTrade 项目
3. 点击 scheduler 服务
4. 进入 Settings → Cron Jobs
5. 添加新 Cron Job:

| 项目 | 值 |
|-----|---|
| Name | `signal-monitor` |
| Schedule | `*/30 7-23 * * *` |
| Command | `python cron_check_signals.py` |
| Timezone | `America/New_York` |

## 2. 环境变量配置

在 Settings → Variables 中添加:

### NASA Earthdata (降水数据)
```
NASA_EARTHDATA_USERNAME=你的用户名
NASA_EARTHDATA_PASSWORD=你的密码
```

注册: https://urs.earthdata.nasa.gov/

### Gmail SMTP (邮件通知)
```
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=你的邮箱@gmail.com
SMTP_PASSWORD=你的应用密码
NOTIFICATION_TO=收件人邮箱
NOTIFICATION_FROM=你的邮箱@gmail.com
```

### SMS (可选, 文字通知)
```
SMS_GATEWAY=手机号@vtext.com
```

运营商网关:
- Verizon: `手机号@vtext.com`
- AT&T: `手机号@txt.att.net`
- T-Mobile: `手机号@tmomail.net`

## 3. Gmail 应用密码获取

1. 访问: https://myaccount.google.com/apppasswords
2. 选择 "邮件" 和 "自定义名称"
3. 生成 16 位应用密码
4. 复制密码到 `SMTP_PASSWORD`

## 4. 验证配置

部署后检查日志:
```bash
railway logs --service scheduler --tail
```

应看到:
```
Checking signals for actionable notifications...
Signal check complete: X actionable signals
```
