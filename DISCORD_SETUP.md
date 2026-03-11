# QuantTrade Discord通知设置

## 🎯 目标

让QuantTrade每天早上6:00自动发送更新到Discord #quanttrade频道

---

## 📋 步骤

### 1. 创建Discord Webhook

1. 打开Discord，进入你的服务器
2. 进入 #quanttrade 频道
3. 点击频道名称旁边的 ⚙️ (设置)
4. 选择 **整合** (Integrations)
5. 点击 **Webhook**
6. 点击 **新建Webhook**
7. 给Webhook命名，例如: "QuantTrade Bot"
8. 选择 #quanttrade 频道
9. 点击 **复制Webhook URL**
10. 保存

Webhook URL格式: `https://discord.com/api/webhooks/[ID]/[TOKEN]`

---

### 2. 设置环境变量

**方法A: 添加到shell配置文件 (推荐)**

```bash
# 编辑 ~/.zshrc
echo 'export DISCORD_WEBHOOK_URL="你的webhook-url"' >> ~/.zshrc

# 重新加载配置
source ~/.zshrc
```

**方法B: 添加到LaunchAgent**

```bash
# 编辑plist文件
nano ~/Library/LaunchAgents/com.quanttrade.daily.plist

# 在 <dict> 标签后添加:
<key>EnvironmentVariables</key>
<dict>
    <key>DISCORD_WEBHOOK_URL</key>
    <string>你的webhook-url</string>
</dict>

# 重新加载
launchctl unload ~/Library/LaunchAgents/com.quanttrade.daily.plist
launchctl load ~/Library/LaunchAgents/com.quanttrade.daily.plist
```

---

### 3. 测试

```bash
cd ~/clawd/projects/QuantTrade
bash scripts/daily_update_discord.sh
```

如果成功，你应该在Discord看到更新消息。

---

## 🔍 验证自动化

```bash
# 检查LaunchAgent是否已加载
launchctl list | grep quanttrade

# 查看日志
tail -f ~/clawd/projects/QuantTrade/logs/daily_*.log

# 手动触发一次（测试）
bash ~/clawd/projects/QuantTrade/scripts/daily_update_discord.sh
```

---

## 📊 示例输出

每天早上6:00你会收到类似这样的消息:

```
╔══════════════════════════════════════════════════════════════╗
║        QuantTrade 每日更新                                   ║
╚══════════════════════════════════════════════════════════════╝

⏰ 更新时间: 2026-03-10 06:00:00

📊 组合状态
============================================================

🔴 WTI SHORT
   入场: $90.90
   当前: $84.06
   P&L: +$376.24 (+7.52%)

🔴 F SHORT
   入场: $12.19
   当前: $12.51
   P&L: -$258.41 (-2.58%)

💰 总资产: $100,117.83
💵 现金: $85,000.00
📈 总P&L: +$117.83

🎯 信号状态
============================================================

✅ detroit_auto → F: 88.9%
⚠️ hormuz → WTI: 42.9%
❌ 其他信号...

💡 今日建议
============================================================

• WTI: 继续持有
• F: 监控止损

✅ 系统更新完成
```

---

## ⚠️ 故障排除

### 问题: 没有收到Discord消息

1. **检查Webhook URL**
   ```bash
   echo $DISCORD_WEBHOOK_URL
   ```

2. **手动测试webhook**
   ```bash
   curl -X POST "$DISCORD_WEBHOOK_URL" \
     -H "Content-Type: application/json" \
     -d '{"content":"测试消息"}'
   ```

3. **检查日志**
   ```bash
   cat ~/clawd/projects/QuantTrade/logs/daily_$(date +%Y-%m-%d).log
   ```

4. **检查LaunchAgent**
   ```bash
   launchctl list | grep quanttrade
   ```

### 问题: 早上6点没有自动运行

1. **检查电脑是否开机**
   - LaunchAgent只在电脑开机时运行
   - 如果电脑睡眠，任务会在唤醒后运行

2. **检查系统日志**
   ```bash
   log show --predicate 'process == "launchd"' --last 1h
   ```

---

## 🔧 高级配置

### 修改运行时间

编辑 `~/Library/LaunchAgents/com.quanttrade.daily.plist`:

```xml
<key>StartCalendarInterval</key>
<dict>
    <key>Hour</key>
    <integer>7</integer>  <!-- 改为7点 -->
    <key>Minute</key>
    <integer>0</integer>
</dict>
```

然后重新加载:
```bash
launchctl unload ~/Library/LaunchAgents/com.quanttrade.daily.plist
launchctl load ~/Library/LaunchAgents/com.quanttrade.daily.plist
```

### 添加多个通知时间

复制plist文件并修改时间:
```bash
cp ~/Library/LaunchAgents/com.quanttrade.daily.plist \
   ~/Library/LaunchAgents/com.quanttrade.evening.plist

# 编辑evening.plist，改为18:00
# 加载
launchctl load ~/Library/LaunchAgents/com.quanttrade.evening.plist
```

---

## ✅ 完成!

设置完成后，你每天早上6:00会自动收到QuantTrade更新。

**当前状态:**
- ✅ 自动化脚本已配置
- ✅ LaunchAgent已加载
- ⚠️  需要设置Discord Webhook URL

**下一步:**
1. 获取Discord Webhook URL
2. 设置环境变量
3. 测试发送
