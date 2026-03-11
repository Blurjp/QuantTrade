# QuantTrade 数据存储位置说明

## 📁 数据存储结构

```
/Users/jianpinghuang/.openclaw/workspace/projects/QuantTrade/
├── outputs/                     # 所有运行时数据 (112MB)
│   ├── 2026-XX-XX/             # 每日检测数据 (69天)
│   ├── backfill/               # 历史回填数据
│   ├── backtest/               # 回测结果
│   ├── daily_reports/          # 每日报告
│   ├── paper_trading/          # 纸盘交易状态
│   ├── market_data/            # 市场数据缓存
│   └── regions/                # 区域特定数据
├── configs/                    # 配置文件
└── pipeline/                   # 代码和算法
```

---

## 1️⃣ 每日检测数据 (69天)

**位置:** `outputs/2026-XX-XX/`

**结构:**
```
2026-03-10/
├── crossings/          # 船只穿越数据
├── detections/         # 目标检测结果
├── logs/              # 运行日志
├── manifests/         # 数据清单
├── metrics/           # 性能指标
├── qa/                # 质量检查报告
└── tracklets/         # 目标追踪数据
```

**用途:**
- 每天自动运行的检测结果
- 包含船只、车辆等检测数据
- 用于生成每日信号

**总大小:** 112MB (69天数据)

---

## 2️⃣ 历史回填数据 (10个目标)

**位置:** `outputs/backfill/`

**文件:**
```
brazil_soy_backfill.json         # 巴西大豆 (736场景)
cushing_backfill.json            # Cushing油罐 (28场景)
detroit_auto_backfill.json       # Detroit汽车 (9场景)
hormuz_backfill.json             # Hormuz海峡 (68场景)
la_longbeach_backfill.json       # LA港口 (85场景)
panama_canal_backfill.json       # 巴拿马运河 (24场景)
iowa_corn_backfill.json          # 爱荷华玉米
costco_hq_backfill.json          # Costco停车场
walmart_hq_backfill.json         # Walmart停车场
multi_backfill_summary.json      # 回填总结
```

**用途:**
- 历史卫星数据
- 用于回测和训练
- 每个监控目标的长期数据

---

## 3️⃣ 回测结果 (9个信号)

**位置:** `outputs/backtest/`

**文件:**
```
brazil_soy_ZS=F_backtest.json    # 巴西大豆→大豆期货
cushing_WTI_backtest.json        # Cushing→WTI
detroit_auto_F_backtest.json     # Detroit→福特
hormuz_WTI_backtest.json         # Hormuz→WTI
la_longbeach_XLI_backtest.json   # LA港口→工业ETF
panama_canal_BDRY_backtest.json  # 巴拿马→干散货ETF
iowa_corn_Corn_backtest.json     # 爱荷华→玉米
costco_hq_COST_backtest.json     # Costco→Costco股票
walmart_hq_WMT_backtest.json     # Walmart→Walmart股票
```

**内容:**
```json
{
  "region": "detroit_auto",
  "ticker": "F",
  "backtest": {
    "total_signals": 9,
    "overall_accuracy": 0.889,
    "by_direction": {
      "short": {
        "count": 8,
        "accuracy": 1.0,
        "avg_return": 0.108
      }
    }
  }
}
```

**用途:**
- 信号准确率统计
- 历史表现验证
- 评分系统依据

---

## 4️⃣ 每日报告

**位置:** `outputs/daily_reports/`

**文件:**
```
report_2026-03-10.md
report_2026-03-11.md
```

**内容:**
- 组合状态
- 信号评分
- 交易建议
- 止损/止盈提醒

**生成时间:** 每天 6:00 AM EST

---

## 5️⃣ 学习反馈数据

### 信号评分系统

**位置:** `outputs/signal_scoring_report.md`

**内容:**
```
🏆 Top 5 信号评分:
1. Brazil Soy → ZS=F (A, 90分)
2. Hormuz → WTI (A, 86分)
3. Detroit → F (A, 80分)
```

**更新频率:** 每天重新计算

### 纸盘交易历史

**位置:** `outputs/paper_trading/`

**文件:**
```
account_state.json              # 账户状态
multi_asset_portfolio.json      # 多资产组合
report_2026-03-09.json          # 交易报告
```

**内容:**
- 持仓记录
- P&L历史
- 交易决策

### 市场数据缓存

**位置:** `outputs/market_data/`

**文件:**
```
FRO.parquet    # Frontline股票
STNG.parquet   # Scorpio Tankers
USO.parquet    # 石油ETF
ZIM.parquet    # Zim航运
```

**用途:**
- Yahoo Finance数据缓存
- 减少API调用
- 加速回测

---

## 6️⃣ 配置和参数

### 区域注册表

**位置:** `configs/regions/registry_v2.json`

**内容:**
```json
{
  "regions": {
    "hormuz": {
      "name": "Strait of Hormuz",
      "type": "chokepoint",
      "active": true
    },
    "brazil_soy": {
      "name": "Brazil Soybean Regions",
      "type": "agriculture",
      "active": true
    }
  }
}
```

**用途:**
- 监控目标配置
- AOI (区域) 定义
- 活跃状态

### 交易目标

**位置:** `configs/trading_targets.json`

**内容:**
```json
{
  "chokepoints": {
    "hormuz": {
      "instruments": ["WTI", "Brent", "XLE", "USO"],
      "signal_logic": {
        "long_disruption": "SHORT oil",
        "short_disruption": "CLOSE short"
      }
    }
  }
}
```

**用途:**
- 信号→标的映射
- 交易逻辑
- 风险参数

### 校准数据

**位置:** `outputs/cushing_calibration.json`

**内容:**
```json
{
  "tank_radius_pixels": 15,
  "tank_area_m2": 1000,
  "calibration_date": "2026-03-10",
  "eia_validation": {
    "correlation": 0.85
  }
}
```

**用途:**
- 算法校准参数
- EIA验证数据
- 检测阈值

---

## 7️⃣ 代码和算法

**位置:** `pipeline/`

**关键文件:**
```
signal_scoring.py              # 信号评分系统
signal_generator_optimized.py  # 优化信号生成
backtest.py                    # 回测引擎
detection_multi.py             # 多目标检测
run_daily.py                   # 每日运行
```

**学习到的参数:**
- 信号阈值 (农业3%, 航运10%)
- 基线窗口 (7天)
- 评分权重 (准确率50%, 样本30%, 可操作性20%)

---

## 8️⃣ 数据流

```
卫星数据 → 每日检测 (outputs/2026-XX-XX/)
         ↓
         回填历史 (outputs/backfill/)
         ↓
         回测验证 (outputs/backtest/)
         ↓
         信号评分 (signal_scoring.py)
         ↓
         每日报告 (outputs/daily_reports/)
         ↓
         交易决策 (outputs/paper_trading/)
```

---

## 9️⃣ 数据备份建议

### 已有备份
- ✅ 代码已推送到GitHub
- ✅ 配置文件已提交

### 需要备份
- ⚠️ outputs/ 目录 (112MB)
- ⚠️ 回测结果 (学习反馈)
- ⚠️ 纸盘交易历史

### 备份方法
```bash
# 备份outputs目录
tar -czf quanttrade_outputs_$(date +%Y%m%d).tar.gz outputs/

# 或使用rsync
rsync -av outputs/ /path/to/backup/quanttrade_outputs/
```

---

## 🔟 数据增长速度

**每日增长:**
```
检测数据: ~1.5MB/天
回填数据: 不定期 (批量)
回测数据: 不定期 (重新计算时)
报告: 2.5KB/天
```

**预期大小 (1年):**
```
检测数据: ~550MB
回填数据: ~50MB
回测数据: ~10MB
报告: ~1MB
总计: ~600MB/年
```

---

## 💡 总结

**所有数据存储在:**
```
/Users/jianpinghuang/.openclaw/workspace/projects/QuantTrade/outputs/
```

**核心数据:**
1. 每日检测 → outputs/2026-XX-XX/ (69天)
2. 历史回填 → outputs/backfill/ (10目标)
3. 回测结果 → outputs/backtest/ (9信号)
4. 学习反馈 → 回测JSON + 评分系统
5. 交易历史 → outputs/paper_trading/

**总大小:** 112MB

**备份状态:**
- ✅ 代码: GitHub
- ⚠️ 数据: 本地 (需手动备份)
