# QuantTrade Multi-Asset Satellite Trading System

## 系统概览

QuantTrade 是一个基于卫星图像的多资产交易信号系统，监控全球供应链、零售活动、农业产量和能源运输。

---

## 📊 监控目标 (7个活跃)

| 目标 | 类型 | 交易工具 | 检测方法 | 状态 |
|------|------|----------|----------|------|
| 🛢️ **霍尔木兹海峡** | 咽喉要道 | WTI, Brent, XLE | SAR船只检测 | ✅ 生产 |
| 🛒 **沃尔玛停车场** | 零售客流 | WMT | YOLO车辆检测 | 🔧 需模型 |
| 🛒 **Costco停车场** | 零售客流 | COST | YOLO车辆检测 | 🔧 需模型 |
| ⛽ **Cushing油罐** | 原油库存 | WTI, USO | 阴影分析 | 🔧 需校准 |
| 🌾 **爱荷华玉米带** | 农作物 | Corn, Soybeans | NDVI分析 | 🔧 需pipeline |
| 🚗 **底特律汽车库存** | 汽车库存 | F, GM, CARZ | YOLO车辆检测 | 🔧 需模型 |
| 📦 **LA/Long Beach港** | 港口物流 | XLI, FDX, UPS | SAR船只 | ✅ 就绪 |

---

## 📁 系统架构

```
QuantTrade/
├── configs/
│   ├── aoi_*.geojson          # 7个监控区域
│   ├── monitoring_types.json  # 监控类型定义
│   ├── trading_targets.json   # 交易目标配置
│   └── regions/registry_v2.json
│
├── pipeline/
│   ├── detection.py           # SAR船只检测 (生产)
│   ├── detection_multi.py     # 多类型检测注册表
│   ├── detection_vehicles.py  # 车辆检测 (YOLO)
│   ├── detection_agriculture.py # NDVI农作物
│   ├── detection_storage.py   # 油罐液位
│   ├── signals.py             # 霍尔木兹信号
│   ├── signals_multi.py       # 多类型信号
│   ├── price_feed.py          # 价格数据源
│   └── run_daily.py           # 统一日跑脚本
│
├── paper_trading/
│   ├── portfolio.py           # 单资产账户
│   ├── multi_asset_portfolio.py # 多资产组合
│   ├── daily_report.py        # 单资产报告
│   └── daily_multi_report.py  # 多资产报告
│
├── automation/
│   ├── daily.py               # 每日自动化
│   └── alerts.py              # 警报系统
│
├── scripts/
│   ├── daily_run.sh           # 原始日跑
│   └── daily_run_unified.sh   # 统一日跑
│
└── outputs/
    ├── YYYY-MM-DD/            # 每日输出
    ├── global_tracklets/      # 跨日追踪
    └── paper_trading/         # 交易状态
```

---

## 🔄 每日自动化

### 时间表
- **6:00 AM EST**: 自动运行检测和信号生成
- **6:05 AM EST**: 更新组合盈亏
- **6:10 AM EST**: 生成每日报告

### 日跑内容
1. 处理所有活跃监控区域
2. 运行对应的检测pipeline
3. 生成交易信号
4. 更新组合持仓和盈亏
5. 检查止损/止盈触发
6. 保存每日报告

---

## 📈 当前持仓

| 仓位 | 方向 | 入场价 | 目标价 | 止损价 | 理由 |
|------|------|--------|--------|--------|------|
| WTI | 做空 | $120 | $102 | $125 | 霍尔木兹航运恢复，风险溢价过高 |

---

## 🛠️ 检测模块

### 1. 船只检测 (SAR)
```python
from pipeline.detection import run_cfar_detection
detections = run_cfar_detection(scene_path, aoi)
```

### 2. 车辆检测 (光学)
```python
from pipeline.detection_vehicles import VehicleDetector
detector = VehicleDetector(use_yolo=False)
vehicles = detector.detect(image)
```

### 3. 农作物NDVI
```python
from pipeline.detection_agriculture import process_sentinel2_for_ndvi
result = process_sentinel2_for_ndvi(aoi_path, date)
```

### 4. 油罐液位
```python
from pipeline.detection_storage import TankLevelDetector
detector = TankLevelDetector()
levels = detector.analyze_tank_farm(image)
```

---

## 💰 信号逻辑

### 咽喉要道
- Long disruption risk → **做空原油** (供应紧张预期)
- Short disruption risk → **平仓** (流量正常)

### 零售客流
- 流量 > 基线15% → **做多股票** (销售强劲)
- 流量 < 基线15% → **做空股票** (销售疲软)

### 油罐库存
- 库存上升 → **看空原油** (供过于求)
- 库存下降 → **看多原油** (供应紧张)

### 农作物
- NDVI > 基线5% → **做空农产品** (丰收预期)
- NDVI < 基线5% → **做多农产品** (产量担忧)

### 汽车库存
- 库存上升 → **做空汽车股** (需求疲软)
- 库存下降 → **做多汽车股** (需求强劲)

---

## 📊 风险管理

| 参数 | 值 |
|------|-----|
| 初始资金 | $100,000 |
| 最大持仓数 | 10 |
| 单仓位上限 | 10%资金 |
| 行业上限 | 25%资金 |
| 默认止损 | 4-5% |
| 默认止盈 | 15-20% |

---

## 🚀 下一步开发

### 本周
- [ ] 添加YOLOv8模型权重
- [ ] 校准油罐液位算法
- [ ] 集成EIA数据对比
- [ ] 添加更多零售目标 (Target, Home Depot)

### 本月
- [ ] 历史回测信号vs实际价格
- [ ] 添加中国港口 (上海, 深圳)
- [ ] 添加巴西大豆区
- [ ] 优化NDVI季节性调整

### 未来
- [ ] 实时价格推送
- [ ] Discord/Telegram警报
- [ ] Web仪表板
- [ ] 多用户支持

---

## 📞 命令参考

### 运行每日pipeline
```bash
cd /path/to/QuantTrade
./scripts/daily_run_unified.sh
```

### 检查组合状态
```bash
source .venv/bin/activate
python -c "
from paper_trading.multi_asset_portfolio import MultiAssetPortfolio
p = MultiAssetPortfolio(100000, 'outputs')
s = p.get_summary({'WTI': 120})
print(f\"Total: \${s['total_value']:,.2f}\")
print(f\"Return: {s['total_return_pct']:+.2f}%\")
"
```

### 更新价格
```bash
source .venv/bin/activate
python -m pipeline.price_feed
```

### 查看信号
```bash
cat outputs/$(date +%Y-%m-%d)/daily_summary.json
```

---

## 📈 数据源

| 数据源 | 类型 | 分辨率 | 重访周期 | 成本 |
|--------|------|--------|----------|------|
| Sentinel-1 | SAR | 10m | 6天 | 免费 |
| Sentinel-2 | 光学 | 10m | 5天 | 免费 |
| Landsat | 光学 | 30m | 16天 | 免费 |
| Planet | 光学 | 3m | 每日 | 付费 |
| Maxar | 光学 | 30cm | 按需 | 付费 |

---

## ⚠️ 免责声明

这是一个模拟交易系统，仅用于研究和教育目的。不构成投资建议。卫星数据可能存在延迟和检测误差。过去的表现不保证未来的结果。

---

**系统版本**: 2.0
**最后更新**: 2026-03-09
**活跃监控**: 7个目标
**覆盖资产**: 15+交易工具
