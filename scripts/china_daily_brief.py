#!/usr/bin/env python3
"""Generate a Chinese daily trading brief from the latest summary."""

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def classify_signal(region_id, signal):
    action = signal.get("trading_action", "FLAT")
    raw_action = signal.get("raw_trading_action", action)
    actionability = signal.get("actionability", "Ignore")
    confidence = signal.get("confidence", "Low")

    if actionability == "Actionable" and action != "FLAT":
        return "tradable"
    if raw_action != "FLAT" and action == "FLAT":
        return "pending"
    if region_id.endswith("_meta") and raw_action == "FLAT":
        return "blocked"
    return "inactive"


def format_line(region_id, signal):
    action = signal.get("trading_action", "FLAT")
    raw_action = signal.get("raw_trading_action", action)
    confidence = signal.get("confidence", "Low")
    text = signal.get("signal", "No data")

    extras = []
    if "vote_score" in signal:
        extras.append(f"vote={signal['vote_score']:.3f}")
    elif "ndvi_change" in signal:
        extras.append(f"ndvi={signal['ndvi_change']:.3f}")

    suffix = f" | {'; '.join(extras)}" if extras else ""
    if raw_action != action:
        suffix += f" | 原始={raw_action}"

    return f"- {region_id}: {action} / {confidence} / {text}{suffix}"


def build_brief(summary, persistence_state):
    signals = summary.get("signals", {})
    tradable = []
    pending = []
    blocked = []
    inactive = []

    for region_id in sorted(signals.keys()):
        signal = signals[region_id]
        bucket = classify_signal(region_id, signal)
        if bucket == "tradable":
            tradable.append((region_id, signal))
        elif bucket == "pending":
            pending.append((region_id, signal))
        elif bucket == "blocked":
            blocked.append((region_id, signal))
        else:
            inactive.append((region_id, signal))

    lines = []
    lines.append(f"# QuantTrade 每日中文简报 - {summary.get('date')}")
    lines.append("")
    lines.append("## 今日结论")
    if tradable:
        lines.append(f"- 今日可交易信号 {len(tradable)} 个，优先按系统规则执行。")
    else:
        lines.append("- 今日没有新的系统级可交易信号，建议以观望为主。")
    if pending:
        lines.append(f"- 有 {len(pending)} 个信号处于确认期，先观察，不立即下单。")
    lines.append(f"- 活跃区域处理成功 {summary.get('regions_successful', 0)}/{summary.get('regions_processed', 0)}。")
    lines.append("")

    lines.append("## 今日可交易")
    if tradable:
        lines.extend(format_line(region_id, signal) for region_id, signal in tradable)
    else:
        lines.append("- 无")
    lines.append("")

    lines.append("## 今日不可交易")
    if blocked:
        lines.extend(format_line(region_id, signal) for region_id, signal in blocked)
    else:
        lines.append("- 无")
    lines.append("")

    lines.append("## 今日观察名单")
    if pending:
        lines.extend(format_line(region_id, signal) for region_id, signal in pending)
    else:
        lines.append("- 无")
    lines.append("")

    lines.append("## 其他中性信号")
    neutral_items = [item for item in inactive if item[0] not in {"brazil_soy_meta", "us_retail_meta"}]
    if neutral_items:
        lines.extend(format_line(region_id, signal) for region_id, signal in neutral_items)
    else:
        lines.append("- 无")
    lines.append("")

    lines.append("## 风险提示")
    brazil_meta = signals.get("brazil_soy_meta")
    if brazil_meta:
        lines.append(
            f"- 巴西大豆元信号为 {brazil_meta.get('trading_action')}，原始方向 {brazil_meta.get('raw_trading_action', brazil_meta.get('trading_action'))}，说明区域信号仍有分歧。"
        )
    lines.append("- 若信号显示 pending confirmation，代表系统要求至少连续确认后才允许翻仓。")
    lines.append("- 本简报基于输出文件自动生成，正式交易前仍应结合流动性、事件风险和仓位限制。")
    lines.append("")

    lines.append("## 持久化状态")
    for key in sorted(persistence_state.keys()):
        state = persistence_state[key]
        lines.append(
            f"- {key}: live={state.get('live_action')} pending={state.get('pending_action')} count={state.get('pending_count')}"
        )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Chinese daily trade brief")
    parser.add_argument("--date", required=True, help="Daily summary date")
    parser.add_argument("--output", default="outputs", help="Output directory")
    args = parser.parse_args()

    output_base = Path(args.output)
    summary = load_json(output_base / args.date / "daily_summary.json")
    if not summary:
        print("Daily summary not found")
        return 1

    persistence_state = load_json(output_base / "signal_persistence_state.json") or {}
    brief = build_brief(summary, persistence_state)

    target_dir = output_base / args.date
    target_dir.mkdir(parents=True, exist_ok=True)
    md_path = target_dir / "daily_brief_zh.md"
    txt_path = target_dir / "daily_brief_zh.txt"
    md_path.write_text(brief)
    txt_path.write_text(brief)
    print(brief)
    print()
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
