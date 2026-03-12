#!/usr/bin/env python3
"""
Generate Weekly Performance Review

Usage:
    PYTHONPATH=. python scripts/weekly_review.py
    
This script:
1. Evaluates pending signals (3+ days old)
2. Calculates win rate per region
3. Generates threshold adjustment suggestions
4. Saves report to outputs/weekly_review/
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.signal_tracker import (
    evaluate_pending_signals,
    calculate_region_stats,
    generate_weekly_review,
)


def main():
    parser = argparse.ArgumentParser(description="Generate weekly performance review")
    parser.add_argument("--output", default="outputs", help="Output directory")
    args = parser.parse_args()
    
    print("=" * 60)
    print("每周信号表现复盘")
    print("=" * 60)
    
    # Step 1: Evaluate pending signals
    print("\n[1/3] 评估待处理信号...")
    results = evaluate_pending_signals(args.output)
    print(f"   评估了 {len(results)} 个信号")
    
    # Step 2: Calculate region stats
    print("\n[2/3] 计算区域统计...")
    stats = calculate_region_stats(args.output)
    print(f"   统计了 {len(stats)} 个区域")
    
    # Step 3: Generate review
    print("\n[3/3] 生成每周报告...")
    report = generate_weekly_review(args.output)
    
    # Save report
    review_dir = Path(args.output) / "weekly_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = review_dir / f"review_{datetime.now().strftime('%Y%m%d')}.json"
    report_file.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    
    # Also save as markdown
    md_file = review_dir / f"review_{datetime.now().strftime('%Y%m%d')}.md"
    md_content = _format_report_markdown(report)
    md_file.write_text(md_content)
    
    print(f"\n报告已保存到:")
    print(f"   JSON: {report_file}")
    print(f"   MD:   {md_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("摘要")
    print("=" * 60)
    print(f"本周信号数: {report['summary']['total_signals']}")
    print(f"已评估: {report['summary']['evaluated_signals']}")
    print(f"胜率: {report['summary']['overall_win_rate']}")
    
    if report.get("best_performing"):
        best = report["best_performing"][0]
        print(f"最佳区域: {best['region_id']} ({best['win_rate']})")
    
    if report.get("worst_performing"):
        worst = report["worst_performing"][0]
        print(f"最差区域: {worst['region_id']} ({worst['win_rate']})")
    
    if report.get("suggestions"):
        print(f"\n系统建议 ({len(report['suggestions'])} 条):")
        for sug in report["suggestions"][:3]:
            print(f"   - {sug['region_id']}: {sug['suggestion']}")


def _format_report_markdown(report: dict) -> str:
    """Format report as markdown for display."""
    lines = [
        f"# 📊 每周信号表现报告",
        f"",
        f"**报告日期:** {report['report_date']}",
        f"**统计周期:** {report['period']}",
        "",
        "## 📈 总体表现",
        "",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| 本周信号数 | {report['summary']['total_signals']} |",
        f"| 已评估信号 | {report['summary']['evaluated_signals']} |",
        f"| 正确信号 | {report['summary']['correct_signals']} |",
        f"| 胜率 | {report['summary']['overall_win_rate']} |",
        "",
    ]
    
    if report.get("best_performing"):
        lines.append("## 🏆 最佳表现区域")
        lines.append("")
        lines.append("| 区域 | 胜率 | 信号数 |")
        lines.append("|------|------|--------|")
        for item in report["best_performing"]:
            lines.append(f"| {item['region_id']} | {item['win_rate']} | {item['signals']} |")
        lines.append("")
    
    if report.get("worst_performing"):
        lines.append("## 🥉 最差表现区域")
        lines.append("")
        lines.append("| 区域 | 胜率 | 信号数 |")
        lines.append("|------|------|--------|")
        for item in report["worst_performing"]:
            lines.append(f"| {item['region_id']} | {item['win_rate']} | {item['signals']} |")
        lines.append("")
    
    if report.get("region_stats"):
        lines.append("## 📋 各区域详细统计")
        lines.append("")
        lines.append("| 区域 | 信号数 | 已评估 | 胜率 | 平均涨跌 |")
        lines.append("|------|--------|--------|------|----------|")
        for region_id, stats in sorted(report["region_stats"].items()):
            wr = stats.get("win_rate", "N/A")
            if isinstance(wr, float):
                wr = f"{wr:.1%}"
            avg = stats.get("avg_price_change", "N/A")
            if isinstance(avg, float):
                avg = f"{avg:.2f}%"
            lines.append(f"| {region_id} | {stats['total_signals']} | {stats['evaluated_count']} | {wr} | {avg} |")
        lines.append("")
    
    if report.get("suggestions"):
        lines.append("## 💡 系统建议")
        lines.append("")
        for sug in report["suggestions"]:
            priority = "🔴" if sug["priority"] == "high" else "🟡"
            lines.append(f"### {priority} {sug['region_id']}")
            lines.append(f"- **问题:** {sug['reason']}")
            lines.append(f"- **建议:** {sug['suggestion']}")
            lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    main()
