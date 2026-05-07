"""
Signal Performance Tracker

Tracks trading signals and evaluates their accuracy over time.
After N days, fetches actual price movement and scores the signal.

This enables:
- Win rate calculation per region
- Automatic threshold adjustment suggestions
- Weekly performance reports
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


SIGNAL_TRACKER_FILE = "signal_performance.json"
EVALUATION_DELAY_DAYS = 3  # How many days after signal to evaluate


def _load_tracker(output_base: str = "outputs") -> Dict:
    """Load existing tracker data."""
    tracker_path = Path(output_base) / SIGNAL_TRACKER_FILE
    if tracker_path.exists():
        return json.loads(tracker_path.read_text())
    return {
        "signals": [],
        "evaluations": [],
        "region_stats": {},
        "last_updated": None,
    }


def _save_tracker(tracker: Dict, output_base: str = "outputs") -> None:
    """Save tracker data."""
    tracker_path = Path(output_base) / SIGNAL_TRACKER_FILE
    tracker["last_updated"] = datetime.now().isoformat()
    tracker_path.write_text(json.dumps(tracker, indent=2, default=str))


def record_signal(
    region_id: str,
    signal: Dict,
    output_base: str = "outputs",
) -> None:
    """
    Record a trading signal for later evaluation.
    
    Args:
        region_id: Region identifier (e.g., "brazil_soy_meta")
        signal: Signal dict with trading_action, confidence, instruments, etc.
        output_base: Output directory
    """
    tracker = _load_tracker(output_base)
    
    record = {
        "signal_id": f"{region_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "region_id": region_id,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "trading_action": signal.get("trading_action", "FLAT"),
        "raw_action": signal.get("raw_trading_action", signal.get("trading_action", "FLAT")),
        "confidence": signal.get("confidence", "Low"),
        "signal_text": signal.get("signal", ""),
        "instruments": signal.get("instruments", []),
        "vote_score": signal.get("vote_score"),
        "ndvi_change": signal.get("ndvi_change"),
        "evaluated": False,
        "evaluation_date": None,
        "price_entry": None,
        "price_exit": None,
        "price_change_pct": None,
        "was_correct": None,
    }
    
    tracker["signals"].append(record)
    _save_tracker(tracker, output_base)


def fetch_price_data(
    instrument: str,
    date_str: str,
    days_forward: int = 3,
) -> Optional[Dict]:
    """
    Fetch price data for an instrument around a date.
    
    Returns:
        {"entry_price": float, "exit_price": float, "change_pct": float}
    """
    try:
        import yfinance as yf
        
        ticker_map = {
            "Soybeans": "SOYB",
            "Corn": "CORN",
            "WTI": "USO",
            "Brent": "BNO",
            "XLE": "XLE",
            "XRT": "XRT",
            "XLI": "XLI",
            "WMT": "WMT",
            "COST": "COST",
            "F": "F",
            "GM": "GM",
            "FDX": "FDX",
            "UPS": "UPS",
            "BDI": "BDRY",
        }
        
        ticker = ticker_map.get(instrument, instrument)
        
        entry_date = datetime.strptime(date_str, "%Y-%m-%d")
        exit_date = entry_date + timedelta(days=days_forward)
        
        # Fetch data
        df = yf.download(
            ticker,
            start=(entry_date - timedelta(days=3)).strftime("%Y-%m-%d"),
            end=(exit_date + timedelta(days=3)).strftime("%Y-%m-%d"),
            progress=False,
        )
        
        if df.empty or len(df) < 2:
            return None
        
        # Find closest trading days
        df = df.sort_index()
        
        entry_rows = df[df.index >= entry_date.strftime("%Y-%m-%d")]
        if entry_rows.empty:
            return None
        entry_price = float(entry_rows.iloc[0]["Close"])
        entry_date_actual = entry_rows.index[0].strftime("%Y-%m-%d")
        
        exit_rows = df[df.index >= exit_date.strftime("%Y-%m-%d")]
        if exit_rows.empty:
            # Use last available date
            exit_price = float(df.iloc[-1]["Close"])
            exit_date_actual = df.index[-1].strftime("%Y-%m-%d")
        else:
            exit_price = float(exit_rows.iloc[0]["Close"])
            exit_date_actual = exit_rows.index[0].strftime("%Y-%m-%d")
        
        if entry_price == 0:
            return None
        change_pct = ((exit_price - entry_price) / entry_price) * 100
        
        return {
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "entry_date": entry_date_actual,
            "exit_date": exit_date_actual,
            "change_pct": round(change_pct, 2),
        }
    
    except Exception as e:
        return {"error": str(e)}


def evaluate_signal(
    signal_record: Dict,
    output_base: str = "outputs",
) -> Dict:
    """
    Evaluate a signal by comparing to actual price movement.
    
    Returns:
        Evaluation dict with was_correct, price_change_pct, etc.
    """
    instruments = signal_record.get("instruments", [])
    if not instruments:
        return {"was_correct": None, "error": "No instruments"}
    
    # Use first instrument for evaluation
    instrument = instruments[0]
    action = signal_record.get("trading_action", "FLAT")
    
    if action == "FLAT":
        return {"was_correct": None, "error": "FLAT signal, no evaluation"}
    
    price_data = fetch_price_data(
        instrument,
        signal_record["date"],
        days_forward=EVALUATION_DELAY_DAYS,
    )
    
    if not price_data or "error" in price_data:
        return {"was_correct": None, "error": price_data.get("error", "No price data")}
    
    change_pct = price_data["change_pct"]
    
    # Determine if signal was correct
    # LONG + price up = correct
    # SHORT + price down = correct
    if action == "LONG":
        was_correct = change_pct > 0
    elif action == "SHORT":
        was_correct = change_pct < 0
    else:
        was_correct = None
    
    return {
        "was_correct": was_correct,
        "price_entry": price_data["entry_price"],
        "price_exit": price_data["exit_price"],
        "price_change_pct": change_pct,
        "evaluation_date": datetime.now().strftime("%Y-%m-%d"),
        "instrument": instrument,
    }


def evaluate_pending_signals(output_base: str = "outputs") -> List[Dict]:
    """
    Evaluate all signals that are old enough but haven't been evaluated yet.
    
    Returns:
        List of evaluation results
    """
    tracker = _load_tracker(output_base)
    results = []
    
    cutoff_date = (datetime.now() - timedelta(days=EVALUATION_DELAY_DAYS)).strftime("%Y-%m-%d")
    
    for signal in tracker["signals"]:
        if signal.get("evaluated"):
            continue
        
        if signal["date"] > cutoff_date:
            # Not enough time has passed
            continue
        
        if signal.get("trading_action") == "FLAT":
            signal["evaluated"] = True
            signal["was_correct"] = None
            continue
        
        # Evaluate this signal
        evaluation = evaluate_signal(signal, output_base)
        
        signal["evaluated"] = True
        signal["evaluation_date"] = evaluation.get("evaluation_date")
        signal["price_entry"] = evaluation.get("price_entry")
        signal["price_exit"] = evaluation.get("price_exit")
        signal["price_change_pct"] = evaluation.get("price_change_pct")
        signal["was_correct"] = evaluation.get("was_correct")
        
        tracker["evaluations"].append({
            "signal_id": signal["signal_id"],
            **evaluation,
        })
        
        results.append(evaluation)
    
    _save_tracker(tracker, output_base)
    return results


def calculate_region_stats(output_base: str = "outputs") -> Dict:
    """
    Calculate performance statistics per region.
    
    Returns:
        Dict of region_id -> stats
    """
    tracker = _load_tracker(output_base)
    
    region_stats: Dict = {}
    
    for signal in tracker["signals"]:
        if not signal.get("evaluated"):
            continue
        
        region_id = signal["region_id"]
        action = signal.get("trading_action", "FLAT")
        
        if region_id not in region_stats:
            region_stats[region_id] = {
                "total_signals": 0,
                "long_signals": 0,
                "short_signals": 0,
                "correct_long": 0,
                "correct_short": 0,
                "total_price_change": 0,
                "evaluated_count": 0,
            }
        
        stats = region_stats[region_id]
        stats["total_signals"] += 1
        
        if signal.get("was_correct") is None:
            continue
        
        stats["evaluated_count"] += 1
        
        if action == "LONG":
            stats["long_signals"] += 1
            if signal["was_correct"]:
                stats["correct_long"] += 1
        elif action == "SHORT":
            stats["short_signals"] += 1
            if signal["was_correct"]:
                stats["correct_short"] += 1
        
        if signal.get("price_change_pct"):
            stats["total_price_change"] += signal["price_change_pct"]
    
    # Calculate win rates
    for region_id, stats in region_stats.items():
        if stats["evaluated_count"] > 0:
            total_correct = stats["correct_long"] + stats["correct_short"]
            stats["win_rate"] = round(total_correct / stats["evaluated_count"] * 100, 1)
            stats["avg_price_change"] = round(stats["total_price_change"] / stats["evaluated_count"], 2)
        else:
            stats["win_rate"] = None
            stats["avg_price_change"] = None
    
    # Update tracker
    tracker["region_stats"] = region_stats
    _save_tracker(tracker, output_base)
    
    return region_stats


def generate_weekly_review(output_base: str = "outputs") -> Dict:
    """
    Generate a weekly review report with performance analysis and suggestions.
    
    Returns:
        Report dict with summary, stats, and suggestions
    """
    # First, evaluate any pending signals
    evaluate_pending_signals(output_base)
    
    # Calculate stats
    region_stats = calculate_region_stats(output_base)
    
    tracker = _load_tracker(output_base)
    
    # Get signals from last 7 days
    cutoff_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    recent_signals = [
        s for s in tracker["signals"]
        if s["date"] >= cutoff_date
    ]
    
    recent_evaluated = [s for s in recent_signals if s.get("evaluated")]
    recent_correct = sum(1 for s in recent_evaluated if s.get("was_correct"))
    
    # Generate suggestions
    suggestions = []
    
    for region_id, stats in region_stats.items():
        if stats["evaluated_count"] < 3:
            # Not enough data
            continue
        
        win_rate = stats.get("win_rate", 0)
        
        if win_rate is not None and win_rate < 40:
            suggestions.append({
                "region_id": region_id,
                "type": "threshold_increase",
                "reason": f"Win rate {win_rate}% is too low",
                "suggestion": f"Increase confirmation threshold or raise signal threshold",
                "current_win_rate": win_rate,
                "priority": "high",
            })
        elif win_rate is not None and win_rate > 70:
            suggestions.append({
                "region_id": region_id,
                "type": "increase_weight",
                "reason": f"Win rate {win_rate}% is excellent",
                "suggestion": f"Consider increasing meta_weight for this region",
                "current_win_rate": win_rate,
                "priority": "medium",
            })
    
    # Build report
    report = {
        "report_date": datetime.now().strftime("%Y-%m-%d"),
        "period": "last_7_days",
        "summary": {
            "total_signals": len(recent_signals),
            "evaluated_signals": len(recent_evaluated),
            "correct_signals": recent_correct,
            "overall_win_rate": round(recent_correct / len(recent_evaluated) * 100, 1) if recent_evaluated else None,
        },
        "region_stats": region_stats,
        "suggestions": suggestions,
        "best_performing": None,
        "worst_performing": None,
    }
    
    # Find best/worst performing regions
    valid_regions = [
        (rid, stats) for rid, stats in region_stats.items()
        if stats.get("win_rate") is not None and stats["evaluated_count"] >= 3
    ]
    
    if valid_regions:
        sorted_regions = sorted(valid_regions, key=lambda x: x[1]["win_rate"], reverse=True)
        if sorted_regions:
            report["best_performing"] = {
                "region_id": sorted_regions[0][0],
                "win_rate": sorted_regions[0][1]["win_rate"],
            }
            if len(sorted_regions) > 1:
                report["worst_performing"] = {
                    "region_id": sorted_regions[-1][0],
                    "win_rate": sorted_regions[-1]["win_rate"],
                }
    
    # Save report
    report_path = Path(output_base) / "weekly_review.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    
    return report


def format_review_report_cn(report: Dict) -> str:
    """
    Format the weekly review report in Chinese for display.
    """
    lines = [
        "# 📊 每周信号表现报告",
        f"\n**报告日期:** {report['report_date']}",
        f"**统计周期:** 过去 7 天",
        "",
        "## 📈 总体表现",
        "",
    ]
    
    summary = report["summary"]
    win_rate = summary["overall_win_rate"]
    
    if win_rate is not None:
        if win_rate >= 60:
            emoji = "✅"
            status = "表现良好"
        elif win_rate >= 50:
            emoji = "⚠️"
            status = "表现一般"
        else:
            emoji = "❌"
            status = "表现不佳"
    else:
        emoji = "⏳"
        status = "数据不足"
    
    lines.extend([
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| 本周信号数 | {summary['total_signals']} |",
        f"| 已评估信号 | {summary['evaluated_signals']} |",
        f"| 正确信号 | {summary['correct_signals']} |",
        f"| 胜率 | {win_rate}% {emoji} {status} |",
        "",
    ])
    
    # Best/worst performing
    if report.get("best_performing"):
        lines.extend([
            "## 🏆 最佳/最差表现区域",
            "",
            f"| 排名 | 区域 | 胜率 |",
            f"|------|------|------|",
            f"| 🥇 最佳 | {report['best_performing']['region_id']} | {report['best_performing']['win_rate']}% |",
        ])
        if report.get("worst_performing"):
            lines.append(f"| 🥉 最差 | {report['worst_performing']['region_id']} | {report['worst_performing']['win_rate']}% |")
        lines.append("")
    
    # Region stats
    if report["region_stats"]:
        lines.extend([
            "## 📋 各区域详细统计",
            "",
            "| 区域 | 信号数 | 已评估 | 胜率 | 平均涨跌 |",
            "|------|--------|--------|------|----------|",
        ])
        
        for region_id, stats in sorted(report["region_stats"].items()):
            if stats["evaluated_count"] == 0:
                continue
            wr = f"{stats.get('win_rate', 'N/A')}%" if stats.get("win_rate") else "N/A"
            avg = f"{stats.get('avg_price_change', 'N/A')}%" if stats.get("avg_price_change") else "N/A"
            lines.append(f"| {region_id} | {stats['total_signals']} | {stats['evaluated_count']} | {wr} | {avg} |")
        
        lines.append("")
    
    # Suggestions
    if report["suggestions"]:
        lines.extend([
            "## 💡 系统建议",
            "",
        ])
        
        for sug in report["suggestions"]:
            priority_emoji = "🔴" if sug["priority"] == "high" else "🟡"
            lines.extend([
                f"### {priority_emoji} {sug['region_id']}",
                f"- **问题:** {sug['reason']}",
                f"- **建议:** {sug['suggestion']}",
                "",
            ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test the tracker
    print("Signal Performance Tracker")
    print("=" * 40)
    
    # Generate a test report
    report = generate_weekly_review()
    print(format_review_report_cn(report))
