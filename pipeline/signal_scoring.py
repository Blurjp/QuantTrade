"""
Signal Scoring System
综合评分系统，评估所有信号的质量和可操作性
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

class SignalScorer:
    """
    为所有信号打分，生成交易建议
    """
    
    def __init__(self):
        self.signals = {}
        self.load_all_signals()
    
    def load_all_signals(self):
        """加载所有回测结果"""
        backtest_dir = Path("outputs/backtest")
        if not backtest_dir.exists():
            return
        
        for file in backtest_dir.glob("*.json"):
            data = json.loads(file.read_text())
            region = data.get("region")
            ticker = data.get("ticker")
            backtest = data.get("backtest", {})
            
            overall_acc = backtest.get("overall_accuracy", 0) * 100
            total_signals = backtest.get("total_signals", 0)
            
            # Check best direction (excluding neutral)
            by_direction = backtest.get("by_direction", {})
            best_direction = None
            best_accuracy = 0
            has_directional_signal = False
            
            for direction, stats in by_direction.items():
                count = stats.get("count", 0)
                acc = stats.get("accuracy", 0) * 100
                # Only consider long/short signals, not neutral
                if direction not in ["neutral", "No trade"] and count >= 3 and acc > best_accuracy:
                    best_accuracy = acc
                    best_direction = direction
                    has_directional_signal = True
            
            # Calculate score (0-100)
            # Factors: directional accuracy (50%), sample size (30%), actionability (20%)
            
            # Accuracy score (50 points) - only for directional signals
            if has_directional_signal:
                accuracy_score = min(50, best_accuracy / 2)
            else:
                # Penalize signals with only neutral
                accuracy_score = 10
            
            # Sample size score (30 points)
            directional_signals = sum(
                stats.get("count", 0)
                for direction, stats in by_direction.items()
                if direction not in ["neutral", "No trade"]
            )

            if directional_signals >= 30:
                sample_score = 30
            elif directional_signals >= 10:
                sample_score = 20
            elif directional_signals >= 5:
                sample_score = 10
            else:
                sample_score = 5
            
            # Actionability score (20 points) - reward directional signals
            if has_directional_signal:
                actionability_score = 20
            else:
                actionability_score = 0
            
            total_score = accuracy_score + sample_score + actionability_score
            
            # Rating
            if total_score >= 70:
                rating = "A"
                recommendation = "强烈推荐"
            elif total_score >= 50:
                rating = "B"
                recommendation = "推荐使用"
            elif total_score >= 30:
                rating = "C"
                recommendation = "谨慎使用"
            else:
                rating = "D"
                recommendation = "不建议"
            
            self.signals[f"{region}→{ticker}"] = {
                "region": region,
                "ticker": ticker,
                "overall_accuracy": overall_acc,
                "best_direction": best_direction,
                "best_accuracy": best_accuracy,
                "total_signals": total_signals,
                "directional_signals": directional_signals,
                "score": total_score,
                "rating": rating,
                "recommendation": recommendation
            }
    
    def get_top_signals(self, n: int = 5) -> List[Tuple[str, Dict]]:
        """获取评分最高的N个信号"""
        sorted_signals = sorted(
            self.signals.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        return sorted_signals[:n]
    
    def get_trading_recommendations(self) -> Dict:
        """生成交易建议"""
        recommendations = {
            "strong_buy": [],
            "buy": [],
            "hold": [],
            "avoid": []
        }
        
        for signal_id, signal in self.signals.items():
            if signal["rating"] == "A":
                recommendations["strong_buy"].append({
                    "signal": signal_id,
                    "direction": signal["best_direction"],
                    "accuracy": signal["best_accuracy"],
                    "score": signal["score"]
                })
            elif signal["rating"] == "B":
                recommendations["buy"].append({
                    "signal": signal_id,
                    "direction": signal["best_direction"],
                    "accuracy": signal["best_accuracy"],
                    "score": signal["score"]
                })
            elif signal["rating"] == "C":
                recommendations["hold"].append({
                    "signal": signal_id,
                    "direction": signal["best_direction"],
                    "accuracy": signal["best_accuracy"],
                    "score": signal["score"]
                })
            else:
                recommendations["avoid"].append({
                    "signal": signal_id,
                    "score": signal["score"]
                })
        
        return recommendations
    
    def generate_report(self) -> str:
        """生成评分报告"""
        report = []
        report.append("╔══════════════════════════════════════════════════════════════╗")
        report.append("║        QuantTrade 信号评分系统                               ║")
        report.append("╚══════════════════════════════════════════════════════════════╝")
        report.append("")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Top signals
        report.append("="*60)
        report.append("🏆 评分最高的信号 (Top 5)")
        report.append("="*60)
        report.append("")
        
        for i, (signal_id, signal) in enumerate(self.get_top_signals(5), 1):
            report.append(f"{i}. {signal_id}")
            report.append(f"   评分: {signal['score']:.0f}/100 ({signal['rating']})")
            report.append(f"   最佳方向: {signal['best_direction']}")
            report.append(f"   准确率: {signal['best_accuracy']:.1f}%")
            report.append(f"   信号数: {signal['total_signals']}")
            report.append(f"   建议: {signal['recommendation']}")
            report.append("")
        
        # Trading recommendations
        recommendations = self.get_trading_recommendations()
        
        report.append("="*60)
        report.append("💡 交易建议")
        report.append("="*60)
        report.append("")
        
        if recommendations["strong_buy"]:
            report.append("强烈推荐 (A级):")
            for rec in recommendations["strong_buy"]:
                report.append(f"  • {rec['signal']}")
                report.append(f"    方向: {rec['direction']} ({rec['accuracy']:.0f}%准确率)")
            report.append("")
        
        if recommendations["buy"]:
            report.append("推荐使用 (B级):")
            for rec in recommendations["buy"]:
                report.append(f"  • {rec['signal']}")
                report.append(f"    方向: {rec['direction']} ({rec['accuracy']:.0f}%准确率)")
            report.append("")
        
        if recommendations["hold"]:
            report.append("谨慎使用 (C级):")
            for rec in recommendations["hold"]:
                report.append(f"  • {rec['signal']} ({rec['accuracy']:.0f}%)")
            report.append("")
        
        if recommendations["avoid"]:
            report.append("不建议 (D级):")
            for rec in recommendations["avoid"]:
                report.append(f"  • {rec['signal']}")
            report.append("")
        
        return "\n".join(report)


if __name__ == "__main__":
    scorer = SignalScorer()
    print(scorer.generate_report())
    
    # Save report
    report_file = Path("outputs/signal_scoring_report.md")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(scorer.generate_report())
    print(f"\n✅ 报告已保存: {report_file}")
