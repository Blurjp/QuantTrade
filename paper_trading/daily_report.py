"""
Daily Paper Trading Runner

Generates daily trading report and executes decisions based on signals.
"""

from datetime import date, datetime
from pathlib import Path
import json

from paper_trading.portfolio import (
    PaperTradingAccount, 
    execute_trading_decision,
    get_current_oil_price
)
from pipeline.signals import latest_region_signal


def generate_daily_report(
    region: str = "hormuz",
    output_base: str = "outputs",
    initial_capital: float = 100000,
    current_price: float = None,
) -> dict:
    """
    Generate daily trading report.
    
    Args:
        region: Region to trade on
        output_base: Output directory
        initial_capital: Starting capital
        current_price: Current oil price (if None, uses default)
    
    Returns:
        Report dictionary
    """
    # Initialize account
    account = PaperTradingAccount(
        initial_capital=initial_capital,
        output_base=output_base
    )
    
    # Get latest signal
    signal = latest_region_signal(region, output_base=output_base, version="v2")
    
    if signal is None:
        return {
            "status": "error",
            "message": "No signal available",
        }
    
    # Get current price
    if current_price is None:
        current_price = get_current_oil_price()
    
    # Execute trading decision
    trade = execute_trading_decision(
        account=account,
        signal=signal["signal"],
        confidence=signal["confidence"],
        actionability=signal["actionability"],
        current_price=current_price,
        signal_date=signal["date"],
    )
    
    # Record daily P&L
    snapshot = account.record_daily_pnl(current_price, signal["signal"])
    
    # Get account summary
    summary = account.get_summary(current_price)
    
    # Build report
    report = {
        "report_date": date.today().isoformat(),
        "signal_date": signal["date"],
        "region": region,
        "market": {
            "oil_price": current_price,
            "price_note": "WTI crude (assumed)",
        },
        "signal": {
            "signal": signal["signal"],
            "confidence": signal["confidence"],
            "actionability": signal["actionability"],
            "bias": signal.get("bias", ""),
            "zscore": signal.get("zscore", 0),
            "throughput": signal.get("throughput_index_corrected", 0),
            "coverage": signal.get("coverage_score", 0),
        },
        "account": {
            "total_value": summary["total_value"],
            "cash": summary["cash"],
            "position": summary["position"],
            "position_pnl": summary["position_pnl"],
            "total_return_pct": summary["total_return_pct"],
        },
        "trade": trade,
        "rationale": _generate_rationale(signal, summary, trade),
    }
    
    # Save report
    report_path = Path(output_base) / "paper_trading" / f"report_{date.today().isoformat()}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    
    return report


def _generate_rationale(signal: dict, summary: dict, trade: dict) -> str:
    """Generate human-readable rationale."""
    lines = []
    
    # Signal interpretation
    if signal["signal"] == "Long disruption risk":
        lines.append("📍 Signal: Supply disruption risk elevated")
        lines.append("→ Bias: Bullish crude")
        if signal["actionability"] == "Actionable":
            lines.append("→ Action: SHORT oil (bet against risk premium)")
    elif signal["signal"] == "Short disruption risk":
        lines.append("📍 Signal: Supply disruption risk low")
        lines.append("→ Bias: Bearish crude risk premium")
        if signal["actionability"] == "Actionable":
            lines.append("→ Action: Close short / stay flat")
    else:
        lines.append("📍 Signal: No clear trade signal")
        lines.append("→ Action: No position change")
    
    # Position status
    if summary["position"] < 0:
        lines.append(f"\n📊 Current Position: SHORT")
        lines.append(f"   Entry: ${summary['entry_price']:.2f}")
        lines.append(f"   Unrealized P&L: ${summary['position_pnl']:+,.2f}")
    elif summary["position"] > 0:
        lines.append(f"\n📊 Current Position: LONG")
        lines.append(f"   Entry: ${summary['entry_price']:.2f}")
        lines.append(f"   Unrealized P&L: ${summary['position_pnl']:+,.2f}")
    else:
        lines.append(f"\n📊 Current Position: FLAT")
    
    # Performance
    lines.append(f"\n💰 Total Value: ${summary['total_value']:,.2f}")
    lines.append(f"   Return: {summary['total_return_pct']:+.2f}%")
    
    # Trade action
    if trade:
        lines.append(f"\n⚡ Trade Executed: {trade['type']}")
        if 'pnl' in trade:
            lines.append(f"   Realized P&L: ${trade['pnl']:+,.2f}")
    
    return "\n".join(lines)


def format_report_message(report: dict) -> str:
    """Format report for Discord/message."""
    lines = [
        f"**📊 QuantTrade Daily Report**",
        f"Date: {report['report_date']}",
        f"",
        f"**Market**",
        f"WTI: ${report['market']['oil_price']:.2f}",
        f"",
        f"**Signal** ({report['signal_date']})",
        f"Type: {report['signal']['signal']}",
        f"Confidence: {report['signal']['confidence']}",
        f"Actionability: {report['signal']['actionability']}",
        f"",
        f"**Account**",
        f"Total: ${report['account']['total_value']:,.2f}",
        f"Return: {report['account']['total_return_pct']:+.2f}%",
        f"",
    ]
    
    if report['account']['position'] != 0:
        pos_type = "SHORT" if report['account']['position'] < 0 else "LONG"
        lines.append(f"**Position**: {pos_type}")
        lines.append(f"P&L: ${report['account']['position_pnl']:+,.2f}")
    
    if report.get('trade'):
        lines.append(f"")
        lines.append(f"**Trade**: {report['trade']['type']}")
        if 'pnl' in report['trade']:
            lines.append(f"Realized: ${report['trade']['pnl']:+,.2f}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Daily paper trading report")
    parser.add_argument("--region", default="hormuz")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--price", type=float, help="Current oil price")
    parser.add_argument("--message", action="store_true", help="Output as Discord message")
    args = parser.parse_args()
    
    report = generate_daily_report(
        region=args.region,
        output_base=args.output,
        initial_capital=args.capital,
        current_price=args.price,
    )
    
    if args.message:
        print(format_report_message(report))
    else:
        print(report["rationale"])
