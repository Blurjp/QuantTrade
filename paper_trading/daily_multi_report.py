"""
Multi-Asset Daily Report Generator

Generates comprehensive daily reports for the multi-asset portfolio.
"""

from datetime import date
from pathlib import Path
import json
from typing import Dict, Optional

from paper_trading.multi_asset_portfolio import MultiAssetPortfolio


def format_currency(value: float) -> str:
    """Format currency with sign."""
    if value >= 0:
        return f"+${value:,.2f}"
    return f"-${abs(value):,.2f}"


def format_pct(value: float) -> str:
    """Format percentage with sign."""
    if value >= 0:
        return f"+{value:.2f}%"
    return f"{value:.2f}%"


def generate_multi_asset_report(
    portfolio: MultiAssetPortfolio,
    prices: Dict[str, float],
    signals: dict = None,
) -> str:
    """Generate comprehensive daily report."""
    
    summary = portfolio.get_summary(prices)
    
    lines = []
    
    # Header
    lines.append("╔════════════════════════════════════════════════════╗")
    lines.append("║     QuantTrade Multi-Asset Portfolio Report        ║")
    lines.append("╚════════════════════════════════════════════════════╝")
    lines.append("")
    lines.append(f"📅 Date: {date.today().isoformat()}")
    lines.append("")
    
    # Account Overview
    lines.append("📊 ACCOUNT OVERVIEW")
    lines.append("─" * 52)
    lines.append(f"  Total Value:      ${summary['total_value']:>12,.2f}")
    lines.append(f"  Cash:             ${summary['cash']:>12,.2f}")
    lines.append(f"  Total Return:     {format_pct(summary['total_return_pct']):>12}")
    lines.append(f"  Unrealized P&L:   {format_currency(summary['unrealized_pnl']):>12}")
    lines.append("")
    
    # Positions
    if portfolio.positions:
        lines.append("📍 OPEN POSITIONS")
        lines.append("─" * 52)
        
        for ticker, pos in portfolio.positions.items():
            current_price = prices.get(ticker, pos.entry_price)
            pnl_pct = (pos.unrealized_pnl / pos.position_value * 100) if pos.position_value > 0 else 0
            
            direction_icon = "🟢" if pos.direction == "long" else "🔴"
            lines.append(f"")
            lines.append(f"  {direction_icon} {ticker} ({pos.direction.upper()})")
            lines.append(f"     Entry:    ${pos.entry_price:.2f}")
            lines.append(f"     Current:  ${current_price:.2f}")
            lines.append(f"     P&L:      {format_currency(pos.unrealized_pnl)} ({format_pct(pnl_pct)})")
            lines.append(f"     Stop:     ${pos.stop_loss:.2f}")
            lines.append(f"     Target:   ${pos.take_profit:.2f}")
            lines.append(f"     Sector:   {portfolio.sector_map.get(ticker, 'other')}")
        lines.append("")
    else:
        lines.append("📍 No open positions")
        lines.append("")
    
    # Sector Breakdown
    if summary['sector_breakdown']:
        lines.append("🏭 SECTOR EXPOSURE")
        lines.append("─" * 52)
        for sector, data in summary['sector_breakdown'].items():
            lines.append(f"  {sector.capitalize():12} ${data['value']:>8,.2f}  P&L: {format_currency(data['pnl'])}")
        lines.append("")
    
    # Trade Statistics
    lines.append("📈 TRADING STATISTICS")
    lines.append("─" * 52)
    total_trades = summary['num_trades']
    wins = summary['winning_trades']
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    lines.append(f"  Closed Trades:    {total_trades}")
    lines.append(f"  Winning Trades:   {wins}")
    lines.append(f"  Win Rate:         {win_rate:.1f}%")
    lines.append("")
    
    # Signals (if provided)
    if signals:
        lines.append("📡 ACTIVE SIGNALS")
        lines.append("─" * 52)
        for region, signal in signals.items():
            icon = "🟢" if signal.get('actionability') == 'Actionable' else "🟡"
            lines.append(f"  {icon} {region}: {signal['signal']} ({signal['confidence']})")
        lines.append("")
    
    # Available Opportunities
    lines.append("🎯 MONITORING OPPORTUNITIES")
    lines.append("─" * 52)
    opportunities = [
        ("🛢️ Energy", "Hormuz/Suez chokepoints", "WTI, Brent, XLE"),
        ("🛒 Retail", "Walmart/Costco parking", "WMT, COST, XRT"),
        ("🚗 Auto", "Dealer inventory lots", "F, GM, CARZ"),
        ("🌾 Agriculture", "Crop health monitoring", "Corn, Soybeans"),
        ("📦 Logistics", "Port container activity", "Shipping ETFs"),
        ("⛽ Storage", "Oil tank levels", "WTI spread"),
    ]
    for icon_name, desc, tickers in opportunities:
        lines.append(f"  {icon_name}")
        lines.append(f"     {desc}")
        lines.append(f"     → {tickers}")
    lines.append("")
    
    # Risk Management
    lines.append("⚠️  RISK PARAMETERS")
    lines.append("─" * 52)
    lines.append(f"  Max Positions:    {portfolio.max_positions}")
    lines.append(f"  Position Limit:   {portfolio.max_position_pct*100}% of capital")
    lines.append(f"  Sector Limit:     {portfolio.max_sector_pct*100}% of capital")
    lines.append("")
    
    return "\n".join(lines)


def format_discord_report(
    portfolio: MultiAssetPortfolio,
    prices: Dict[str, float],
    signals: dict = None,
) -> str:
    """Format report for Discord."""
    summary = portfolio.get_summary(prices)
    
    lines = [
        f"**📊 QuantTrade Portfolio - {date.today().isoformat()}**",
        f"",
        f"**Account**",
        f"Total: ${summary['total_value']:,.2f} ({format_pct(summary['total_return_pct'])})",
        f"Cash: ${summary['cash']:,.2f}",
        f"",
    ]
    
    if portfolio.positions:
        lines.append("**Positions**")
        for ticker, pos in portfolio.positions.items():
            current_price = prices.get(ticker, pos.entry_price)
            pnl_pct = (pos.unrealized_pnl / pos.position_value * 100) if pos.position_value > 0 else 0
            direction = "L" if pos.direction == "long" else "S"
            lines.append(f"{ticker} [{direction}]: ${current_price:.2f} ({format_pct(pnl_pct)})")
        lines.append("")
    
    if signals:
        lines.append("**Signals**")
        for region, signal in signals.items():
            icon = "✅" if signal.get('actionability') == 'Actionable' else "⏸️"
            lines.append(f"{icon} {region}: {signal['signal'][:20]}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    portfolio = MultiAssetPortfolio(initial_capital=100000, output_base="outputs")
    
    # Example prices
    prices = {
        "WTI": 120.0,
        "WMT": 165.0,
        "COST": 720.0,
    }
    
    # Example: open a position
    portfolio.open_position(
        ticker="WTI",
        asset_class="commodity",
        direction="short",
        price=120.0,
        value=5000,
        rationale="Hormuz flow normalized, risk premium overpriced",
    )
    
    # Generate report
    print(generate_multi_asset_report(portfolio, prices))
