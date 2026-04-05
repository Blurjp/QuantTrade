"""
Backtest Engine

Loads historical signals, simulates trades using the same logic as auto-trade,
and calculates performance metrics: win rate, avg return, Sharpe ratio, max drawdown.
Generates report with equity curve.
"""

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Simulates historical trading based on past signals."""

    def __init__(
        self,
        output_base: str = "outputs",
        initial_capital: float = 100000,
        position_size_pct: float = 0.05,
        max_positions: int = 10,
        stop_loss_pct: float = 0.08,
        take_profit_pct: float = 0.15,
        min_confidence: float = 75,
    ):
        self.output_base = Path(output_base)
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_confidence = min_confidence

    def run(
        self,
        from_date: str,
        to_date: str,
    ) -> Dict:
        """
        Run backtest over date range.

        Args:
            from_date: Start date YYYY-MM-DD
            to_date: End date YYYY-MM-DD

        Returns:
            Backtest results dict
        """
        logger.info(f"Running backtest from {from_date} to {to_date}")

        # Load historical signals
        signals_by_date = self._load_signals(from_date, to_date)
        logger.info(f"Loaded signals for {len(signals_by_date)} dates")

        if not signals_by_date:
            return {"error": "No signals found in date range"}

        # Fetch historical prices for all tickers
        all_tickers = set()
        for sigs in signals_by_date.values():
            for sig in sigs:
                for inst in sig.get("instruments", []):
                    if isinstance(inst, str):
                        all_tickers.add(inst)

        price_history = self._fetch_price_history(
            list(all_tickers), from_date, to_date
        )
        logger.info(f"Fetched price history for {len(price_history)} tickers")

        # Simulate trading
        return self._simulate(signals_by_date, price_history, from_date, to_date)

    def _load_signals(self, from_date: str, to_date: str) -> Dict[str, List[Dict]]:
        """Load signals for each date in range."""
        signals_by_date = {}

        for date_dir in sorted(self.output_base.iterdir()):
            if not date_dir.is_dir():
                continue
            dirname = date_dir.name
            if not dirname[:4].isdigit():
                continue
            if dirname < from_date or dirname > to_date:
                continue

            # Load daily summary
            summary_file = date_dir / "daily_summary.json"
            if not summary_file.exists():
                continue

            try:
                summary = json.loads(summary_file.read_text())
                day_signals = []

                for region_id, sig_data in summary.get("signals", {}).items():
                    direction = sig_data.get("direction", "neutral")
                    if direction == "neutral":
                        continue

                    confidence = sig_data.get("confidence", 0)
                    # Handle string confidence values
                    if isinstance(confidence, str):
                        conf_map = {"High": 85, "Medium": 65, "Low": 40}
                        confidence = conf_map.get(confidence, 50)

                    if confidence < self.min_confidence:
                        continue

                    instruments = sig_data.get("instruments", [])
                    if not instruments:
                        continue

                    day_signals.append({
                        "region_id": region_id,
                        "direction": direction,
                        "confidence": confidence,
                        "instruments": instruments,
                        "signal_type": sig_data.get("signal_type", ""),
                    })

                if day_signals:
                    signals_by_date[dirname] = day_signals

            except Exception as e:
                logger.warning(f"Failed to load signals for {dirname}: {e}")

        return signals_by_date

    def _fetch_price_history(
        self, tickers: List[str], from_date: str, to_date: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Fetch historical price data for tickers.

        Returns: {ticker: {date_str: price}}
        """
        price_history = {}

        try:
            import yfinance as yf

            # Extend range for stop-loss/take-profit checks after to_date
            end_dt = datetime.strptime(to_date, "%Y-%m-%d") + timedelta(days=30)

            for ticker in tickers:
                try:
                    data = yf.download(
                        ticker,
                        start=from_date,
                        end=end_dt.strftime("%Y-%m-%d"),
                        progress=False,
                    )
                    if data.empty:
                        continue

                    prices = {}
                    for idx, row in data.iterrows():
                        date_str = idx.strftime("%Y-%m-%d")
                        close = row["Close"]
                        # Handle MultiIndex columns from yfinance
                        if hasattr(close, 'item'):
                            close = close.item()
                        prices[date_str] = float(close)

                    if prices:
                        price_history[ticker] = prices
                        logger.info(f"Fetched {len(prices)} days for {ticker}")

                except Exception as e:
                    logger.warning(f"Failed to fetch prices for {ticker}: {e}")

        except ImportError:
            logger.error("yfinance not installed, using synthetic prices")
            # Generate synthetic prices for testing
            import random
            start_dt = datetime.strptime(from_date, "%Y-%m-%d")
            end_dt = datetime.strptime(to_date, "%Y-%m-%d") + timedelta(days=30)
            for ticker in tickers:
                prices = {}
                price = random.uniform(10, 100)
                current = start_dt
                while current <= end_dt:
                    if current.weekday() < 5:  # Skip weekends
                        price *= 1 + random.uniform(-0.03, 0.03)
                        prices[current.strftime("%Y-%m-%d")] = round(price, 2)
                    current += timedelta(days=1)
                price_history[ticker] = prices

        return price_history

    def _simulate(
        self,
        signals_by_date: Dict[str, List[Dict]],
        price_history: Dict[str, Dict[str, float]],
        from_date: str,
        to_date: str,
    ) -> Dict:
        """Run the trading simulation."""
        cash = self.initial_capital
        positions = {}  # ticker -> {direction, entry_price, quantity, value, entry_date}
        trades = []
        equity_curve = []

        # Get all trading days in range
        all_dates = sorted(set(
            d for prices in price_history.values() for d in prices.keys()
            if from_date <= d <= to_date
        ))

        for current_date in all_dates:
            # Check stop-loss / take-profit on existing positions
            for ticker in list(positions.keys()):
                pos = positions[ticker]
                current_price = price_history.get(ticker, {}).get(current_date)
                if current_price is None:
                    continue

                should_close = False
                reason = ""

                if pos["direction"] == "long":
                    pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
                    if pnl_pct <= -self.stop_loss_pct:
                        should_close = True
                        reason = "stop_loss"
                    elif pnl_pct >= self.take_profit_pct:
                        should_close = True
                        reason = "take_profit"
                else:
                    pnl_pct = (pos["entry_price"] - current_price) / pos["entry_price"]
                    if pnl_pct <= -self.stop_loss_pct:
                        should_close = True
                        reason = "stop_loss"
                    elif pnl_pct >= self.take_profit_pct:
                        should_close = True
                        reason = "take_profit"

                if should_close:
                    if pos["direction"] == "long":
                        pnl = pos["quantity"] * (current_price - pos["entry_price"])
                    else:
                        pnl = pos["quantity"] * (pos["entry_price"] - current_price)

                    cash += pos["value"] + pnl
                    trades.append({
                        "date": current_date,
                        "ticker": ticker,
                        "action": "CLOSE",
                        "price": current_price,
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct * 100, 2),
                        "reason": reason,
                        "held_days": (
                            datetime.strptime(current_date, "%Y-%m-%d")
                            - datetime.strptime(pos["entry_date"], "%Y-%m-%d")
                        ).days,
                    })
                    del positions[ticker]

            # Open new positions from signals
            day_signals = signals_by_date.get(current_date, [])
            for sig in sorted(day_signals, key=lambda s: -s.get("confidence", 0)):
                if len(positions) >= self.max_positions:
                    break

                ticker = None
                for inst in sig.get("instruments", []):
                    if isinstance(inst, str) and inst not in positions:
                        if inst in price_history and current_date in price_history[inst]:
                            ticker = inst
                            break

                if not ticker:
                    continue

                price = price_history[ticker][current_date]
                position_value = cash * self.position_size_pct
                if position_value < 500 or position_value > cash:
                    continue

                quantity = position_value / price
                positions[ticker] = {
                    "direction": sig["direction"],
                    "entry_price": price,
                    "quantity": quantity,
                    "value": position_value,
                    "entry_date": current_date,
                }
                cash -= position_value

                trades.append({
                    "date": current_date,
                    "ticker": ticker,
                    "action": f"OPEN_{sig['direction'].upper()}",
                    "price": price,
                    "pnl": 0,
                    "pnl_pct": 0,
                    "reason": f"signal_{sig.get('region_id', '')}",
                    "held_days": 0,
                })

            # Calculate equity
            total = cash
            for ticker, pos in positions.items():
                current_price = price_history.get(ticker, {}).get(current_date)
                if current_price:
                    if pos["direction"] == "long":
                        total += pos["quantity"] * current_price
                    else:
                        total += pos["value"] + pos["quantity"] * (pos["entry_price"] - current_price)
                else:
                    total += pos["value"]

            equity_curve.append({
                "date": current_date,
                "equity": round(total, 2),
                "cash": round(cash, 2),
                "positions": len(positions),
            })

        # Calculate metrics
        return self._calculate_metrics(trades, equity_curve)

    def _calculate_metrics(
        self, trades: List[Dict], equity_curve: List[Dict]
    ) -> Dict:
        """Calculate backtest performance metrics."""
        close_trades = [t for t in trades if t["action"] == "CLOSE"]
        open_trades = [t for t in trades if t["action"].startswith("OPEN")]

        if not equity_curve:
            return {"error": "No equity data generated"}

        equities = [e["equity"] for e in equity_curve]
        final_equity = equities[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100

        # Win rate
        winners = [t for t in close_trades if t["pnl"] > 0]
        losers = [t for t in close_trades if t["pnl"] <= 0]
        win_rate = len(winners) / len(close_trades) * 100 if close_trades else 0

        # Average return
        avg_return = (
            sum(t["pnl_pct"] for t in close_trades) / len(close_trades)
            if close_trades
            else 0
        )

        # Max drawdown
        peak = equities[0]
        max_dd = 0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(equities) > 1:
            daily_returns = []
            for i in range(1, len(equities)):
                if equities[i - 1] > 0:
                    daily_returns.append(
                        (equities[i] - equities[i - 1]) / equities[i - 1]
                    )

            if daily_returns:
                mean_return = sum(daily_returns) / len(daily_returns)
                variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
                std_return = math.sqrt(variance)
                sharpe = (mean_return / std_return * math.sqrt(252)) if std_return > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Total P&L
        total_pnl = sum(t["pnl"] for t in close_trades)

        # Average holding period
        avg_hold = (
            sum(t["held_days"] for t in close_trades) / len(close_trades)
            if close_trades
            else 0
        )

        return {
            "summary": {
                "initial_capital": self.initial_capital,
                "final_equity": round(final_equity, 2),
                "total_return_pct": round(total_return, 2),
                "total_pnl": round(total_pnl, 2),
                "sharpe_ratio": round(sharpe, 3),
                "max_drawdown_pct": round(max_dd, 2),
                "win_rate_pct": round(win_rate, 1),
                "avg_return_pct": round(avg_return, 2),
                "total_trades": len(close_trades),
                "winning_trades": len(winners),
                "losing_trades": len(losers),
                "avg_holding_days": round(avg_hold, 1),
                "signals_processed": len(open_trades),
            },
            "trades": trades,
            "equity_curve": equity_curve,
        }

    def generate_report(self, results: Dict, output_file: Optional[str] = None) -> str:
        """Generate text report from backtest results."""
        s = results.get("summary", {})
        if not s:
            return "No results to report."

        lines = [
            "=" * 60,
            "BACKTEST REPORT",
            "=" * 60,
            "",
            f"Initial Capital:     ${s['initial_capital']:,.2f}",
            f"Final Equity:        ${s['final_equity']:,.2f}",
            f"Total Return:        {s['total_return_pct']:+.2f}%",
            f"Total P&L:           ${s['total_pnl']:+,.2f}",
            "",
            f"Sharpe Ratio:        {s['sharpe_ratio']:.3f}",
            f"Max Drawdown:        {s['max_drawdown_pct']:.2f}%",
            "",
            f"Total Trades:        {s['total_trades']}",
            f"Win Rate:            {s['win_rate_pct']:.1f}%",
            f"Avg Return/Trade:    {s['avg_return_pct']:+.2f}%",
            f"Avg Holding Period:  {s['avg_holding_days']:.1f} days",
            f"Winning Trades:      {s['winning_trades']}",
            f"Losing Trades:       {s['losing_trades']}",
            "",
            "=" * 60,
            "TRADE LOG",
            "=" * 60,
        ]

        for t in results.get("trades", []):
            pnl_str = ""
            if t["action"] == "CLOSE":
                pnl_str = f"  P&L: ${t['pnl']:+.2f} ({t['pnl_pct']:+.1f}%) [{t['reason']}]"
            lines.append(
                f"  {t['date']}  {t['action']:12s}  {t['ticker']:6s}  "
                f"${t['price']:.2f}{pnl_str}"
            )

        lines.extend(["", "=" * 60])
        report = "\n".join(lines)

        if output_file:
            Path(output_file).write_text(report)
            logger.info(f"Report saved to {output_file}")

        return report
