"""
Multi-Asset Paper Trading Portfolio

Supports trading across multiple asset classes based on satellite signals.
"""

import json
import logging
import threading
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Single position in the portfolio."""
    ticker: str
    asset_class: str  # commodity, equity, etf, fx
    direction: str  # long, short
    quantity: float
    entry_price: float
    entry_date: str
    position_value: float
    rationale: str
    stop_loss: float = 0
    take_profit: float = 0
    unrealized_pnl: float = 0
    signal_accuracy: float = 0
    signal_grade: str = ""
    region_id: str = ""
    current_price: Optional[float] = None


@dataclass
class Trade:
    """Executed trade record."""
    date: str
    ticker: str
    action: str  # OPEN_LONG, OPEN_SHORT, CLOSE
    price: float
    quantity: float
    value: float
    pnl: float = 0
    rationale: str = ""


class MultiAssetPortfolio:
    """Portfolio supporting multiple asset classes and positions."""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        output_base: str = "outputs",
        max_positions: int = 10,
        max_position_pct: float = 0.10,  # 10% per position
        max_sector_pct: float = 0.25,  # 25% per sector
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.output_base = Path(output_base)
        self.state_file = self.output_base / "paper_trading" / "multi_asset_portfolio.json"
        
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        
        self.positions: Dict[str, Position] = {}  # ticker -> Position
        self.trades: List[Trade] = []
        self.daily_snapshots: List[dict] = []
        
        # Thread lock for concurrent access (reentrant to allow nested calls)
        self._lock = threading.RLock()
        
        # Asset class definitions
        self.asset_classes = {
            "commodity": {
                "description": "Energy, metals, agriculture futures",
                "examples": ["WTI", "Brent", "Corn", "Soybeans", "Gold"],
                "default_stop_loss": 0.04,
                "default_take_profit": 0.15,
            },
            "equity": {
                "description": "Individual stocks",
                "examples": ["WMT", "COST", "F", "GM", "CAT"],
                "default_stop_loss": 0.05,
                "default_take_profit": 0.20,
            },
            "etf": {
                "description": "Exchange-traded funds",
                "examples": ["XRT", "CARZ", "XLE"],
                "default_stop_loss": 0.05,
                "default_take_profit": 0.15,
            },
        }
        
        # Sector mapping
        self.sector_map = {
            "WTI": "energy",
            "Brent": "energy",
            "Natural Gas": "energy",
            "XLE": "energy",
            "WMT": "retail",
            "COST": "retail",
            "TGT": "retail",
            "HD": "retail",
            "XRT": "retail",
            "F": "auto",
            "GM": "auto",
            "TM": "auto",
            "CARZ": "auto",
            "Corn": "agriculture",
            "Soybeans": "agriculture",
            "Wheat": "agriculture",
            "CAT": "industrial",
            "DE": "industrial",
        }
        
        self._load_state()
    
    def _load_state(self):
        """Load portfolio state from file."""
        if self.state_file.exists():
            state = json.loads(self.state_file.read_text())
            self.cash = state.get("cash", self.initial_capital)
            
            # Load positions
            for ticker, pos_data in state.get("positions", {}).items():
                self.positions[ticker] = Position(**pos_data)
            
            # Load trades
            self.trades = [Trade(**t) for t in state.get("trades", [])]
            self.daily_snapshots = state.get("daily_snapshots", [])
    
    def _save_state(self):
        """Save portfolio state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "cash": self.cash,
            "positions": {
                ticker: {
                    "ticker": pos.ticker,
                    "asset_class": pos.asset_class,
                    "direction": pos.direction,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "entry_date": pos.entry_date,
                    "position_value": pos.position_value,
                    "rationale": pos.rationale,
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "current_price": getattr(pos, 'current_price', None),
                    "signal_accuracy": pos.signal_accuracy,
                    "signal_grade": pos.signal_grade,
                }
                for ticker, pos in self.positions.items()
            },
            "trades": [
                {
                    "date": t.date,
                    "ticker": t.ticker,
                    "action": t.action,
                    "price": t.price,
                    "quantity": t.quantity,
                    "value": t.value,
                    "pnl": t.pnl,
                    "rationale": t.rationale,
                }
                for t in self.trades
            ],
            "daily_snapshots": self.daily_snapshots,
            "last_updated": datetime.now().isoformat(),
        }
        
        self.state_file.write_text(json.dumps(state, indent=2))
    
    def get_sector_exposure(self, sector: str) -> float:
        """Calculate total exposure to a sector."""
        total = 0
        for ticker, pos in self.positions.items():
            if self.sector_map.get(ticker) == sector:
                total += pos.position_value
        return total
    
    def can_open_position(self, ticker: str, value: float, direction: str = "long") -> tuple[bool, str]:
        """Check if position can be opened."""
        if len(self.positions) >= self.max_positions:
            return False, "Max positions reached"
        
        if direction != "short" and value > self.cash:
            return False, "Insufficient cash"
        
        if value > self.initial_capital * self.max_position_pct:
            return False, f"Position exceeds {self.max_position_pct*100}% limit"
        
        sector = self.sector_map.get(ticker, "other")
        current_sector_exp = self.get_sector_exposure(sector)
        if current_sector_exp + value > self.initial_capital * self.max_sector_pct:
            return False, f"Sector {sector} exposure exceeds {self.max_sector_pct*100}% limit"
        
        if ticker in self.positions:
            return False, "Position already exists"
        
        return True, "OK"
    
    def open_position(
        self,
        ticker: str,
        asset_class: str,
        direction: str,
        price: float,
        value: float,
        rationale: str,
        stop_loss_pct: float = None,
        take_profit_pct: float = None,
    ) -> Optional[Trade]:
        """Open a new position."""
        with self._lock:
            can_open, reason = self.can_open_position(ticker, value, direction)
            if not can_open:
                logger.warning("Cannot open position: %s", reason)
                return None
            
            # Get default risk parameters
            asset_config = self.asset_classes.get(asset_class, {})
            if stop_loss_pct is None:
                stop_loss_pct = asset_config.get("default_stop_loss", 0.05)
            if take_profit_pct is None:
                take_profit_pct = asset_config.get("default_take_profit", 0.15)
            
            # Calculate quantity
            quantity = value / price
            
            # Set stop loss and take profit
            if direction == "long":
                stop_loss = price * (1 - stop_loss_pct)
                take_profit = price * (1 + take_profit_pct)
            else:  # short
                stop_loss = price * (1 + stop_loss_pct)
                take_profit = price * (1 - take_profit_pct)
            
            # Create position
            position = Position(
                ticker=ticker,
                asset_class=asset_class,
                direction=direction,
                quantity=quantity,
                entry_price=price,
                entry_date=date.today().isoformat(),
                position_value=value,
                rationale=rationale,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            
            self.positions[ticker] = position
            if direction == "short":
                # Short: receive cash from selling borrowed shares
                self.cash += value
            else:
                # Long: pay cash to buy
                self.cash -= value
            
            # Record trade
            trade = Trade(
                date=date.today().isoformat(),
                ticker=ticker,
                action=f"OPEN_{direction.upper()}",
                price=price,
                quantity=quantity,
                value=value,
                rationale=rationale,
            )
            self.trades.append(trade)
            
            self._save_state()
            return trade
    
    def close_position(self, ticker: str, price: float, rationale: str = "") -> Optional[Trade]:
        """Close a position."""
        with self._lock:
            return self._close_position_unlocked(ticker, price, rationale)

    def _close_position_unlocked(self, ticker: str, price: float, rationale: str = "") -> Optional[Trade]:
        if ticker not in self.positions:
            return None

        pos = self.positions[ticker]

        if pos.direction == "long":
            pnl = pos.quantity * (price - pos.entry_price)
        else:
            pnl = pos.quantity * (pos.entry_price - price)

        if pos.direction == "short":
            self.cash += pnl
        else:
            self.cash += pos.position_value + pnl

        trade = Trade(
            date=date.today().isoformat(),
            ticker=ticker,
            action="CLOSE",
            price=price,
            quantity=pos.quantity,
            value=pos.position_value,
            pnl=pnl,
            rationale=rationale,
        )
        self.trades.append(trade)

        del self.positions[ticker]

        self._save_state()
        return trade
    
    def update_position_prices(self, prices: Dict[str, float]):
        """Update all position prices and check risk management."""
        with self._lock:
            closed_trades = []
            
            for ticker, pos in list(self.positions.items()):
                if ticker not in prices:
                    continue
                
                current_price = prices[ticker]
                
                # Calculate unrealized P&L
                if pos.direction == "long":
                    pos.unrealized_pnl = pos.quantity * (current_price - pos.entry_price)
                    
                    # Check stop loss
                    if current_price <= pos.stop_loss:
                        trade = self._close_position_unlocked(ticker, current_price, f"Stop loss triggered at ${current_price:.2f}")
                        if trade:
                            closed_trades.append(trade)
                    elif current_price >= pos.take_profit:
                        trade = self._close_position_unlocked(ticker, current_price, f"Take profit triggered at ${current_price:.2f}")
                        if trade:
                            closed_trades.append(trade)
                
                else:  # short
                    pos.unrealized_pnl = pos.quantity * (pos.entry_price - current_price)

                    if current_price >= pos.stop_loss:
                        trade = self._close_position_unlocked(ticker, current_price, f"Stop loss triggered at ${current_price:.2f}")
                        if trade:
                            closed_trades.append(trade)
                    elif current_price <= pos.take_profit:
                        trade = self._close_position_unlocked(ticker, current_price, f"Take profit triggered at ${current_price:.2f}")
                        if trade:
                            closed_trades.append(trade)
            
            return closed_trades
    
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        with self._lock:
            total = self.cash

            for ticker, pos in self.positions.items():
                price = prices.get(ticker)
                if price is not None and price > 0:
                    if pos.direction == "long":
                        total += pos.quantity * price
                    else:  # short
                        total += pos.position_value + pos.quantity * (pos.entry_price - price)
                else:
                    # Fallback to position value if price unavailable
                    total += pos.position_value + (pos.unrealized_pnl or 0)

            return total
    
    def get_summary(self, prices: Dict[str, float]) -> dict:
        """Get portfolio summary."""
        with self._lock:
            total_value = self.get_total_value(prices)
            total_return = (total_value - self.initial_capital) / self.initial_capital
            
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            # Sector breakdown
            sector_breakdown = {}
            for ticker, pos in self.positions.items():
                sector = self.sector_map.get(ticker, "other")
                if sector not in sector_breakdown:
                    sector_breakdown[sector] = {"value": 0, "pnl": 0}
                sector_breakdown[sector]["value"] += pos.position_value
                sector_breakdown[sector]["pnl"] += pos.unrealized_pnl
            
            return {
                "initial_capital": self.initial_capital,
                "cash": self.cash,
                "num_positions": len(self.positions),
                "total_value": total_value,
                "total_return_pct": total_return * 100,
            "unrealized_pnl": unrealized_pnl,
            "sector_breakdown": sector_breakdown,
            "num_trades": len([t for t in self.trades if t.action == "CLOSE"]),
            "winning_trades": len([t for t in self.trades if t.pnl > 0]),
        }
    
    def record_snapshot(self, prices: Dict[str, float], signals: dict = None):
        """Record daily portfolio snapshot."""
        summary = self.get_summary(prices)
        
        snapshot = {
            "date": date.today().isoformat(),
            "cash": self.cash,
            "total_value": summary["total_value"],
            "total_return_pct": summary["total_return_pct"],
            "positions": [
                {
                    "ticker": pos.ticker,
                    "direction": pos.direction,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": prices.get(pos.ticker, pos.entry_price),
                    "unrealized_pnl": pos.unrealized_pnl,
                }
                for pos in self.positions.values()
            ],
            "signals": signals,
        }
        
        self.daily_snapshots.append(snapshot)
        self._save_state()

        return snapshot

    def sync_from_alpaca(self) -> tuple[int, List[str]]:
        """Reconcile local positions against Alpaca. Alpaca is truth.

        Returns (closed_count, warnings) where closed_count is the number of
        local positions closed because they no longer exist in Alpaca.
        """
        from execution.brokers.alpaca import AlpacaBrokerClient

        warnings: List[str] = []
        closed = 0

        broker = AlpacaBrokerClient(paper=True)

        try:
            alpaca_positions = {p.symbol: p for p in broker.get_positions()}
        except Exception as e:
            logger.error("sync_from_alpaca: failed to fetch Alpaca positions: %s", e)
            return 0, [f"fetch_failed: {e}"]

        with self._lock:
            # Close local positions that Alpaca no longer holds
            for ticker in list(self.positions.keys()):
                if ticker not in alpaca_positions:
                    pos = self.positions[ticker]
                    close_price = getattr(pos, 'current_price', None) or pos.entry_price
                    reason = "sync: closed — not found in Alpaca"
                    trade = self._close_position_unlocked(ticker, close_price, reason)
                    if trade:
                        closed += 1
                        logger.info(
                            "sync_from_alpaca: closed %s (%s) @ $%.2f P&L=$%.2f",
                            ticker, reason, close_price, trade.pnl,
                        )

            # Check Alpaca-only positions (local can't reconstruct metadata)
            for symbol, bp in alpaca_positions.items():
                if symbol not in self.positions:
                    msg = (
                        f"Alpaca holds {symbol} ({bp.side} {bp.qty} @ ${bp.avg_entry_price:.2f}) "
                        f"but local portfolio has no record — skipped"
                    )
                    warnings.append(msg)
                    logger.warning("sync_from_alpaca: %s", msg)

            # Update current_price and check for quantity/side mismatches
            for ticker, pos in self.positions.items():
                if ticker not in alpaca_positions:
                    continue
                bp = alpaca_positions[ticker]
                pos.current_price = bp.current_price
                pos.unrealized_pnl = bp.unrealized_pnl
                if pos.direction != bp.side:
                    msg = f"{ticker}: direction mismatch local={pos.direction} alpaca={bp.side}"
                    warnings.append(msg)
                    logger.warning("sync_from_alpaca: %s", msg)
                if abs(pos.quantity - bp.qty) > 0.01:
                    msg = f"{ticker}: quantity mismatch local={pos.quantity:.4f} alpaca={bp.qty:.4f}"
                    warnings.append(msg)
                    logger.warning("sync_from_alpaca: %s", msg)

            if closed > 0:
                self._save_state()

        return closed, warnings


if __name__ == "__main__":
    # Example usage
    portfolio = MultiAssetPortfolio(initial_capital=100000, output_base="outputs")
    
    print("Multi-Asset Portfolio Initialized")
    print(f"Initial Capital: ${portfolio.initial_capital:,.2f}")
    print(f"Max Positions: {portfolio.max_positions}")
    print(f"Max Position Size: {portfolio.max_position_pct*100}%")
    print(f"Max Sector Exposure: {portfolio.max_sector_pct*100}%")
