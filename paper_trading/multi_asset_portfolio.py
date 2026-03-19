"""
Multi-Asset Paper Trading Portfolio

Supports trading across multiple asset classes based on satellite signals.
"""

import json
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import pandas as pd


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
    
    def can_open_position(self, ticker: str, value: float) -> tuple[bool, str]:
        """Check if position can be opened."""
        if len(self.positions) >= self.max_positions:
            return False, "Max positions reached"
        
        if value > self.cash:
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
        can_open, reason = self.can_open_position(ticker, value)
        if not can_open:
            print(f"Cannot open position: {reason}")
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
        if ticker not in self.positions:
            return None
        
        pos = self.positions[ticker]
        
        # Calculate P&L
        if pos.direction == "long":
            pnl = pos.quantity * (price - pos.entry_price)
        else:  # short
            pnl = pos.quantity * (pos.entry_price - price)
        
        # Return cash + P&L
        self.cash += pos.position_value + pnl
        
        # Record trade
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
        
        # Remove position
        del self.positions[ticker]
        
        self._save_state()
        return trade
    
    def update_position_prices(self, prices: Dict[str, float]):
        """Update all position prices and check risk management."""
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
                    trade = self.close_position(ticker, current_price, f"Stop loss triggered at ${current_price:.2f}")
                    if trade:
                        closed_trades.append(trade)
                # Check take profit
                elif current_price >= pos.take_profit:
                    trade = self.close_position(ticker, current_price, f"Take profit triggered at ${current_price:.2f}")
                    if trade:
                        closed_trades.append(trade)
            
            else:  # short
                pos.unrealized_pnl = pos.quantity * (pos.entry_price - current_price)
                
                # Check stop loss
                if current_price >= pos.stop_loss:
                    trade = self.close_position(ticker, current_price, f"Stop loss triggered at ${current_price:.2f}")
                    if trade:
                        closed_trades.append(trade)
                # Check take profit
                elif current_price <= pos.take_profit:
                    trade = self.close_position(ticker, current_price, f"Take profit triggered at ${current_price:.2f}")
                    if trade:
                        closed_trades.append(trade)
        
        return closed_trades
    
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
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


if __name__ == "__main__":
    # Example usage
    portfolio = MultiAssetPortfolio(initial_capital=100000, output_base="outputs")
    
    print("Multi-Asset Portfolio Initialized")
    print(f"Initial Capital: ${portfolio.initial_capital:,.2f}")
    print(f"Max Positions: {portfolio.max_positions}")
    print(f"Max Position Size: {portfolio.max_position_pct*100}%")
    print(f"Max Sector Exposure: {portfolio.max_sector_pct*100}%")
