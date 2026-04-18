"""
QuantTrade Paper Trading System

Simulates trading based on system signals with $100,000 starting capital.
"""

import json
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd


class PaperTradingAccount:
    """Simulated trading account for QuantTrade signals."""
    
    def __init__(self, initial_capital: float = 100000, output_base: str = "outputs"):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.output_base = Path(output_base)
        self.state_file = self.output_base / "paper_trading" / "account_state.json"
        
        # Position tracking
        self.position = 0  # Positive = long, negative = short
        self.entry_price = 0
        self.position_value = 0
        
        # Trade history
        self.trades = []
        self.daily_pnl = []
        
        # Risk parameters
        self.max_position_pct = 0.05  # Max 5% of capital per position
        self.stop_loss_pct = 0.04  # 4% stop loss
        self.take_profit_pct = 0.15  # 15% take profit
        
        # Load existing state
        self._load_state()
    
    def _load_state(self):
        """Load account state from file."""
        if self.state_file.exists():
            state = json.loads(self.state_file.read_text())
            self.cash = state.get("cash", self.initial_capital)
            self.position = state.get("position", 0)
            self.entry_price = state.get("entry_price", 0)
            self.position_value = state.get("position_value", 0)
            self.trades = state.get("trades", [])
            self.daily_pnl = state.get("daily_pnl", [])
    
    def _save_state(self):
        """Save account state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "cash": self.cash,
            "position": self.position,
            "entry_price": self.entry_price,
            "position_value": self.position_value,
            "trades": self.trades,
            "daily_pnl": self.daily_pnl,
            "last_updated": datetime.now().isoformat(),
        }
        self.state_file.write_text(json.dumps(state, indent=2))
    
    def get_total_value(self, current_price: float) -> float:
        """Calculate total account value at current price."""
        position_pnl = 0
        if self.position != 0:
            if self.position < 0:  # Short position
                position_pnl = abs(self.position) * (self.entry_price - current_price)
            else:  # Long position
                position_pnl = self.position * (current_price - self.entry_price)
        return self.cash + self.position_value + position_pnl
    
    def get_position_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L for current position."""
        if self.position == 0:
            return 0
        if self.position < 0:  # Short
            return abs(self.position) * (self.entry_price - current_price)
        else:  # Long
            return self.position * (current_price - self.entry_price)
    
    def open_position(self, direction: str, price: float, rationale: str, signal_date: str):
        """Open a new position."""
        # Close existing position first
        if self.position != 0:
            self.close_position(price, "Reversing position")
        
        # Calculate position size
        max_position_value = self.cash * self.max_position_pct
        position_size = max_position_value / price
        
        if direction == "short":
            self.position = -position_size
            self.entry_price = price
            self.position_value = max_position_value
            self.cash -= max_position_value  # Margin
            trade_type = "SHORT"
        else:  # long
            self.position = position_size
            self.entry_price = price
            self.position_value = max_position_value
            self.cash -= max_position_value
        
        trade = {
            "date": signal_date,
            "type": "OPEN_" + trade_type,
            "price": price,
            "quantity": abs(self.position),
            "value": max_position_value,
            "rationale": rationale,
        }
        self.trades.append(trade)
        self._save_state()
        
        return trade
    
    def close_position(self, price: float, rationale: str):
        """Close current position."""
        if self.position == 0:
            return None
        
        pnl = self.get_position_pnl(price)
        
        trade = {
            "date": date.today().isoformat(),
            "type": "CLOSE",
            "price": price,
            "quantity": abs(self.position),
            "pnl": pnl,
            "rationale": rationale,
        }
        self.trades.append(trade)
        
        # Update cash
        self.cash += self.position_value + pnl
        self.position = 0
        self.entry_price = 0
        self.position_value = 0
        
        self._save_state()
        return trade
    
    def check_risk_management(self, current_price: float) -> Optional[Dict]:
        """Check stop loss and take profit."""
        if self.position == 0:
            return None
        
        pnl_pct = self.get_position_pnl(current_price) / self.position_value if self.position_value > 0 else 0
        
        if self.position < 0:  # Short
            # Stop loss: price went up
            if pnl_pct <= -self.stop_loss_pct:
                return self.close_position(current_price, f"Stop loss triggered (-{abs(pnl_pct)*100:.1f}%)")
            # Take profit: price went down
            if pnl_pct >= self.take_profit_pct:
                return self.close_position(current_price, f"Take profit triggered (+{pnl_pct*100:.1f}%)")
        else:  # Long
            # Stop loss: price went down
            if pnl_pct <= -self.stop_loss_pct:
                return self.close_position(current_price, f"Stop loss triggered (-{abs(pnl_pct)*100:.1f}%)")
            # Take profit: price went up
            if pnl_pct >= self.take_profit_pct:
                return self.close_position(current_price, f"Take profit triggered (+{pnl_pct*100:.1f}%)")
        
        return None
    
    def record_daily_pnl(self, current_price: float, signal: str):
        """Record daily P&L snapshot."""
        total_value = self.get_total_value(current_price)
        daily_return = (total_value - self.initial_capital) / self.initial_capital
        
        snapshot = {
            "date": date.today().isoformat(),
            "cash": self.cash,
            "position": self.position,
            "entry_price": self.entry_price,
            "current_price": current_price,
            "position_pnl": self.get_position_pnl(current_price),
            "total_value": total_value,
            "total_return": daily_return,
            "signal": signal,
        }
        self.daily_pnl.append(snapshot)
        self._save_state()
        
        return snapshot
    
    def get_summary(self, current_price: float) -> Dict[str, Any]:
        """Get account summary."""
        total_value = self.get_total_value(current_price)
        total_return = (total_value - self.initial_capital) / self.initial_capital
        
        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "position": self.position,
            "entry_price": self.entry_price,
            "current_price": current_price,
            "position_pnl": self.get_position_pnl(current_price),
            "total_value": total_value,
            "total_return_pct": total_return * 100,
            "num_trades": len([t for t in self.trades if t["type"] == "CLOSE"]),
            "winning_trades": len([t for t in self.trades if t.get("pnl", 0) > 0]),
        }


def execute_trading_decision(
    account: PaperTradingAccount,
    signal: str,
    confidence: str,
    actionability: str,
    current_price: float,
    signal_date: str,
) -> Optional[Dict]:
    """
    Execute trading decision based on signal.
    
    Strategy:
    - Long disruption risk (High/Medium) + Actionable → Short oil
    - Short disruption risk (High/Medium) + Actionable → Close position / stay flat
    - Low confidence or Ignore → No action
    """
    if actionability != "Actionable":
        # Check risk management
        return account.check_risk_management(current_price)
    
    if confidence == "Low":
        return account.check_risk_management(current_price)
    
    if signal == "Long disruption risk":
        # Bullish crude → Short oil
        if account.position >= 0:  # No position or long
            return account.open_position("short", current_price, f"{signal} ({confidence})", signal_date)
        # Already short, hold
        return None
    
    elif signal == "Short disruption risk":
        # Bearish crude risk premium → Close short position
        if account.position < 0:  # Short position
            return account.close_position(current_price, f"{signal} ({confidence})")
        return None
    
    return None


def get_current_oil_price() -> float:
    """
    Get current WTI price.
    In production, this would fetch from an API.
    For simulation, use a default or cached value.
    """
    from pipeline.price_feed import get_price
    try:
        price = get_price("WTI")
        return price if price > 0 else 86.0
    except Exception:
        return 86.0


if __name__ == "__main__":
    # Example usage
    account = PaperTradingAccount(initial_capital=100000)
    print("Paper Trading Account Initialized")
    print(f"Initial Capital: ${account.initial_capital:,.2f}")
    print(f"State file: {account.state_file}")
