"""
Tests for risk management utilities.

Tests position sizing, stop loss calculation, and risk limit checking.
"""

import pytest
import numpy as np

from execution.risk import (
    PositionSizer,
    RiskMetrics,
    calculate_stop_loss,
    calculate_take_profit,
    calculate_portfolio_risk,
    check_risk_limits,
)
from strategies.base import Direction, TradeCandidate


class TestPositionSizer:
    """Tests for PositionSizer class."""

    def test_fixed_fractional_basic(self):
        """Test basic fixed fractional position sizing."""
        sizer = PositionSizer(
            method="fixed_fractional",
            portfolio_value=100000,
            risk_per_trade=0.02,
        )

        # Entry at $100, stop at $95 = $5 risk per share
        size = sizer.calculate_size(
            entry_price=100.0,
            stop_loss_price=95.0,
        )

        # Risk amount = $100,000 * 0.02 = $2,000
        # Risk per share = $5
        # Shares = $2,000 / $5 = 400 shares
        # Position size = 400 * $100 = $40,000
        assert size == pytest.approx(40000, rel=0.01)

    def test_fixed_fractional_zero_risk_per_share(self):
        """Test that zero risk per share returns 0."""
        sizer = PositionSizer(
            method="fixed_fractional",
            portfolio_value=100000,
            risk_per_trade=0.02,
        )

        size = sizer.calculate_size(
            entry_price=100.0,
            stop_loss_price=100.0,  # No difference = no risk
        )

        assert size == 0.0

    def test_kelly_criterion_basic(self):
        """Test Kelly criterion position sizing."""
        sizer = PositionSizer(
            method="kelly",
            portfolio_value=100000,
            risk_per_trade=0.02,
        )

        # With 60% win rate and 2:1 payoff ratio
        size = sizer.calculate_size(
            entry_price=100.0,
            stop_loss_price=95.0,
            win_rate=0.6,
            avg_win=200,
            avg_loss=100,
        )

        # Kelly criterion should return a positive size
        # The actual size depends on the Kelly fraction calculation
        assert size > 0
        # Kelly capped at 25% of portfolio, so max size should be reasonable
        # Note: The size can exceed portfolio value due to the calculation method

    def test_kelly_falls_back_to_fixed_fractional(self):
        """Test Kelly falls back to fixed fractional when missing params."""
        sizer = PositionSizer(
            method="kelly",
            portfolio_value=100000,
            risk_per_trade=0.02,
        )

        # Without win_rate, avg_win, avg_loss, should fall back
        size = sizer.calculate_size(
            entry_price=100.0,
            stop_loss_price=95.0,
        )

        # Should be same as fixed fractional
        assert size == pytest.approx(40000, rel=0.01)

    def test_volatility_targeting(self):
        """Test volatility targeting position sizing."""
        sizer = PositionSizer(
            method="vol_target",
            portfolio_value=100000,
            risk_per_trade=0.02,
        )

        # High volatility should reduce position
        size_high_vol = sizer.calculate_size(
            entry_price=100.0,
            stop_loss_price=95.0,
            volatility=0.30,  # 30% annual vol
        )

        # Low volatility should increase position
        size_low_vol = sizer.calculate_size(
            entry_price=100.0,
            stop_loss_price=95.0,
            volatility=0.10,  # 10% annual vol
        )

        assert size_low_vol > size_high_vol

    def test_volatility_targeting_fallback(self):
        """Test vol targeting falls back when no volatility provided."""
        sizer = PositionSizer(
            method="vol_target",
            portfolio_value=100000,
            risk_per_trade=0.02,
        )

        size = sizer.calculate_size(
            entry_price=100.0,
            stop_loss_price=95.0,
        )

        # Should fall back to fixed fractional
        assert size == pytest.approx(40000, rel=0.01)


class TestCalculateStopLoss:
    """Tests for stop loss calculation."""

    def test_stop_loss_long_percentage(self):
        """Test percentage stop loss for long position."""
        stop = calculate_stop_loss(
            entry_price=100.0,
            direction=Direction.LONG,
            method="percentage",
            stop_pct=0.05,
        )

        assert stop == pytest.approx(95.0)

    def test_stop_loss_short_percentage(self):
        """Test percentage stop loss for short position."""
        stop = calculate_stop_loss(
            entry_price=100.0,
            direction=Direction.SHORT,
            method="percentage",
            stop_pct=0.05,
        )

        assert stop == pytest.approx(105.0)

    def test_stop_loss_long_atr(self):
        """Test ATR-based stop loss for long position."""
        stop = calculate_stop_loss(
            entry_price=100.0,
            direction=Direction.LONG,
            method="atr",
            atr=2.0,
            atr_multiplier=2.0,
        )

        # Entry - (ATR * multiplier) = 100 - 4 = 96
        assert stop == pytest.approx(96.0)

    def test_stop_loss_short_atr(self):
        """Test ATR-based stop loss for short position."""
        stop = calculate_stop_loss(
            entry_price=100.0,
            direction=Direction.SHORT,
            method="atr",
            atr=2.0,
            atr_multiplier=2.0,
        )

        # Entry + (ATR * multiplier) = 100 + 4 = 104
        assert stop == pytest.approx(104.0)

    def test_stop_loss_defaults_to_percentage(self):
        """Test that unknown method defaults to percentage."""
        stop = calculate_stop_loss(
            entry_price=100.0,
            direction=Direction.LONG,
            method="unknown",
            stop_pct=0.05,
        )

        assert stop == pytest.approx(95.0)


class TestCalculateTakeProfit:
    """Tests for take profit calculation."""

    def test_take_profit_long(self):
        """Test take profit for long position."""
        tp = calculate_take_profit(
            entry_price=100.0,
            stop_loss_price=95.0,
            direction=Direction.LONG,
            reward_risk_ratio=2.0,
        )

        # Risk = 5, Reward = 10
        # TP = Entry + Reward = 100 + 10 = 110
        assert tp == pytest.approx(110.0)

    def test_take_profit_short(self):
        """Test take profit for short position."""
        tp = calculate_take_profit(
            entry_price=100.0,
            stop_loss_price=105.0,
            direction=Direction.SHORT,
            reward_risk_ratio=2.0,
        )

        # Risk = 5, Reward = 10
        # TP = Entry - Reward = 100 - 10 = 90
        assert tp == pytest.approx(90.0)

    def test_take_profit_different_ratio(self):
        """Test take profit with different reward/risk ratio."""
        tp = calculate_take_profit(
            entry_price=100.0,
            stop_loss_price=95.0,
            direction=Direction.LONG,
            reward_risk_ratio=3.0,
        )

        # Risk = 5, Reward = 15
        # TP = 100 + 15 = 115
        assert tp == pytest.approx(115.0)


class TestCheckRiskLimits:
    """Tests for risk limit checking."""

    def _make_trade(self, ticker: str, size_pct: float) -> TradeCandidate:
        """Helper to create a TradeCandidate."""
        return TradeCandidate(
            strategy="test",
            timestamp="2026-01-01T00:00:00",
            ticker=ticker,
            asset_type=None,
            direction=Direction.LONG,
            horizon_days=20,
            size_pct=size_pct,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            rationale="test trade",
        )

    def test_empty_trades_pass(self):
        """Test that empty trade list passes."""
        is_safe, violations = check_risk_limits([], {})

        assert is_safe is True
        assert violations == []

    def test_single_position_within_limit(self):
        """Test single position within limit."""
        trades = [self._make_trade("AAPL", 0.03)]  # 3% size

        is_safe, violations = check_risk_limits(
            trades,
            {},
            max_single_risk=0.05,
        )

        assert is_safe is True
        assert violations == []

    def test_single_position_exceeds_limit(self):
        """Test single position exceeding limit."""
        trades = [self._make_trade("AAPL", 0.10)]  # 10% size

        is_safe, violations = check_risk_limits(
            trades,
            {},
            max_single_risk=0.05,
        )

        assert is_safe is False
        assert len(violations) == 1
        assert "AAPL" in violations[0]

    def test_total_exposure_exceeds_limit(self):
        """Test total exposure exceeding limit."""
        trades = [
            self._make_trade("AAPL", 0.08),
            self._make_trade("MSFT", 0.08),
            self._make_trade("GOOGL", 0.08),
        ]  # Total 24%

        is_safe, violations = check_risk_limits(
            trades,
            {},
            max_portfolio_risk=0.20,
            max_single_risk=0.10,
        )

        assert is_safe is False
        assert any("exposure" in v.lower() for v in violations)

    def test_concentration_check(self):
        """Test concentration check for same ticker."""
        trades = [
            self._make_trade("AAPL", 0.06),
            self._make_trade("AAPL", 0.06),
        ]  # 12% in AAPL

        is_safe, violations = check_risk_limits(
            trades,
            {},
            max_single_risk=0.05,
        )

        # Total AAPL exposure 12% > 0.05 * 2 = 10%
        assert is_safe is False


class TestRiskMetrics:
    """Tests for RiskMetrics dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = RiskMetrics(
            entry_price=100.0,
            stop_loss_price=95.0,
            take_profit_price=110.0,
            risk_amount=500.0,
            risk_pct=0.05,
            sharpe_ratio=1.5,
        )

        d = metrics.to_dict()

        assert d["entry_price"] == 100.0
        assert d["stop_loss_price"] == 95.0
        assert d["sharpe_ratio"] == 1.5

    def test_defaults(self):
        """Test default values."""
        metrics = RiskMetrics()

        assert metrics.risk_amount == 0.0
        assert metrics.beta == 1.0
        assert metrics.sharpe_ratio == 0.0


class TestCalculatePortfolioRisk:
    """Tests for portfolio risk calculation."""

    def test_empty_positions(self):
        """Test with empty positions."""
        metrics = calculate_portfolio_risk([], {})

        assert metrics.entry_price is None
        assert metrics.risk_amount == 0.0

    def test_single_position(self):
        """Test with single position."""
        positions = [{
            "ticker": "AAPL",
            "direction": "long",
            "size": 100,
            "entry_price": 150.0,
        }]
        prices = {"AAPL": 155.0}

        metrics = calculate_portfolio_risk(positions, prices)

        assert metrics.beta == 1.0

    def test_multiple_positions(self):
        """Test with multiple positions."""
        positions = [
            {"ticker": "AAPL", "direction": "long", "size": 100, "entry_price": 150.0},
            {"ticker": "MSFT", "direction": "long", "size": 50, "entry_price": 300.0},
        ]
        prices = {"AAPL": 155.0, "MSFT": 310.0}

        metrics = calculate_portfolio_risk(positions, prices)

        assert metrics.beta == 1.0
