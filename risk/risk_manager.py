"""
Risk Management Module

Enforces position sizing, exposure limits, and correlation checks
before opening new positions in the auto-trade flow.
"""

import logging
import os
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# Asset class groupings for correlation checks
ASSET_CLASS_GROUPS = {
    "agriculture": ["CORN", "SOYB", "WEAT", "Corn", "Soybeans", "Wheat", "ZS=F", "ZC=F", "ZW=F"],
    "energy": ["WTI", "Brent", "XLE", "Natural Gas", "CL=F", "BZ=F"],
    "auto": ["F", "GM", "TM", "CARZ"],
    "retail": ["WMT", "COST", "TGT", "HD", "XRT"],
    "industrial": ["CAT", "DE"],
    "china": ["FXI", "KWEB", "BABA", "JD"],
    "metals": ["GLD", "SLV", "Gold", "Silver"],
}

# Reverse lookup: ticker -> group
TICKER_TO_GROUP = {}
for group, tickers in ASSET_CLASS_GROUPS.items():
    for t in tickers:
        TICKER_TO_GROUP[t] = group


class RiskManager:
    """Enforces risk limits on portfolio positions."""

    def __init__(
        self,
        max_position_pct: float = None,
        max_total_exposure_pct: float = None,
    ):
        self.max_position_pct = max_position_pct or float(
            os.getenv("MAX_POSITION_PCT", "10")
        ) / 100.0
        self.max_total_exposure_pct = max_total_exposure_pct or float(
            os.getenv("MAX_TOTAL_EXPOSURE_PCT", "50")
        ) / 100.0

    def check_risk(
        self,
        ticker: str,
        direction: str,
        position_value: float,
        portfolio,
    ) -> Tuple[bool, str]:
        """
        Check if a new position passes all risk checks.

        Args:
            ticker: Instrument ticker
            direction: "long" or "short"
            position_value: Dollar value of proposed position
            portfolio: MultiAssetPortfolio instance

        Returns:
            (approved, reason) tuple
        """
        checks = [
            self._check_position_size(position_value, portfolio),
            self._check_total_exposure(position_value, portfolio),
            self._check_correlation(ticker, portfolio),
        ]

        for approved, reason in checks:
            if not approved:
                logger.warning(f"Risk check FAILED for {ticker}: {reason}")
                return False, reason

        logger.info(f"Risk check PASSED for {ticker} {direction} ${position_value:.0f}")
        return True, "All risk checks passed"

    def _check_position_size(
        self, position_value: float, portfolio
    ) -> Tuple[bool, str]:
        """Max position size: configurable % of portfolio per position."""
        max_value = portfolio.initial_capital * self.max_position_pct
        if position_value > max_value:
            return (
                False,
                f"Position ${position_value:.0f} exceeds {self.max_position_pct*100:.0f}% limit (${max_value:.0f})",
            )
        return True, "OK"

    def _check_total_exposure(
        self, position_value: float, portfolio
    ) -> Tuple[bool, str]:
        """Max total exposure: configurable % of portfolio."""
        current_exposure = sum(
            pos.position_value for pos in portfolio.positions.values()
        )
        new_exposure = current_exposure + position_value
        max_exposure = portfolio.initial_capital * self.max_total_exposure_pct
        if new_exposure > max_exposure:
            return (
                False,
                f"Total exposure ${new_exposure:.0f} would exceed {self.max_total_exposure_pct*100:.0f}% limit (${max_exposure:.0f})",
            )
        return True, "OK"

    def _check_correlation(
        self, ticker: str, portfolio
    ) -> Tuple[bool, str]:
        """Warn if opening position too similar to existing (same asset class group)."""
        new_group = TICKER_TO_GROUP.get(ticker)
        if not new_group:
            return True, "OK"

        existing_in_group = []
        for pos_ticker in portfolio.positions:
            if TICKER_TO_GROUP.get(pos_ticker) == new_group:
                existing_in_group.append(pos_ticker)

        if existing_in_group:
            return (
                False,
                f"Correlated position: {ticker} is in '{new_group}' group with existing {existing_in_group}",
            )
        return True, "OK"
