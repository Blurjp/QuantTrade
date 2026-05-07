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
    "energy": ["WTI", "Brent", "XLE", "Natural Gas", "CL=F", "BZ=F", "USO", "BNO", "OIH", "XOP"],
    "auto": ["F", "GM", "TM", "CARZ"],
    "retail": ["WMT", "COST", "TGT", "HD", "XRT"],
    "industrial": ["CAT", "DE"],
    "china": ["FXI", "KWEB", "BABA", "JD", "MCHI", "ASHR", "KANG"],
    "emerging_europe": ["EPOL", "EWG", "EWO", "EWN"],
    "metals": ["GLD", "SLV", "Gold", "Silver"],
}

# Reverse lookup: ticker -> group
TICKER_TO_GROUP = {}
for group, tickers in ASSET_CLASS_GROUPS.items():
    for t in tickers:
        TICKER_TO_GROUP[t] = group

# Sector-level groupings (superset of asset class groups for concentration checks)
SECTOR_MAP = {
    "china": ["FXI", "KWEB", "BABA", "JD", "MCHI", "ASHR", "KANG"],
    "energy": ["USO", "BNO", "XLE", "OIH", "XOP", "WTI", "Brent", "CL=F", "BZ=F"],
    "emerging": ["EPOL", "EWZ", "EEM", "VWO"],
    "europe": ["EWG", "EWO", "EWN", "EWQ", "EWU"],
    "industrial": ["CAT", "DE", "FXD"],
}

TICKER_TO_SECTOR = {}
for sector, tickers in SECTOR_MAP.items():
    for t in tickers:
        TICKER_TO_SECTOR[t] = sector

# Max % of holdings allowed in any single sector
MAX_SECTOR_PCT = float(os.getenv("MAX_SECTOR_PCT", "40")) / 100.0
# Max new positions per day
MAX_DAILY_OPENS = int(os.getenv("MAX_DAILY_OPENS", "4"))


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
        self._opens_today = 0
        self._opens_date = None

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
            self._check_sector_concentration(ticker, position_value, portfolio),
            self._check_daily_limit(ticker),
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

    def _check_sector_concentration(
        self, ticker: str, position_value: float, portfolio
    ) -> Tuple[bool, str]:
        """Block new position if its sector would exceed MAX_SECTOR_PCT of holdings."""
        sector = TICKER_TO_SECTOR.get(ticker)
        if not sector:
            return True, "OK (no sector mapping)"

        total_holdings = sum(pos.position_value for pos in portfolio.positions.values())
        if total_holdings <= 0:
            return True, "OK"

        sector_holdings = 0.0
        for pos_ticker, pos in portfolio.positions.items():
            if TICKER_TO_SECTOR.get(pos_ticker) == sector:
                sector_holdings += pos.position_value

        new_sector_total = sector_holdings + position_value
        new_total = total_holdings + position_value
        pct = new_sector_total / new_total if new_total > 0 else 0

        if pct > MAX_SECTOR_PCT:
            return (
                False,
                f"Sector '{sector}' would be {pct*100:.0f}% of holdings (limit {MAX_SECTOR_PCT*100:.0f}%)",
            )
        return True, f"OK (sector '{sector}' at {pct*100:.0f}%)"

    def _check_daily_limit(self, ticker: str) -> Tuple[bool, str]:
        """Limit number of new positions opened per day."""
        from datetime import date
        today = date.today().isoformat()
        if self._opens_date != today:
            self._opens_date = today
            self._opens_today = 0

        if self._opens_today >= MAX_DAILY_OPENS:
            return (
                False,
                f"Daily open limit reached ({MAX_DAILY_OPENS}/day), {ticker} blocked",
            )

        self._opens_today += 1
        return True, f"OK (daily open #{self._opens_today}/{MAX_DAILY_OPENS})"
