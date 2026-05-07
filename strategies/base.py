"""
Base strategy protocol and data structures for QuantTrade.

All strategies must implement the BaseStrategy protocol to ensure
consistent integration with backtesting, paper trading, and dashboard.
"""
from typing import Protocol, Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Direction(Enum):
    """Signal direction."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"
    NEUTRAL = "neutral"


class AssetType(Enum):
    """Asset types for trade mapping."""
    EQUITY = "equity"
    ETF = "etf"
    COMMODITY_FUTURES = "commodity_futures"
    CRYPTO = "crypto"
    FOREX = "forex"
    BOND = "bond"
    INDEX = "index"


@dataclass
class ResearchSignal:
    """
    Research-level signal expressing an economic observation.

    This is NOT a trading recommendation. It represents a data-driven
    thesis about market conditions derived from alternative data.

    Examples:
    - "Auto inventory in Texas increased 130% vs baseline"
    - "Chokepoint throughput decreased 40% week-over-week"
    - "Oil storage levels at Cushing are 5th percentile low"

    These observations must be mapped to tradable expressions via
    TradeCandidate through execution/trade_mapper.py.
    """
    # Identity
    strategy: str
    timestamp: str
    region: str

    # Signal assessment
    direction: Direction
    strength: float  # Raw signal strength (e.g., z-score, % change)
    confidence: float  # 0-1, model's confidence in signal validity

    # Data quality (independent of signal strength)
    data_quality: float  # 0-1, overall quality score
    sample_count: int  # Number of observations
    coverage_ratio: float  # Spatial/temporal coverage

    # Economic thesis
    thesis: str  # Human-readable explanation
    raw_value: Optional[float] = None  # Original observed value
    baseline_value: Optional[float] = None  # Baseline for comparison
    percentile_rank: Optional[float] = None  # Historical percentile

    # Metadata
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    detection_date: Optional[str] = None
    age_days: Optional[int] = None  # Signal age since detection

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/API."""
        return {
            "strategy": self.strategy,
            "timestamp": self.timestamp,
            "region": self.region,
            "direction": self.direction.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "data_quality": self.data_quality,
            "sample_count": self.sample_count,
            "coverage_ratio": self.coverage_ratio,
            "thesis": self.thesis,
            "raw_value": self.raw_value,
            "baseline_value": self.baseline_value,
            "percentile_rank": self.percentile_rank,
            "feature_contributions": self.feature_contributions,
            "detection_date": self.detection_date,
            "age_days": self.age_days,
        }


@dataclass
class TradeCandidate:
    """
    Tradable expression of a research signal.

    This is the output of the execution/trade_mapper.py layer that
    converts ResearchSignal into actionable trade ideas.

    A single ResearchSignal may map to zero, one, or multiple TradeCandidates.
    Multiple ResearchSignals may map to a single TradeCandidate (aggregation).

    Risk management parameters are included but can be overridden by
    portfolio-level rules in execution/portfolio_rules.py.
    """
    # Identity
    strategy: str
    timestamp: str
    ticker: str  # Tradable symbol
    asset_type: AssetType

    # Trade parameters
    direction: Direction
    horizon_days: int  # Expected holding period
    size_pct: float  # Portfolio allocation % (pre-risk-limits)

    # Risk management
    stop_loss_pct: float  # Stop loss as % of entry price
    take_profit_pct: float  # Take profit as % of entry price

    # Rationale
    rationale: str  # Why this trade makes sense
    source_signals: List[str] = field(default_factory=list)  # IDs of source ResearchSignals

    # Optional metadata
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    expected_return: Optional[float] = None  # Expected return in %
    probability: Optional[float] = None  # Success probability 0-1

    # Execution hints
    limit_order: bool = False
    time_in_force: str = "day"  # day, gtc, ioc, fok

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/API."""
        return {
            "strategy": self.strategy,
            "timestamp": self.timestamp,
            "ticker": self.ticker,
            "asset_type": self.asset_type.value,
            "direction": self.direction.value,
            "horizon_days": self.horizon_days,
            "size_pct": self.size_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "rationale": self.rationale,
            "source_signals": self.source_signals,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "expected_return": self.expected_return,
            "probability": self.probability,
            "limit_order": self.limit_order,
            "time_in_force": self.time_in_force,
        }


class BaseStrategy(Protocol):
    """
    Protocol that all trading strategies must implement.

    This ensures consistent integration across:
    - Backtesting (research/backtest/)
    - Paper trading (paper_trading/)
    - Dashboard (dashboard/)
    - API (backend/api/)

    Each strategy should be self-contained and define its own:
    - Data requirements
    - Feature engineering
    - Signal generation logic
    - Confidence estimation
    - Trade mapping rules
    """

    # Strategy identity
    name: str
    version: str
    description: str

    def load_inputs(
        self,
        start_date: str,
        end_date: str,
        region: Optional[str] = None,
    ) -> "pd.DataFrame":
        """
        Load raw input data for the strategy.

        Args:
            start_date: ISO date string (YYYY-MM-DD)
            end_date: ISO date string (YYYY-MM-DD)
            region: Optional region filter

        Returns:
            DataFrame with raw input data. Schema is strategy-specific.
            Must include at minimum: date, region (if multi-region)
        """
        ...

    def build_features(self, raw_df: "pd.DataFrame") -> "pd.DataFrame":
        """
        Transform raw inputs into features.

        Args:
            raw_df: Raw data from load_inputs()

        Returns:
            DataFrame with engineered features. Must include:
            - date, region (if applicable)
            - All feature columns used by generate_signal()
        """
        ...

    def generate_signal(self, feature_df: "pd.DataFrame") -> "pd.DataFrame":
        """
        Generate research signals from features.

        Args:
            feature_df: Feature data from build_features()

        Returns:
            DataFrame with one row per signal containing:
            - date, region, direction, strength, thesis
            - Any intermediate values used for confidence estimation
        """
        ...

    def estimate_confidence(self, signal_df: "pd.DataFrame") -> "pd.DataFrame":
        """
        Estimate confidence for each signal.

        Should consider:
        - Data quality (coverage, sample count, recency)
        - Signal strength distribution
        - Historical performance (if available)

        Args:
            signal_df: Output from generate_signal()

        Returns:
            Same DataFrame with additional columns:
            - confidence: float 0-1
            - data_quality: float 0-1
        """
        ...

    def map_to_trade(
        self,
        signal_df: "pd.DataFrame",
    ) -> List["TradeCandidate"]:
        """
        Convert research signals to trade candidates.

        This is where the "economic observation" becomes "tradable idea".

        Args:
            signal_df: Output from estimate_confidence()

        Returns:
            List of TradeCandidate objects. May be empty if no
            signals meet trading thresholds.
        """
        ...

    def get_config(self) -> Dict[str, Any]:
        """
        Get strategy configuration parameters.

        Returns:
            Dict with configurable parameters:
            - signal_thresholds: min_strength, min_quality, etc.
            - trade_params: default_horizon, default_size, etc.
            - risk_params: default_stop_loss, default_take_profit, etc.
        """
        ...


@dataclass
class StrategyConfig:
    """
    Default configuration structure for strategies.

    Each strategy can extend this with custom parameters.
    """
    # Signal generation thresholds
    min_quality_score: float = 0.5
    min_confidence: float = 0.5
    min_sample_count: int = 3

    # Direction thresholds (e.g., z-score cutoffs)
    long_threshold: float = -1.5  # Negative for "low is bullish" (e.g., inventory)
    short_threshold: float = 1.5   # Positive for "high is bearish"

    # Default trade parameters
    default_horizon_days: int = 20
    default_size_pct: float = 0.02  # 2% of portfolio
    default_stop_loss_pct: float = 0.05  # 5%
    default_take_profit_pct: float = 0.10  # 10%

    # Feature configuration
    use_seasonal_adjustment: bool = True
    use_cross_region_normalization: bool = True
    min_coverage_ratio: float = 0.3

    # Data quality weights
    coverage_weight: float = 0.3
    recency_weight: float = 0.2
    sample_weight: float = 0.2
    detector_weight: float = 0.3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_quality_score": self.min_quality_score,
            "min_confidence": self.min_confidence,
            "min_sample_count": self.min_sample_count,
            "long_threshold": self.long_threshold,
            "short_threshold": self.short_threshold,
            "default_horizon_days": self.default_horizon_days,
            "default_size_pct": self.default_size_pct,
            "default_stop_loss_pct": self.default_stop_loss_pct,
            "default_take_profit_pct": self.default_take_profit_pct,
            "use_seasonal_adjustment": self.use_seasonal_adjustment,
            "use_cross_region_normalization": self.use_cross_region_normalization,
            "min_coverage_ratio": self.min_coverage_ratio,
            "coverage_weight": self.coverage_weight,
            "recency_weight": self.recency_weight,
            "sample_weight": self.sample_weight,
            "detector_weight": self.detector_weight,
        }
