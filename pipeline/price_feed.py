"""
Price feed integration for trading instruments.

Fetches current prices from various sources:
- Yahoo Finance (free, delayed)
- Alpha Vantage (free tier available)
- Polygon.io (crypto, stocks)
"""

from typing import Dict, Optional
from datetime import datetime, date
import json
from pathlib import Path
import time

import logging


logger = logging.getLogger(__name__)


# Fallback prices (used when API unavailable)
# Last updated: 2026-03-09 (from Yahoo Finance)
DEFAULT_PRICES = {
    # Commodities / Commodity ETFs
    "WTI": 86.0,
    "Brent": 89.0,
    "Natural Gas": 2.50,
    "Corn": 4.50,
    "Soybeans": 10.50,
    "Wheat": 5.50,
    "CORN": 18.0,
    "SOYB": 25.0,
    "WEAT": 22.0,
    "UNG": 11.0,
    "USO": 116.0,
    "BNO": 45.0,
    "BTU": 26.0,
    "WOOD": 73.0,
    
    # Energy ETFs
    "XLE": 85.0,
    "XOP": 160.0,
    "OIH": 400.0,
    "VLO": 224.0,
    "MPC": 214.0,
    "PSX": 156.0,
    "CVX": 184.0,
    "XOM": 146.0,
    "FANG": 180.0,
    "VST": 163.0,
    
    # Industrial / Sector ETFs
    "XLI": 171.0,
    "XLK": 154.0,
    "XLU": 46.0,
    "FXD": 70.0,
    "NUE": 196.0,
    "STLD": 200.0,
    "CAT": 795.0,
    "DE": 590.0,
    
    # Tech / Solar ETFs
    "TAN": 55.0,
    "ICLN": 19.0,
    "PBW": 36.0,
    "QCLN": 54.0,
    "FSLR": 190.0,
    "ENPH": 32.0,
    "SEDG": 38.0,
    "SPWR": 1.2,
    "NEE": 92.0,
    "TECL": 134.0,
    
    # International ETFs
    "FXI": 38.0,
    "MCHI": 59.0,
    "KWEB": 30.0,
    "ASHR": 35.0,
    "INDA": 51.0,
    "EPI": 45.0,
    "EWG": 43.0,
    "EPOL": 41.0,
    
    # Tech equities
    "INTC": 69.0,
    "TSM": 371.0,
    "AMD": 278.0,
    "NVDA": 202.0,
    "MSFT": 423.0,
    "GOOGL": 342.0,
    "META": 689.0,
    "AMZN": 251.0,
    "QQQ": 649.0,
    
    # Auto / Retail
    "F": 13.0,
    "GM": 81.0,
    "STLA": 9.0,
    "FB": 43.0,
    "FMC": 17.0,
    
    # Utilities
    "AEP": 134.0,
    "D": 62.0,
    "DUK": 128.0,
    "PEG": 82.0,
    
    # Legacy equities
    "WMT": 124.0,
    "COST": 1005.0,
    "TGT": 120.0,
    "HD": 360.0,
    "TM": 185.0,
    "FDX": 260.0,
    "UPS": 155.0,
    "XRT": 80.0,
    "CARZ": 28.0,
    
    # Livestock (COW delisted, use LE futures proxy)
    "LE": 160.0,  # CME Live Cattle
}

# Removed SYMBOL_ALIASES - all tickers use their yfinance names directly
# DEFAULT_PRICES keys match yfinance ticker names exactly


def _normalize_ticker(ticker: str) -> str:
    # No normalization - use ticker as-is
    return ticker


def fetch_price_yahoo(ticker: str) -> Optional[float]:
    """
    Fetch price from Yahoo Finance (free, may have rate limits).
    
    Uses yfinance library. Uses the ticker as-is for Yahoo lookup.
    """
    try:
        import yfinance as yf
        
        # Map specific names to Yahoo futures tickers
        yahoo_ticker = {
            "WTI": "CL=F",
            "Brent": "BZ=F",
            "Natural Gas": "NG=F",
        }.get(ticker, ticker)  # Use original ticker, not normalized
        
        stock = yf.Ticker(yahoo_ticker)
        hist = stock.history(period="1d")
        
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        
        return None
    
    except ImportError:
        logger.debug("yfinance not installed; using fallback price for %s", ticker)
        return None
    except Exception as e:
        logger.warning("Error fetching %s from Yahoo: %s", ticker, e)
        return None


def fetch_price_alpha_vantage(ticker: str, api_key: str = None) -> Optional[float]:
    """
    Fetch price from Alpha Vantage (free tier available).
    
    Requires API key from alphavantage.co
    """
    import os
    
    if api_key is None:
        api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    
    if not api_key:
        return None
    
    try:
        import requests
        
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": ticker,
            "apikey": api_key,
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if "Global Quote" in data:
            return float(data["Global Quote"]["05. price"])
        
        return None
    
    except Exception as e:
        print(f"Error fetching {ticker} from Alpha Vantage: {e}")
        return None


def fetch_all_prices(
    tickers: list = None,
    use_cache: bool = True,
    cache_ttl_seconds: int = 3600,
) -> Dict[str, float]:
    """
    Fetch prices for all tracked instruments.
    
    Args:
        tickers: List of tickers to fetch (default: all tracked)
        use_cache: Use cached prices if available
        cache_ttl_seconds: Cache time-to-live
    
    Returns:
        Dictionary of ticker -> price
    """
    cache_file = Path("outputs/price_cache.json")
    
    # Check cache
    if use_cache and cache_file.exists():
        cache = json.loads(cache_file.read_text())
        cached_time = datetime.fromisoformat(cache.get("timestamp", "2000-01-01"))
        age = (datetime.now() - cached_time).total_seconds()
        
        if age < cache_ttl_seconds:
            return cache.get("prices", DEFAULT_PRICES)
    
    if tickers is None:
        tickers = list(DEFAULT_PRICES.keys())
    
    prices = {}
    
    for ticker in tickers:
        # Try Yahoo first
        price = fetch_price_yahoo(ticker)
        
        if price is None:
            # Fall back to default
            price = DEFAULT_PRICES.get(_normalize_ticker(ticker))
        
        if price is not None:
            prices[ticker] = price
        else:
            # Use default if all else fails
            prices[ticker] = DEFAULT_PRICES.get(_normalize_ticker(ticker), 0)
        
        # Rate limiting
        time.sleep(0.1)
    
    # Save cache (only non-zero, non-None values)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cached_prices = {k: v for k, v in prices.items() if v and v > 0}
    cache_file.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "prices": cached_prices,
    }, indent=2))
    
    return prices


def get_price(ticker: str) -> float:
    """
    Get price for a single ticker.
    
    Tries Yahoo Finance first, falls back to defaults.
    """
    price = fetch_price_yahoo(ticker)
    
    if price is None:
        price = DEFAULT_PRICES.get(_normalize_ticker(ticker), 0)
    
    return price


def get_prices_for_portfolio(portfolio) -> Dict[str, float]:
    """
    Get prices for all instruments in portfolio.
    """
    tickers = set()

    # Add tickers from open positions
    for ticker in portfolio.positions.keys():
        tickers.add(ticker)

    # Add tickers from all active regions
    from pipeline.regions import load_registry
    try:
        registry = load_registry()
        for region_config in registry.get("regions", {}).values():
            if region_config.get("active"):
                instruments = region_config.get("instruments", [])
                for inst in instruments:
                    if isinstance(inst, dict):
                        ticker = inst.get("ticker")
                        if ticker:
                            tickers.add(ticker)
                    elif inst:
                        tickers.add(inst)
    except Exception:
        pass

    prices = fetch_all_prices(list(tickers))

    # Ensure no None values
    return {k: v if v is not None else DEFAULT_PRICES.get(k, 0) for k, v in prices.items()}


if __name__ == "__main__":
    print("Fetching prices for tracked instruments...")
    print()
    
    prices = fetch_all_prices(use_cache=False)
    
    print("Current Prices:")
    print("-" * 40)
    
    categories = {
        "Commodities": ["WTI", "Brent", "Natural Gas", "Corn", "Soybeans", "Wheat"],
        "Retail": ["WMT", "COST", "TGT", "HD"],
        "Auto": ["F", "GM", "TM"],
        "Industrial": ["CAT", "DE"],
        "Logistics": ["FDX", "UPS"],
        "ETFs": ["XLE", "XLI", "XRT", "CARZ", "USO"],
    }
    
    for category, tickers in categories.items():
        print(f"\n{category}:")
        for ticker in tickers:
            price = prices.get(ticker, 0)
            print(f"  {ticker:>12}: ${price:,.2f}")
# Force rebuild Sat Apr 18 20:51:01 EDT 2026
