"""
Historical market data loading and caching.
"""

from pathlib import Path

import pandas as pd


def _cache_path(output_base: str, symbol: str) -> Path:
    return Path(output_base) / "market_data" / f"{symbol}.parquet"


def load_cached_prices(output_base: str, symbol: str) -> pd.DataFrame:
    cache_path = _cache_path(output_base, symbol)
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return pd.DataFrame()


def fetch_yahoo_prices(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    output_base: str = "outputs",
    refresh: bool = False,
) -> pd.DataFrame:
    cached = load_cached_prices(output_base, symbol)
    if not cached.empty and not refresh:
        return cached

    import yfinance as yf

    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False, interval="1d")
    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [column[0] for column in df.columns]

    df = df.reset_index().rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    cache_path = _cache_path(output_base, symbol)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df
