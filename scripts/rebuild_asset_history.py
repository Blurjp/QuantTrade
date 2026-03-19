"""Rebuild marked-to-market asset history from portfolio activity."""

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from paper_trading.multi_asset_portfolio import MultiAssetPortfolio
from pipeline.price_feed import get_prices_for_portfolio


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _yahoo_symbol(ticker: str) -> str:
    return {
        "WTI": "CL=F",
        "Brent": "BZ=F",
        "Natural Gas": "NG=F",
        "Corn": "ZC=F",
        "Soybeans": "ZS=F",
        "Wheat": "ZW=F",
    }.get(ticker, ticker)


def _collect_rebuild_dates(portfolio: Dict) -> List[str]:
    trade_dates = {
        item.get("date")
        for item in portfolio.get("trades", [])
        if item.get("date")
    }
    snapshot_dates = {
        item.get("date")
        for item in portfolio.get("daily_snapshots", [])
        if item.get("date")
    }
    position_dates = {
        item.get("entry_date")
        for item in portfolio.get("positions", {}).values()
        if item.get("entry_date")
    }

    known_dates = sorted(trade_dates | snapshot_dates | position_dates)
    if not known_dates:
        return []

    start_day = datetime.strptime(known_dates[0], "%Y-%m-%d").date()
    end_candidates = [datetime.strptime(known_dates[-1], "%Y-%m-%d").date()]
    if portfolio.get("positions"):
        end_candidates.append(date.today())
    end_day = max(end_candidates)

    return [
        ts.strftime("%Y-%m-%d")
        for ts in pd.bdate_range(start=start_day, end=end_day)
    ]


def _fetch_history(tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.Series]:
    if not tickers:
        return {}

    try:
        import yfinance as yf
    except Exception:
        return {}

    history = {}
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    for ticker in tickers:
        yahoo_ticker = _yahoo_symbol(ticker)
        try:
            frame = yf.download(
                yahoo_ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=False,
                progress=False,
                interval="1d",
            )
        except Exception:
            frame = pd.DataFrame()

        if frame.empty or "Close" not in frame.columns:
            history[ticker] = pd.Series(dtype=float)
            continue

        close_series = frame["Close"].copy()
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]
        close_series.index = pd.to_datetime(close_series.index).normalize()
        history[ticker] = close_series.astype(float)

    return history


def _price_on_or_before(series: pd.Series, day: pd.Timestamp, fallback: float) -> float:
    if series.empty:
        return fallback

    eligible = series[series.index <= day]
    if eligible.empty:
        return fallback
    return float(eligible.iloc[-1]) if not isinstance(eligible.iloc[-1], pd.Series) else float(eligible.iloc[-1].iloc[0])


def rebuild_asset_history(output_base: str = "outputs", initial_capital: float = 100000.0) -> Dict:
    output_dir = PROJECT_ROOT / output_base
    portfolio_path = output_dir / "paper_trading" / "multi_asset_portfolio.json"
    portfolio = _load_json(portfolio_path)
    if not portfolio:
        raise FileNotFoundError("Portfolio file not found")

    rebuild_dates = _collect_rebuild_dates(portfolio)
    if not rebuild_dates:
        history = {"daily_assets": [], "last_updated": datetime.now().isoformat()}
        (output_dir / "asset_history.json").write_text(json.dumps(history, indent=2))
        return history

    trades = sorted(portfolio.get("trades", []), key=lambda item: (item.get("date", ""), item.get("ticker", "")))
    snapshots = {
        item.get("date"): item
        for item in portfolio.get("daily_snapshots", [])
        if item.get("date")
    }

    tickers = sorted({item.get("ticker") for item in trades if item.get("ticker")})
    tickers.extend(
        ticker for ticker in portfolio.get("positions", {}).keys()
        if ticker not in tickers
    )
    price_history = _fetch_history(tickers, rebuild_dates[0], rebuild_dates[-1])
    live_prices = get_prices_for_portfolio(MultiAssetPortfolio(output_base=output_base))
    today_str = date.today().strftime("%Y-%m-%d")

    open_positions = {}
    cash = float(initial_capital)
    trade_index = 0
    daily_assets = []

    for day_str in rebuild_dates:
        while trade_index < len(trades) and trades[trade_index].get("date") <= day_str:
            trade = trades[trade_index]
            ticker = trade.get("ticker")
            action = trade.get("action")

            if action in {"OPEN_LONG", "OPEN_SHORT"}:
                open_positions[ticker] = {
                    "ticker": ticker,
                    "direction": "long" if action == "OPEN_LONG" else "short",
                    "quantity": float(trade.get("quantity", 0)),
                    "entry_price": float(trade.get("price", 0)),
                    "position_value": float(trade.get("value", 0)),
                }
                cash -= float(trade.get("value", 0))
            elif action == "CLOSE" and ticker in open_positions:
                cash += float(trade.get("value", 0)) + float(trade.get("pnl", 0))
                del open_positions[ticker]

            trade_index += 1

        snapshot = snapshots.get(day_str)
        if snapshot and "total_value" in snapshot:
            total_value = float(snapshot["total_value"])
        else:
            day = pd.Timestamp(day_str)
            total_value = cash
            for ticker, position in open_positions.items():
                if day_str == today_str and ticker in live_prices:
                    raw_price = live_prices[ticker]
                    current_price = float(raw_price) if raw_price is not None else float(position["entry_price"])
                else:
                    current_price = _price_on_or_before(
                        price_history.get(ticker, pd.Series(dtype=float)),
                        day,
                        float(position["entry_price"]),
                    )
                # Ensure current_price is valid
                if current_price is None:
                    current_price = float(position["entry_price"])
                if position["direction"] == "long":
                    total_value += float(position["quantity"]) * current_price
                else:
                    total_value += float(position["position_value"]) + float(position["quantity"]) * (float(position["entry_price"]) - current_price)

        daily_assets.append({
            "date": day_str,
            "total_value": round(float(total_value), 2),
            "timestamp": datetime.now().isoformat(),
        })

    history = {
        "daily_assets": daily_assets,
        "last_updated": datetime.now().isoformat(),
        "source": "rebuild_asset_history.py",
    }
    (output_dir / "asset_history.json").write_text(json.dumps(history, indent=2))
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild asset history from portfolio activity")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Starting portfolio capital")
    args = parser.parse_args()

    rebuilt = rebuild_asset_history(output_base=args.output, initial_capital=args.initial_capital)
    print(f"Rebuilt {len(rebuilt.get('daily_assets', []))} asset history points")
