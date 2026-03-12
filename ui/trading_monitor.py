"""
Real-time Trading Monitor with auto-refresh
Real-time Trading Monitor with auto-refresh

Run with:
  streamlit run ui/trading_monitor.py
"""

from pathlib import Path
import json
import sys
from datetime import datetime
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from pipeline.price_feed import fetch_price_yahoo


# Page config
st.set_page_config(
    page_title="QuantTrade Real-Time Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .position-card {
        background-color: #fff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin-bottom: 10px;
    }
    .short-position {
        border-left-color: #f44336;
    }
    .long-position {
        border-left-color: #4CAF50;
    }
    .refresh-counter {
        position: fixed;
        top: 60px;
        right: 20px;
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 20px;
        font-weight: bold;
        z-index: 999;
    }
</style>
""", unsafe_allow_html=True)


def load_portfolio():
    """Load portfolio data"""
    portfolio_path = PROJECT_ROOT / "outputs" / "paper_trading" / "multi_asset_portfolio.json"
    if portfolio_path.exists():
        return json.loads(portfolio_path.read_text())
    return None


def get_current_prices(tickers):
    """Fetch current prices for given tickers"""
    prices = {}
    for ticker in tickers:
        try:
            price = fetch_price_yahoo(ticker)
            prices[ticker] = price
        except Exception as e:
            st.warning(f"Failed to fetch price for {ticker}: {e}")
            prices[ticker] = None
    return prices


def calculate_pnl(position, current_price):
    """Calculate P&L for a position"""
    entry_price = position.get("entry_price", 0)
    quantity = position.get("quantity", 0)
    direction = position.get("direction", "long")
    
    if direction == "long":
        pnl = (current_price - entry_price) * quantity
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
    else:  # short
        pnl = (entry_price - current_price) * quantity
        pnl_pct = ((entry_price - current_price) / entry_price) * 100
    
    return pnl, pnl_pct


def main():
    st.title("📊 QuantTrade Real-Time Trading Monitor")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Auto-refresh configuration
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 300, 900, 10)  # Default 15 min
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    
    # Load portfolio
    portfolio = load_portfolio()
    
    if not portfolio:
        st.error("❌ No portfolio data found!")
        st.info("Please run the trading system first to create a portfolio.")
        return
    
    # Get current prices for all positions
    positions = portfolio.get("positions", {})
    tickers = list(positions.keys())
    
    with st.spinner("Fetching current prices..."):
        current_prices = get_current_prices(tickers)
    
    # Calculate total portfolio value
    cash = portfolio.get("cash", 0)
    total_position_value = 0
    total_pnl = 0
    
    position_data = []
    for ticker, pos in positions.items():
        current_price = current_prices.get(ticker, pos.get("entry_price", 0))
        pnl, pnl_pct = calculate_pnl(pos, current_price)
        
        position_value = current_price * pos.get("quantity", 0)
        total_position_value += position_value
        total_pnl += pnl
        
        position_data.append({
            "Ticker": ticker,
            "Direction": pos.get("direction", "long").upper(),
            "Quantity": pos.get("quantity", 0),
            "Entry Price": pos.get("entry_price", 0),
            "Current Price": current_price,
            "Position Value": position_value,
            "P&L": pnl,
            "P&L %": pnl_pct,
            "Stop Loss": pos.get("stop_loss", 0),
            "Take Profit": pos.get("take_profit", 0),
            "Signal Grade": pos.get("signal_grade", "N/A"),
            "Accuracy": f"{pos.get('signal_accuracy', 0):.0f}%"
        })
    
    total_assets = cash + total_position_value
    
    # Portfolio Summary Section
    st.markdown("---")
    st.markdown("## 💰 Portfolio Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Assets",
            f"${total_assets:,.2f}",
            f"${total_pnl:+,.2f}"
        )
    
    with col2:
        st.metric(
            "Cash",
            f"${cash:,.2f}",
            f"{(cash/total_assets*100):.1f}%"
        )
    
    with col3:
        st.metric(
            "Positions",
            f"${total_position_value:,.2f}",
            f"{(total_position_value/total_assets*100):.1f}%"
        )
    
    with col4:
        st.metric(
            "Total P&L",
            f"${total_pnl:+,.2f}",
            f"{(total_pnl/total_assets*100):+.2f}%"
        )
    
    with col5:
        # Calculate max risk
        max_risk = 0
        for ticker, pos in positions.items():
            current_price = current_prices.get(ticker, pos.get("entry_price", 0))
            stop_loss = pos.get("stop_loss", 0)
            quantity = pos.get("quantity", 0)
            direction = pos.get("direction", "long")
            
            if direction == "long":
                risk = (current_price - stop_loss) * quantity
            else:
                risk = (stop_loss - current_price) * quantity
            
            max_risk += abs(risk)
        
        st.metric(
            "Max Risk",
            f"${max_risk:,.2f}",
            f"{(max_risk/total_assets*100):.2f}%"
        )
    
    # Current Positions Section
    st.markdown("---")
    st.markdown("## 📈 Current Positions")
    
    if position_data:
        df_positions = pd.DataFrame(position_data)
        
        # Style the dataframe
        def style_pnl(val):
            if val > 0:
                return 'color: green'
            elif val < 0:
                return 'color: red'
            return ''
        
        styled_df = df_positions.style.applymap(
            style_pnl,
            subset=['P&L', 'P&L %']
        ).format({
            'Entry Price': '${:.2f}',
            'Current Price': '${:.2f}',
            'Position Value': '${:,.2f}',
            'P&L': '${:+,.2f}',
            'P&L %': '{:+.2f}%',
            'Stop Loss': '${:.2f}',
            'Take Profit': '${:.2f}',
            'Quantity': '{:.2f}'
        })
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Position cards
        st.markdown("### Position Details")
        for pos in position_data:
            ticker = pos["Ticker"]
            direction = pos["Direction"]
            pnl = pos["P&L"]
            pnl_pct = pos["P&L %"]
            
            # Determine color based on direction
            card_class = "short-position" if direction == "SHORT" else "long-position"
            
            # Determine P&L color
            pnl_color = "green" if pnl >= 0 else "red"
            
            st.markdown(f"""
            <div class="position-card {card_class}">
                <h4>{ticker} - {direction}</h4>
                <p><strong>Entry:</strong> ${pos['Entry Price']:.2f} | 
                   <strong>Current:</strong> ${pos['Current Price']:.2f} | 
                   <strong>Value:</strong> ${pos['Position Value']:,.2f}</p>
                <p style="color: {pnl_color}; font-size: 18px; font-weight: bold;">
                   P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)
                </p>
                <p><strong>Stop:</strong> ${pos['Stop Loss']:.2f} | 
                   <strong>Target:</strong> ${pos['Take Profit']:.2f}</p>
                <p><strong>Signal Grade:</strong> {pos['Signal Grade']} ⭐ | 
                   <strong>Accuracy:</strong> {pos['Accuracy']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No open positions")
    
    # Today's Trades Section
    st.markdown("---")
    st.markdown("## 📝 Today's Trades")
    
    trades = portfolio.get("trades", [])
    today = datetime.now().strftime("%Y-%m-%d")
    today_trades = [t for t in trades if t.get("date") == today]
    
    if today_trades:
        df_trades = pd.DataFrame(today_trades)
        df_trades = df_trades[["date", "ticker", "action", "price", "quantity", "value", "rationale"]]
        df_trades.columns = ["Date", "Ticker", "Action", "Price", "Quantity", "Value", "Rationale"]
        
        st.dataframe(
            df_trades.style.format({
                'Price': '${:.2f}',
                'Value': '${:,.2f}',
                'Quantity': '{:.2f}'
            }),
            use_container_width=True
        )
    else:
        st.info("No trades today")
    
    # All Trade History
    if trades:
        st.markdown("### 📊 All Trade History")
        df_all_trades = pd.DataFrame(trades)
        st.dataframe(df_all_trades, use_container_width=True)
    
    # Risk Management Section
    st.markdown("---")
    st.markdown("## ⚠️ Risk Management")
    
    # Check for stop loss / take profit triggers
    alerts = []
    for ticker, pos in positions.items():
        current_price = current_prices.get(ticker, pos.get("entry_price", 0))
        stop_loss = pos.get("stop_loss", 0)
        take_profit = pos.get("take_profit", 0)
        direction = pos.get("direction", "long")
        
        if direction == "long":
            if current_price <= stop_loss:
                alerts.append(f"🔴 {ticker}: Stop loss triggered! Current ${current_price:.2f} <= Stop ${stop_loss:.2f}")
            elif current_price >= take_profit:
                alerts.append(f"🟢 {ticker}: Take profit target hit! Current ${current_price:.2f} >= Target ${take_profit:.2f}")
        else:  # short
            if current_price >= stop_loss:
                alerts.append(f"🔴 {ticker}: Stop loss triggered! Current ${current_price:.2f} >= Stop ${stop_loss:.2f}")
            elif current_price <= take_profit:
                alerts.append(f"🟢 {ticker}: Take profit target hit! Current ${current_price:.2f} <= Target ${take_profit:.2f}")
    
    if alerts:
        st.error("⚠️ ACTIVE ALERTS:")
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("✅ No alerts - all positions within risk parameters")
    
    # Auto-refresh logic
    if auto_refresh:
        st.markdown("---")
        st.info(f"🔄 Auto-refreshing in {refresh_interval} seconds...")
        
        # Show countdown
        countdown_placeholder = st.empty()
        for i in range(refresh_interval, 0, -1):
            countdown_placeholder.markdown(f"⏱️ Refreshing in **{i}** seconds...")
            time.sleep(1)
        
        # Trigger rerun
        st.rerun()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Monitor Info")
    st.sidebar.info(f"""
    **Portfolio:** multi_asset_portfolio.json
    
    **Total Assets:** ${total_assets:,.2f}
    
    **Positions:** {len(positions)}
    
    **Last Update:** {datetime.now().strftime('%H:%M:%S')}
    
    **Refresh Interval:** {refresh_interval}s
    """)
    
    st.sidebar.markdown("### 🎯 Quick Actions")
    if st.sidebar.button("Refresh Now"):
        st.rerun()
    
    if st.sidebar.button("View on GitHub"):
        st.markdown("[Open GitHub](https://github.com/Blurjp/QuantTrade)")


if __name__ == "__main__":
    main()
