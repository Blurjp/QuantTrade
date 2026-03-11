"""
QuantTrade Dashboard - Real-time monitoring and visualization
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paper_trading.multi_asset_portfolio import MultiAssetPortfolio
from pipeline.price_feed import fetch_price_yahoo

# Page config
st.set_page_config(
    page_title="QuantTrade Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .positive { color: #28a745; }
    .negative { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_portfolio():
    """Load portfolio state"""
    try:
        portfolio = MultiAssetPortfolio(100000, 'outputs')
        return portfolio
    except Exception as e:
        st.error(f"Error loading portfolio: {e}")
        return None

@st.cache_data(ttl=300)
def load_backtest_results():
    """Load all backtest results"""
    backtest_dir = Path("outputs/backtest")
    results = {}
    
    if backtest_dir.exists():
        for file in backtest_dir.glob("*.json"):
            data = json.loads(file.read_text())
            key = f"{data.get('region', '?')} → {data.get('ticker', '?')}"
            results[key] = data
    
    return results

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 QuantTrade Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["📈 Portfolio", "🎯 Signals", "📊 Backtests", "🗺️ Monitoring", "⚙️ Settings"]
    )
    
    # Load data
    portfolio = load_portfolio()
    backtests = load_backtest_results()
    
    if portfolio is None:
        st.error("Unable to load portfolio data")
        return
    
    # Page routing
    if page == "📈 Portfolio":
        show_portfolio(portfolio)
    elif page == "🎯 Signals":
        show_signals(backtests)
    elif page == "📊 Backtests":
        show_backtests(backtests)
    elif page == "🗺️ Monitoring":
        show_monitoring()
    elif page == "⚙️ Settings":
        show_settings()

def show_portfolio(portfolio):
    """Display portfolio overview"""
    st.header("Portfolio Overview")
    
    # Get current prices
    wti_price = fetch_price_yahoo("WTI") or 86.0
    f_price = fetch_price_yahoo("F") or 12.20
    
    prices = {"WTI": wti_price, "F": f_price}
    
    # Calculate total value and P&L
    total_pnl = 0
    for ticker, pos in portfolio.positions.items():
        price = prices.get(ticker, 100)
        if pos.direction == "short":
            pnl = (pos.entry_price - price) / pos.entry_price * pos.position_value
        else:
            pnl = (price - pos.entry_price) / pos.entry_price * pos.position_value
        total_pnl += pnl
    
    total_value = portfolio.cash + sum(pos.position_value for pos in portfolio.positions.values()) + total_pnl
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Value",
            f"${total_value:,.2f}",
            f"{(total_value - portfolio.initial_capital) / portfolio.initial_capital * 100:+.2f}%"
        )
    
    with col2:
        st.metric(
            "Cash",
            f"${portfolio.cash:,.2f}"
        )
    
    with col3:
        st.metric(
            "Total P&L",
            f"${total_pnl:+,.2f}"
        )
    
    with col4:
        st.metric(
            "Positions",
            len(portfolio.positions)
        )
    
    st.divider()
    
    # Positions table
    st.subheader("Current Positions")
    
    positions_data = []
    for ticker, pos in portfolio.positions.items():
        price = prices.get(ticker, 100)
        if pos.direction == "short":
            pnl = (pos.entry_price - price) / pos.entry_price * pos.position_value
        else:
            pnl = (price - pos.entry_price) / pos.entry_price * pos.position_value
        
        pnl_pct = pnl / pos.position_value * 100
        
        positions_data.append({
            "Ticker": ticker,
            "Direction": pos.direction.upper(),
            "Entry": f"${pos.entry_price:.2f}",
            "Current": f"${price:.2f}",
            "Size": f"${pos.position_value:,.2f}",
            "P&L": f"${pnl:+,.2f}",
            "P&L %": f"{pnl_pct:+.2f}%",
            "Stop Loss": f"${pos.stop_loss:.2f}",
            "Take Profit": f"${pos.take_profit:.2f}"
        })
    
    if positions_data:
        df = pd.DataFrame(positions_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No open positions")
    
    # Portfolio allocation chart
    st.subheader("Portfolio Allocation")
    
    if portfolio.positions:
        labels = ["Cash"] + list(portfolio.positions.keys())
        values = [portfolio.cash] + [pos.position_value for pos in portfolio.positions.values()]
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_signals(backtests):
    """Display signal quality and performance"""
    st.header("Signal Performance")
    
    if not backtests:
        st.info("No backtest data available")
        return
    
    # Signal accuracy comparison
    st.subheader("Signal Accuracy by Target")
    
    accuracy_data = []
    for key, data in backtests.items():
        backtest = data.get("backtest", {})
        accuracy_data.append({
            "Target": key,
            "Total Signals": backtest.get("total_signals", 0),
            "Overall Accuracy": f"{backtest.get('overall_accuracy', 0) * 100:.1f}%",
            "Avg Return": f"{backtest.get('avg_return', 0) * 100:.2f}%"
        })
    
    df = pd.DataFrame(accuracy_data)
    st.dataframe(df, use_container_width=True)
    
    # Visual comparison
    st.subheader("Accuracy Comparison")
    
    fig_data = []
    for key, data in backtests.items():
        backtest = data.get("backtest", {})
        fig_data.append({
            "Target": key,
            "Accuracy": backtest.get("overall_accuracy", 0) * 100
        })
    
    if fig_data:
        df_fig = pd.DataFrame(fig_data)
        fig = px.bar(df_fig, x="Target", y="Accuracy", color="Accuracy",
                     color_continuous_scale="RdYlGn", range_color=[0, 100])
        fig.update_layout(height=400, yaxis_title="Accuracy (%)")
        st.plotly_chart(fig, use_container_width=True)

def show_backtests(backtests):
    """Display detailed backtest results"""
    st.header("Backtest Results")
    
    if not backtests:
        st.info("No backtest data available")
        return
    
    # Select backtest
    selected = st.selectbox("Select backtest", list(backtests.keys()))
    
    if selected:
        data = backtests[selected]
        backtest = data.get("backtest", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Signals", backtest.get("total_signals", 0))
            st.metric("Overall Accuracy", f"{backtest.get('overall_accuracy', 0) * 100:.1f}%")
            st.metric("Avg Return", f"{backtest.get('avg_return', 0) * 100:.2f}%")
        
        with col2:
            # Direction breakdown
            st.subheader("By Direction")
            by_dir = backtest.get("by_direction", {})
            
            for direction, stats in by_dir.items():
                st.write(f"**{direction}**")
                st.write(f"  Signals: {stats.get('count', 0)}")
                st.write(f"  Accuracy: {stats.get('accuracy', 0) * 100:.1f}%")
                st.write(f"  Avg Return: {stats.get('avg_return', 0) * 100:.2f}%")
                st.write("")

def show_monitoring():
    """Display monitoring targets"""
    st.header("Monitoring Targets")
    
    # Load registry
    registry_path = Path("configs/regions/registry_v2.json")
    if not registry_path.exists():
        registry_path = Path("configs/regions/registry.json")
    
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
        
        for target_id, target in registry.get("targets", {}).items():
            with st.expander(f"📍 {target.get('name', target_id)}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Type**: {target.get('type', 'N/A')}")
                    st.write(f"**Region**: {target.get('region', 'N/A')}")
                    st.write(f"**Ticker**: {target.get('ticker', 'N/A')}")
                
                with col2:
                    st.write(f"**Active**: {'✅' if target.get('active', False) else '❌'}")
                    st.write(f"**Priority**: {target.get('priority', 'N/A')}")
    else:
        st.info("No registry found")

def show_settings():
    """Display settings and configuration"""
    st.header("Settings")
    
    st.subheader("System Configuration")
    
    # Portfolio settings
    st.write("**Portfolio Settings**")
    st.write("• Initial Capital: $100,000")
    st.write("• Max Position Size: 10%")
    st.write("• Max Sector Exposure: 25%")
    
    st.divider()
    
    # Automation settings
    st.write("**Automation**")
    st.write("• Daily Update: 6:00 AM EST")
    st.write("• Stop Loss Check: 9:30 AM EST (market open)")
    
    st.divider()
    
    # Data sources
    st.write("**Data Sources**")
    st.write("• Sentinel-1 (SAR): Planetary Computer")
    st.write("• Sentinel-2 (Optical): Planetary Computer")
    st.write("• Prices: Yahoo Finance")

if __name__ == "__main__":
    main()
