# Page configuration - MUST BE FIRST!
import streamlit as st

st.set_page_config(
    page_title="Crypto Portfolio Optimizer",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from yahooquery import Ticker
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from portfolio_optimizer import PortfolioOptimizer
from robust_data_fetcher import RobustDataFetcher

# Initialize session state for theme - default to light mode
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Apply theme-specific CSS (Coinbase-inspired)
def apply_theme_css(theme):
    if theme == 'light':
        return """
<style>
    /* Import Coinbase Sans font family */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Coinbase Light Mode Variables */
    :root {
        --cb-primary-bg: #ffffff;
        --cb-secondary-bg: #f8f9fa;
        --cb-card-bg: #ffffff;
        --cb-hover-bg: #f0f2f5;
        --cb-border: #e6e8ea;
        --cb-text-primary: #050f19;
        --cb-text-secondary: #5b616e;
        --cb-text-tertiary: #8a919e;
        --cb-blue: #1652f0;
        --cb-green: #00d395;
        --cb-red: #fa3c58;
        --cb-orange: #ff9500;
        --cb-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        --cb-shadow-hover: 0 4px 16px rgba(0, 0, 0, 0.15);
        --cb-radius: 8px;
        --cb-radius-lg: 12px;
    }
    
    /* Global Coinbase Light Theme */
    .stApp {
        font-family: 'Inter', 'Coinbase Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: var(--cb-secondary-bg);
        color: var(--cb-text-primary);
        font-feature-settings: 'tnum';
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Coinbase-style main container */
    .main .block-container {
        padding: 1.5rem 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Coinbase-style sidebar */
    .css-1d391kg {
        background: var(--cb-card-bg) !important;
        border-right: 1px solid var(--cb-border) !important;
        box-shadow: var(--cb-shadow);
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: var(--cb-text-primary) !important;
    }
    
    .css-1d391kg label {
        color: var(--cb-text-primary) !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: var(--cb-text-primary) !important;
        font-weight: 600 !important;
    }
    
    .css-1d391kg p {
        color: var(--cb-text-secondary) !important;
        font-size: 0.875rem !important;
    }
    
    /* Coinbase-style buttons */
    .stButton > button {
        background: var(--cb-blue);
        color: white;
        border: none;
        border-radius: var(--cb-radius);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        box-shadow: none;
        text-transform: none;
        letter-spacing: normal;
    }
    
    .stButton > button:hover {
        background: #1347cc;
        transform: none;
        box-shadow: var(--cb-shadow-hover);
    }
    
    .stButton > button:focus {
        box-shadow: 0 0 0 3px rgba(22, 82, 240, 0.1);
        outline: none;
    }
    
    /* Coinbase-style inputs */
    .stSelectbox > div > div {
        background: var(--cb-card-bg);
        border: 1px solid var(--cb-border);
        border-radius: var(--cb-radius);
        transition: all 0.2s ease;
        font-size: 0.875rem;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--cb-blue);
        box-shadow: 0 0 0 3px rgba(22, 82, 240, 0.1);
    }
    
    .stNumberInput > div > div > input {
        background: var(--cb-card-bg);
        border: 1px solid var(--cb-border);
        border-radius: var(--cb-radius);
        transition: all 0.2s ease;
        font-size: 0.875rem;
        font-feature-settings: 'tnum';
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: var(--cb-blue);
        box-shadow: 0 0 0 3px rgba(22, 82, 240, 0.1);
        outline: none;
    }
    
    .stMultiSelect > div > div {
        background: var(--cb-card-bg);
        border: 1px solid var(--cb-border);
        border-radius: var(--cb-radius);
        font-size: 0.875rem;
    }
    
    .stDateInput > div > div > input {
        background: var(--cb-card-bg);
        border: 1px solid var(--cb-border);
        border-radius: var(--cb-radius);
        font-size: 0.875rem;
    }
    
    /* Style multiselect selected items with white background */
    .stMultiSelect div[data-baseweb="tag"],
    .stMultiSelect > div > div > div[data-baseweb="tag"],
    div[data-baseweb="tag"] {
        background-color: white !important;
        background: white !important;
        color: #050f19 !important;
        border: 1px solid var(--cb-border) !important;
        border-radius: 4px !important;
    }
    
    .stMultiSelect div[data-baseweb="tag"] span,
    .stMultiSelect > div > div > div[data-baseweb="tag"] span,
    div[data-baseweb="tag"] span {
        color: #050f19 !important;
    }
    
    /* Style the close button in multiselect tags */
    .stMultiSelect div[data-baseweb="tag"] div[role="button"],
    .stMultiSelect > div > div > div[data-baseweb="tag"] div[role="button"] {
        color: #5b616e !important;
    }
    
    .stMultiSelect div[data-baseweb="tag"] div[role="button"]:hover,
    .stMultiSelect > div > div > div[data-baseweb="tag"] div[role="button"]:hover {
        color: #050f19 !important;
    }
    
    /* Override any Streamlit default red backgrounds for tags */
    [data-baseweb="tag"] {
        background-color: white !important;
        background: white !important;
        color: #050f19 !important;
    }
    
    /* More specific targeting for all tag text elements */
    [data-baseweb="tag"] span,
    [data-baseweb="tag"] div,
    [data-baseweb="tag"] * {
        color: #050f19 !important;
    }
    
    /* Target default selected items specifically */
    .stMultiSelect [data-baseweb="tag"] {
        background: white !important;
        color: #050f19 !important;
    }
    
    .stMultiSelect [data-baseweb="tag"] span {
        color: #050f19 !important;
    }
    
    /* Coinbase-style tables */
    .stDataFrame {
        background: var(--cb-card-bg) !important;
        box-shadow: var(--cb-shadow);
        border-radius: var(--cb-radius-lg);
        border: 1px solid var(--cb-border);
        overflow: hidden;
    }
    
    .stDataFrame table {
        background: transparent !important;
        color: var(--cb-text-primary) !important;
        font-size: 0.875rem !important;
    }
    
    .stDataFrame th {
        background: var(--cb-secondary-bg) !important;
        color: var(--cb-text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 1px solid var(--cb-border) !important;
        padding: 1rem 0.75rem !important;
    }
    
    .stDataFrame td {
        background: transparent !important;
        color: var(--cb-text-primary) !important;
        border-bottom: 1px solid var(--cb-border) !important;
        padding: 1rem 0.75rem !important;
        font-feature-settings: 'tnum';
    }
    
    .stDataFrame tr:hover td {
        background: var(--cb-hover-bg) !important;
    }
    
    /* Financial data styling */
    .positive-value {
        color: var(--cb-green) !important;
        font-weight: 600 !important;
    }
    
    .negative-value {
        color: var(--cb-red) !important;
        font-weight: 600 !important;
    }
    
    /* Metric cards */
    .stMetric {
        background: var(--cb-card-bg);
        padding: 1.5rem;
        border-radius: var(--cb-radius-lg);
        box-shadow: var(--cb-shadow);
        border: 1px solid var(--cb-border);
        transition: all 0.2s ease;
    }
    
    .stMetric:hover {
        box-shadow: var(--cb-shadow-hover);
        transform: translateY(-1px);
    }
    
    .stMetric [data-testid="metric-value"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: var(--cb-text-primary) !important;
        font-feature-settings: 'tnum';
    }
    
    .stMetric [data-testid="metric-label"] {
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        color: var(--cb-text-secondary) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Alerts */
    .stSuccess {
        background: rgba(0, 211, 149, 0.1) !important;
        border: 1px solid var(--cb-green) !important;
        border-radius: var(--cb-radius) !important;
        color: var(--cb-green) !important;
    }
    
    .stError {
        background: rgba(250, 60, 88, 0.1) !important;
        border: 1px solid var(--cb-red) !important;
        border-radius: var(--cb-radius) !important;
        color: var(--cb-red) !important;
    }
    
    .stInfo {
        background: rgba(22, 82, 240, 0.1) !important;
        border: 1px solid var(--cb-blue) !important;
        border-radius: var(--cb-radius) !important;
        color: var(--cb-blue) !important;
    }
</style>
"""
    else:  # dark theme (Coinbase-inspired)
        return """
<style>
    /* Import Coinbase Sans font family */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Coinbase Dark Mode Variables */
    :root {
        --cb-primary-bg: #0a0e27;
        --cb-secondary-bg: #162d50;
        --cb-card-bg: #1a1f3a;
        --cb-hover-bg: #243056;
        --cb-border: #2d3748;
        --cb-text-primary: #ffffff;
        --cb-text-secondary: #a0aec0;
        --cb-text-tertiary: #718096;
        --cb-blue: #1652f0;
        --cb-green: #00d395;
        --cb-red: #fa3c58;
        --cb-orange: #ff9500;
        --cb-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        --cb-shadow-hover: 0 8px 24px rgba(0, 0, 0, 0.4);
        --cb-radius: 8px;
        --cb-radius-lg: 12px;
    }
    
    /* Global Coinbase Dark Theme */
    .stApp {
        font-family: 'Inter', 'Coinbase Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: var(--cb-primary-bg);
        color: var(--cb-text-primary);
        font-feature-settings: 'tnum';
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Coinbase-style main container */
    .main .block-container {
        padding: 1.5rem 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Coinbase-style dark sidebar */
    .css-1d391kg {
        background: var(--cb-card-bg) !important;
        border-right: 1px solid var(--cb-border) !important;
        box-shadow: var(--cb-shadow);
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: var(--cb-text-primary) !important;
    }
    
    /* Sidebar text and labels */
    .css-1d391kg label {
        color: var(--cb-text-primary) !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .css-1d391kg .css-145kmo2 {
        color: var(--cb-text-primary) !important;
    }
    
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: var(--cb-text-primary) !important;
        font-weight: 600 !important;
    }
    
    .css-1d391kg p {
        color: var(--cb-text-secondary) !important;
        font-size: 0.875rem !important;
    }
    
    /* Sidebar inputs - Dark theme */
    .css-1d391kg .stSelectbox > div > div {
        background: var(--cb-secondary-bg) !important;
        border: 1px solid var(--cb-border) !important;
        color: var(--cb-text-primary) !important;
        border-radius: var(--cb-radius);
        font-size: 0.875rem;
    }
    
    .css-1d391kg .stSelectbox > div > div > div {
        color: var(--cb-text-primary) !important;
    }
    
    .css-1d391kg .stNumberInput > div > div > input {
        background: var(--cb-secondary-bg) !important;
        border: 1px solid var(--cb-border) !important;
        color: var(--cb-text-primary) !important;
        border-radius: var(--cb-radius);
        font-size: 0.875rem;
    }
    
    .css-1d391kg .stMultiSelect > div > div {
        background: var(--cb-secondary-bg) !important;
        border: 1px solid var(--cb-border) !important;
        color: var(--cb-text-primary) !important;
        border-radius: var(--cb-radius);
        font-size: 0.875rem;
    }
    
    .css-1d391kg .stDateInput > div > div > input {
        background: var(--cb-secondary-bg) !important;
        border: 1px solid var(--cb-border) !important;
        color: var(--cb-text-primary) !important;
        border-radius: var(--cb-radius);
        font-size: 0.875rem;
    }
    
    /* Coinbase-style dark buttons */
    .stButton > button {
        background: var(--cb-blue);
        color: white;
        border: none;
        border-radius: var(--cb-radius);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        box-shadow: none;
        text-transform: none;
        letter-spacing: normal;
    }
    
    .stButton > button:hover {
        background: #1347cc;
        transform: none;
        box-shadow: var(--cb-shadow-hover);
    }
    
    .stButton > button:focus {
        box-shadow: 0 0 0 3px rgba(22, 82, 240, 0.2);
        outline: none;
    }
    
    /* Main content inputs - Dark theme */
    .stSelectbox > div > div {
        background: var(--cb-card-bg);
        border: 1px solid var(--cb-border);
        border-radius: var(--cb-radius);
        color: var(--cb-text-primary);
        transition: all 0.2s ease;
        font-size: 0.875rem;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--cb-blue);
        box-shadow: 0 0 0 3px rgba(22, 82, 240, 0.2);
    }
    
    .stNumberInput > div > div > input {
        background: var(--cb-card-bg);
        border: 1px solid var(--cb-border);
        border-radius: var(--cb-radius);
        color: var(--cb-text-primary);
        transition: all 0.2s ease;
        font-size: 0.875rem;
        font-feature-settings: 'tnum';
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: var(--cb-blue);
        box-shadow: 0 0 0 3px rgba(22, 82, 240, 0.2);
        outline: none;
    }
    
    .stMultiSelect > div > div {
        background: var(--cb-card-bg);
        border: 1px solid var(--cb-border);
        border-radius: var(--cb-radius);
        font-size: 0.875rem;
    }
    
    .stDateInput > div > div > input {
        background: var(--cb-card-bg);
        border: 1px solid var(--cb-border);
        border-radius: var(--cb-radius);
        font-size: 0.875rem;
    }
    
    .stSlider > div > div > div {
        background: var(--cb-blue);
    }
    
    .stSlider > div > div > div > div {
        background: var(--cb-blue);
    }
    
    /* Coinbase-style cards */
    .cb-card {
        background: var(--cb-card-bg);
        border-radius: var(--cb-radius-lg);
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid var(--cb-border);
        transition: all 0.2s ease;
        box-shadow: var(--cb-shadow);
    }
    
    .cb-card:hover {
        background: var(--cb-hover-bg);
        transform: translateY(-1px);
        box-shadow: var(--cb-shadow-hover);
    }
    
    /* Financial Data Styling */
    .positive-value {
        color: var(--cb-green) !important;
        font-weight: 600;
    }
    
    .negative-value {
        color: var(--cb-red) !important;
        font-weight: 600;
    }
    
    /* Coinbase-style tables - Dark theme */
    .stDataFrame {
        background: var(--cb-card-bg) !important;
        box-shadow: var(--cb-shadow);
        border-radius: var(--cb-radius-lg);
        border: 1px solid var(--cb-border);
        overflow: hidden;
    }
    
    .stDataFrame table {
        background: transparent !important;
        color: var(--cb-text-primary) !important;
        font-size: 0.875rem !important;
    }
    
    .stDataFrame th {
        background: var(--cb-secondary-bg) !important;
        color: var(--cb-text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 1px solid var(--cb-border) !important;
        padding: 1rem 0.75rem !important;
    }
    
    .stDataFrame td {
        background: transparent !important;
        color: var(--cb-text-primary) !important;
        border-bottom: 1px solid var(--cb-border) !important;
        padding: 1rem 0.75rem !important;
        font-feature-settings: 'tnum';
    }
    
    .stDataFrame tr:hover td {
        background: var(--cb-hover-bg) !important;
    }
    
    /* Alerts - Coinbase dark style */
    .stSuccess {
        background: rgba(0, 211, 149, 0.1) !important;
        border: 1px solid var(--cb-green) !important;
        border-radius: var(--cb-radius) !important;
        color: var(--cb-green) !important;
    }
    
    .stError {
        background: rgba(250, 60, 88, 0.1) !important;
        border: 1px solid var(--cb-red) !important;
        border-radius: var(--cb-radius) !important;
        color: var(--cb-red) !important;
    }
    
    .stInfo {
        background: rgba(22, 82, 240, 0.1) !important;
        border: 1px solid var(--cb-blue) !important;
        border-radius: var(--cb-radius) !important;
        color: var(--cb-blue) !important;
    }
    
    /* Metric cards - Coinbase dark style */
    .stMetric {
        background: var(--cb-card-bg);
        padding: 1.5rem;
        border-radius: var(--cb-radius-lg);
        box-shadow: var(--cb-shadow);
        border: 1px solid var(--cb-border);
        transition: all 0.2s ease;
    }
    
    .stMetric:hover {
        box-shadow: var(--cb-shadow-hover);
        transform: translateY(-1px);
    }
    
    .stMetric [data-testid="metric-value"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: var(--cb-text-primary) !important;
        font-feature-settings: 'tnum';
    }
    
    .stMetric [data-testid="metric-label"] {
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        color: var(--cb-text-secondary) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
"""

# Apply the CSS based on current theme
st.markdown(apply_theme_css(st.session_state.theme), unsafe_allow_html=True)

# Initialize robust data manager
@st.cache_resource
def get_data_manager():
    return RobustDataFetcher()

data_manager = get_data_manager()

# Initialize session state for data caching
if 'cached_data' not in st.session_state:
    st.session_state.cached_data = {}
if 'cached_returns' not in st.session_state:
    st.session_state.cached_returns = {}
if 'data_cache_key' not in st.session_state:
    st.session_state.data_cache_key = None
if 'cache_timestamp' not in st.session_state:
    st.session_state.cache_timestamp = None

# Title and description
st.title("Crypto Portfolio Optimizer & Backtester")
st.markdown("**Live Demonstration** - Advanced portfolio optimization and backtesting for cryptocurrencies")

# Sidebar
st.sidebar.title("Portfolio Configuration")

# Theme settings removed - using light mode only

# Mode selection
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Portfolio Analysis", "Portfolio Optimization", "Sample Backtest", "Market Insights"]
)

# Common parameters
st.sidebar.subheader("Time Period")
end_date = st.sidebar.date_input("End Date", datetime.now() - timedelta(days=1))
start_date = st.sidebar.date_input("Start Date", end_date - timedelta(days=365))

crypto_symbols = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 
    'SOL-USD', 'MATIC-USD', 'DOT-USD', 'AVAX-USD', 'LINK-USD',
    'UNI-USD', 'LTC-USD', 'BCH-USD', 'ATOM-USD', 'ALGO-USD'
]
selected_symbols = st.sidebar.multiselect(
    "Select Cryptocurrencies",
    crypto_symbols,
    default=['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD']
)

if len(selected_symbols) < 2:
    st.error("Please select at least 2 cryptocurrencies")
    st.stop()

# Smart data fetching function (same as original)
def get_cached_data(symbols, start_date, end_date):
    """
    Intelligent data caching - only fetch when symbols or dates change
    """
    # Create cache key from parameters
    cache_key = f"{sorted(symbols)}_{start_date}_{end_date}"
    
    # Check if we already have this data
    if (st.session_state.data_cache_key == cache_key and 
        cache_key in st.session_state.cached_data and
        cache_key in st.session_state.cached_returns):
        
        st.info("Using cached data (no API calls needed)")
        return (st.session_state.cached_data[cache_key], 
                st.session_state.cached_returns[cache_key])
    
    # Need to fetch new data - show discrete loading indicator
    loading_placeholder = st.empty()
    loading_placeholder.markdown("""
    <div style='position: fixed; top: 70px; right: 20px; background: white; padding: 8px 12px; 
                border-radius: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); z-index: 999;
                border: 1px solid #e6e8ea;'>
        <span style='font-size: 12px; color: #5b616e;'>Loading data...</span>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        price_data = data_manager.get_real_data(symbols, start_date, end_date)
        
        if price_data.empty:
            loading_placeholder.empty()
            st.error("Unable to fetch data. Please try again or select different symbols.")
            st.stop()
        
        returns_data = data_manager.calculate_returns(price_data)
        
        # Cache the results
        st.session_state.cached_data[cache_key] = price_data
        st.session_state.cached_returns[cache_key] = returns_data
        st.session_state.data_cache_key = cache_key
        st.session_state.cache_timestamp = datetime.now()
        
        # Show brief success message
        loading_placeholder.markdown("""
        <div style='position: fixed; top: 70px; right: 20px; background: #f0f9ff; padding: 8px 12px; 
                    border-radius: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); z-index: 999;
                    border: 1px solid #00d395;'>
            <span style='font-size: 12px; color: #00d395;'>Data loaded</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Clear the success message after a short delay
        import time
        time.sleep(1.5)
        loading_placeholder.empty()
        
        return price_data, returns_data
        
    except Exception as e:
        loading_placeholder.empty()
        st.error(f"Error fetching data: {str(e)}")
        st.stop()

# Add cache info and refresh button in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Data Management")

# Show cache status
if st.session_state.data_cache_key:
    cached_symbols = st.session_state.data_cache_key.split('_')[0]
    cached_symbols = cached_symbols.replace('[', '').replace(']', '').replace("'", "")
    st.sidebar.success(f"Data cached for: {cached_symbols}")
    
    # Show data freshness
    if st.session_state.cache_timestamp:
        cache_time = st.session_state.cache_timestamp.strftime("%H:%M:%S")
        st.sidebar.info(f"Last updated: {cache_time}")
else:
    st.sidebar.info("No data cached yet")

if st.sidebar.button("Refresh Data", help="Force fetch fresh data from APIs"):
    # Clear cache
    st.session_state.cached_data = {}
    st.session_state.cached_returns = {}
    st.session_state.data_cache_key = None
    st.session_state.cache_timestamp = None
    st.rerun()

# Performance tip
st.sidebar.markdown("**Tip**: Data is automatically cached. Only fetches when symbols/dates change!")

# Main content area
if mode == "Portfolio Optimization":
    st.header("Advanced Portfolio Optimization")
    
    # Get cached data
    price_data, returns = get_cached_data(
        selected_symbols, 
        start_date.strftime('%Y-%m-%d'), 
        end_date.strftime('%Y-%m-%d')
    )
    
    # Initialize portfolio optimizer
    optimizer = PortfolioOptimizer(returns)
    
    # User preferences sidebar
    st.sidebar.subheader("Optimization Preferences")
    
    # Strategy selection
    strategy = st.sidebar.selectbox(
        "Portfolio Strategy",
        [
            "Mean-Variance Optimization",
            "Risk Parity", 
            "Minimum Variance",
            "Maximum Sharpe Ratio",
            "Momentum Portfolio",
            "Equal Weight"
        ]
    )
    
    # Risk tolerance (for Mean-Variance)
    if strategy == "Mean-Variance Optimization":
        risk_tolerance = st.sidebar.slider(
            "Risk Tolerance", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.1,
            help="0 = Risk Averse (focus on minimizing risk), 1 = Risk Seeking (focus on maximizing return)"
        )
        
        target_return = st.sidebar.number_input(
            "Target Annual Return (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=None,
            help="Leave empty for automatic risk-return optimization"
        )
        if target_return is not None:
            target_return = target_return / 100
    
    # Maximum weight constraint
    max_weight = st.sidebar.slider(
        "Maximum Weight per Asset (%)", 
        min_value=10, 
        max_value=100, 
        value=40, 
        step=5
    ) / 100
    
    # Risk-free rate
    risk_free_rate = st.sidebar.number_input(
        "Risk-Free Rate (%)", 
        min_value=0.0, 
        max_value=10.0, 
        value=2.0, 
        step=0.1
    ) / 100
    
    optimizer.risk_free_rate = risk_free_rate
    
    # Run optimization
    if st.button("Optimize Portfolio", type="primary"):
        with st.spinner(f'Optimizing portfolio using {strategy}...'):
            
            # Call appropriate optimization method
            if strategy == "Mean-Variance Optimization":
                if 'target_return' in locals() and target_return is not None:
                    result = optimizer.mean_variance_optimization(
                        target_return=target_return, 
                        max_weight=max_weight
                    )
                else:
                    result = optimizer.mean_variance_optimization(
                        risk_tolerance=risk_tolerance, 
                        max_weight=max_weight
                    )
            elif strategy == "Risk Parity":
                result = optimizer.risk_parity_optimization(max_weight=max_weight)
            elif strategy == "Minimum Variance":
                result = optimizer.minimum_variance_optimization(max_weight=max_weight)
            elif strategy == "Maximum Sharpe Ratio":
                result = optimizer.maximum_sharpe_optimization(max_weight=max_weight)
            elif strategy == "Momentum Portfolio":
                result = optimizer.momentum_portfolio(max_weight=max_weight)
            else:  # Equal Weight
                result = optimizer.equal_weight_portfolio(max_weight=max_weight)
        
        # Display results
        if result['success']:
            st.success(f"Optimization completed using {result['strategy']}")
            
            # Portfolio metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Expected Return", f"{result['expected_return']:.1%}")
            with col2:
                st.metric("Volatility", f"{result['volatility']:.1%}")
            with col3:
                st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
            with col4:
                max_weight_actual = max(result['weights'].values())
                st.metric("Max Weight", f"{max_weight_actual:.1%}")
            
            # Portfolio allocation
            st.subheader("Optimized Portfolio Allocation")
            
            # Create allocation DataFrame
            allocation_data = []
            for asset, weight in result['weights'].items():
                if weight > 0.001:  # Only show meaningful allocations
                    allocation_data.append({
                        'Asset': asset.replace('-USD', ''),
                        'Weight': weight,
                        'Weight (%)': f"{weight:.1%}",
                        'Risk Contribution': result['risk_contributions'][asset],
                        'Risk Contrib (%)': f"{result['risk_contributions'][asset]:.1%}"
                    })
            
            allocation_df = pd.DataFrame(allocation_data)
            allocation_df = allocation_df.sort_values('Weight', ascending=False)
            
            # Display allocation table
            st.dataframe(allocation_df[['Asset', 'Weight (%)', 'Risk Contrib (%)']], 
                        use_container_width=True)
            
            # Portfolio allocation pie chart
            col1, col2 = st.columns(2)
            
            with col1:
                # Professional crypto color palette
                crypto_colors = [
                    '#F7931A',  # Bitcoin Orange
                    '#627EEA',  # Ethereum Blue  
                    '#F3BA2F',  # Binance Yellow
                    '#00AAE4',  # XRP Blue
                    '#0033AD',  # Cardano Blue
                    '#9945FF',  # Solana Purple
                    '#8247E5',  # Polygon Purple
                    '#E6007A',  # Polkadot Pink
                    '#E84142',  # Avalanche Red
                    '#375BD2',  # Chainlink Blue
                    '#FF007A',  # Uniswap Pink
                    '#BFBBBB',  # Litecoin Silver
                    '#8DC351',  # Bitcoin Cash Green
                    '#2E3148',  # Cosmos Dark
                    '#000000'   # Algorand Black
                ]
                
                fig_pie = px.pie(
                    allocation_df,
                    values='Weight',
                    names='Asset',
                    title="Portfolio Allocation",
                    color_discrete_sequence=crypto_colors
                )
                fig_pie.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    textfont_size=11,
                    marker=dict(line=dict(color='white', width=2))
                )
                fig_pie.update_layout(
                    font=dict(size=12),
                    template='plotly_white',
                    showlegend=True,
                    legend=dict(orientation="v", yanchor="middle", y=0.5)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                fig_risk = px.pie(
                    allocation_df,
                    values='Risk Contribution',
                    names='Asset',
                    title="Risk Contribution",
                    color_discrete_sequence=crypto_colors
                )
                fig_risk.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    textfont_size=11,
                    marker=dict(line=dict(color='white', width=2))
                )
                fig_risk.update_layout(
                    font=dict(size=12),
                    template='plotly_white',
                    showlegend=True,
                    legend=dict(orientation="v", yanchor="middle", y=0.5)
                )
                st.plotly_chart(fig_risk, use_container_width=True)
        
        else:
            st.error(f"Optimization failed: {result.get('error', 'Unknown error')}")
            st.info("Try adjusting your parameters or selecting different assets")
    
    else:
        st.info("Configure your preferences and click 'Optimize Portfolio' to get started")

elif mode == "Sample Backtest":
    st.header("Portfolio Backtesting Analysis")
    
    # Get cached data
    price_data, returns = get_cached_data(
        selected_symbols, 
        start_date.strftime('%Y-%m-%d'), 
        end_date.strftime('%Y-%m-%d')
    )
    
    # Manual weight input
    st.subheader("Portfolio Allocation")
    
    weights = {}
    cols = st.columns(len(selected_symbols))
    
    for i, symbol in enumerate(selected_symbols):
        with cols[i]:
            symbol_name = symbol.replace('-USD', '')
            weights[symbol] = st.number_input(
                f"{symbol_name}",
                min_value=0.0,
                max_value=1.0,
                value=1.0/len(selected_symbols),
                step=0.05,
                format="%.2f"
            )
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
    else:
        normalized_weights = {k: 1.0/len(selected_symbols) for k in selected_symbols}
    
    st.info(f"Total weight: {total_weight:.1%} (automatically normalized to 100%)")
    
    # Portfolio parameters
    col1, col2 = st.columns(2)
    with col1:
        initial_capital = st.number_input("Initial Capital ($)", value=100000, min_value=1000, step=10000)
    with col2:
        rebalance_freq = st.selectbox("Rebalancing", ["Monthly", "Quarterly", "Semi-Annual"])
    
    # Enhanced portfolio simulation
    portfolio_values = []
    btc_values = []
    eth_values = []
    equal_weight_values = []
    
    # Equal weight benchmark
    equal_weights = {symbol: 1.0/len(selected_symbols) for symbol in selected_symbols}
    
    for i, date in enumerate(price_data.index):
        if i == 0:
            portfolio_value = initial_capital
            btc_value = initial_capital
            eth_value = initial_capital
            equal_weight_value = initial_capital
        else:
            # Portfolio return
            daily_returns = price_data.iloc[i] / price_data.iloc[i-1] - 1
            
            # Custom portfolio
            portfolio_return = sum(normalized_weights[symbol] * daily_returns[symbol] for symbol in selected_symbols)
            portfolio_value = portfolio_values[-1] * (1 + portfolio_return)
            
            # BTC benchmark
            if 'BTC-USD' in price_data.columns:
                btc_return = daily_returns['BTC-USD']
                btc_value = btc_values[-1] * (1 + btc_return)
            else:
                btc_value = btc_values[-1]
            
            # ETH benchmark
            if 'ETH-USD' in price_data.columns:
                eth_return = daily_returns['ETH-USD']
                eth_value = eth_values[-1] * (1 + eth_return)
            else:
                eth_value = eth_values[-1]
            
            # Equal weight benchmark
            equal_weight_return = sum(equal_weights[symbol] * daily_returns[symbol] for symbol in selected_symbols)
            equal_weight_value = equal_weight_values[-1] * (1 + equal_weight_return)
        
        portfolio_values.append(portfolio_value)
        btc_values.append(btc_value)
        eth_values.append(eth_value)
        equal_weight_values.append(equal_weight_value)
    
    # Performance metrics
    portfolio_total_return = (portfolio_values[-1] / portfolio_values[0] - 1)
    btc_total_return = (btc_values[-1] / btc_values[0] - 1)
    eth_total_return = (eth_values[-1] / eth_values[0] - 1)
    equal_weight_total_return = (equal_weight_values[-1] / equal_weight_values[0] - 1)
    
    # Calculate additional metrics
    portfolio_returns_series = pd.Series(portfolio_values).pct_change().dropna()
    portfolio_volatility = portfolio_returns_series.std() * np.sqrt(252)
    portfolio_sharpe = (portfolio_total_return * 252 - 0.02) / (portfolio_volatility * np.sqrt(252)) if portfolio_volatility > 0 else 0
    
    # Max drawdown calculation
    rolling_max = pd.Series(portfolio_values).expanding().max()
    drawdown = (pd.Series(portfolio_values) - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Display results
    st.subheader("Backtest Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Portfolio Return", 
            f"{portfolio_total_return:.1%}",
            delta=f"{portfolio_total_return - btc_total_return:.1%}"
        )
    
    with col2:
        st.metric("Final Value", f"${portfolio_values[-1]:,.0f}")
    
    with col3:
        days_total = len(portfolio_values)
        annualized_return = (portfolio_values[-1] / portfolio_values[0]) ** (365 / days_total) - 1
        st.metric("Annualized Return", f"{annualized_return:.1%}")
    
    with col4:
        st.metric("Max Drawdown", f"{max_drawdown:.1%}")
    
    # Additional metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Volatility", f"{portfolio_volatility:.1%}")
    with col2:
        st.metric("Sharpe Ratio", f"{portfolio_sharpe:.2f}")
    with col3:
        st.metric("BTC Return", f"{btc_total_return:.1%}")
    with col4:
        st.metric("ETH Return", f"{eth_total_return:.1%}")
    
    # Performance comparison chart
    st.subheader("Performance Comparison")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=portfolio_values,
        name='Your Portfolio',
        line=dict(color='#5ac53a', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=btc_values,
        name='BTC Hold',
        line=dict(color='#f7931a', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=eth_values,
        name='ETH Hold',
        line=dict(color='#627eea', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=equal_weight_values,
        name='Equal Weight',
        line=dict(color='#b3b3b3', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Portfolio Performance vs Benchmarks",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        height=500,
        template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True, hide_index=True)
    
    # Drawdown chart
    st.subheader("Drawdown Analysis")
    
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=price_data.index,
        y=drawdown * 100,
        name='Drawdown',
        fill='tonexty',
        line=dict(color='#e85555'),
        fillcolor='rgba(232, 85, 85, 0.3)'
    ))
    
    fig_dd.update_layout(
        title="Portfolio Drawdown Over Time",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=350,
        template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
    )
    
    st.plotly_chart(fig_dd, use_container_width=True, hide_index=True)
    
    # Portfolio allocation pie chart
    st.subheader("Portfolio Allocation")
    
    allocation_data = pd.DataFrame({
        'Asset': [symbol.replace('-USD', '') for symbol in normalized_weights.keys()],
        'Weight': list(normalized_weights.values())
    })
    
    # Professional crypto color palette
    crypto_colors = [
        '#F7931A',  # Bitcoin Orange
        '#627EEA',  # Ethereum Blue  
        '#F3BA2F',  # Binance Yellow
        '#00AAE4',  # XRP Blue
        '#0033AD',  # Cardano Blue
        '#9945FF',  # Solana Purple
        '#8247E5',  # Polygon Purple
        '#E6007A',  # Polkadot Pink
        '#E84142',  # Avalanche Red
        '#375BD2',  # Chainlink Blue
        '#FF007A',  # Uniswap Pink
        '#BFBBBB',  # Litecoin Silver
        '#8DC351',  # Bitcoin Cash Green
        '#2E3148',  # Cosmos Dark
        '#000000'   # Algorand Black
    ]
    
    fig_pie = px.pie(
        allocation_data,
        values='Weight',
        names='Asset',
        title="Current Portfolio Allocation",
        color_discrete_sequence=crypto_colors
    )
    fig_pie.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        textfont_size=11,
        marker=dict(line=dict(color='white', width=2))
    )
    fig_pie.update_layout(
        font=dict(size=12),
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5)
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Performance summary table
    st.subheader("Performance Summary")
    
    summary_data = [
        {
            'Strategy': 'Your Portfolio',
            'Total Return': f"{portfolio_total_return:.1%}",
            'Annualized Return': f"{annualized_return:.1%}",
            'Volatility': f"{portfolio_volatility:.1%}",
            'Sharpe Ratio': f"{portfolio_sharpe:.2f}",
            'Max Drawdown': f"{max_drawdown:.1%}"
        },
        {
            'Strategy': 'BTC Hold',
            'Total Return': f"{btc_total_return:.1%}",
            'Annualized Return': f"{((btc_values[-1] / btc_values[0]) ** (365 / days_total) - 1):.1%}",
            'Volatility': 'N/A',
            'Sharpe Ratio': 'N/A',
            'Max Drawdown': 'N/A'
        },
        {
            'Strategy': 'ETH Hold',
            'Total Return': f"{eth_total_return:.1%}",
            'Annualized Return': f"{((eth_values[-1] / eth_values[0]) ** (365 / days_total) - 1):.1%}",
            'Volatility': 'N/A',
            'Sharpe Ratio': 'N/A',
            'Max Drawdown': 'N/A'
        },
        {
            'Strategy': 'Equal Weight',
            'Total Return': f"{equal_weight_total_return:.1%}",
            'Annualized Return': f"{((equal_weight_values[-1] / equal_weight_values[0]) ** (365 / days_total) - 1):.1%}",
            'Volatility': 'N/A',
            'Sharpe Ratio': 'N/A',
            'Max Drawdown': 'N/A'
        }
    ]
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

elif mode == "Market Insights":
    st.header("Market Analysis & Insights")
    
    # Get cached data
    price_data, returns = get_cached_data(
        selected_symbols, 
        start_date.strftime('%Y-%m-%d'), 
        end_date.strftime('%Y-%m-%d')
    )
    
    # Market overview
    st.subheader("Market Overview")
    
    # Calculate market metrics
    market_data = []
    for symbol in selected_symbols:
        symbol_name = symbol.replace('-USD', '')
        current_price = price_data[symbol].iloc[-1]
        start_price = price_data[symbol].iloc[0]
        total_return = (current_price / start_price - 1)
        volatility = returns[symbol].std() * np.sqrt(252)
        
        # Risk level
        if volatility > 0.8:
            risk_level = "Very High"
        elif volatility > 0.6:
            risk_level = "High"
        elif volatility > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # 7-day and 30-day returns
        week_return = (current_price / price_data[symbol].iloc[-8] - 1) if len(price_data) >= 8 else 0
        month_return = (current_price / price_data[symbol].iloc[-31] - 1) if len(price_data) >= 31 else 0
        
        market_data.append({
            'Asset': symbol_name,
            'Current Price': f"${current_price:,.2f}",
            'Total Return': f"{total_return:+.1%}",
            '7D Return': f"{week_return:+.1%}",
            '30D Return': f"{month_return:+.1%}",
            'Volatility': f"{volatility:.1%}",
            'Risk Level': risk_level,
            '30-Day Avg': f"${price_data[symbol].tail(30).mean():,.2f}"
        })
    
    market_df = pd.DataFrame(market_data)
    st.dataframe(market_df, use_container_width=True, hide_index=True)
    
    # Risk-Return scatter plot
    st.subheader("Risk-Return Analysis")
    
    risk_return_data = []
    for symbol in selected_symbols:
        symbol_name = symbol.replace('-USD', '')
        annual_return = returns[symbol].mean() * 252
        annual_vol = returns[symbol].std() * np.sqrt(252)
        market_cap_rank = list(selected_symbols).index(symbol) + 1  # Simple ranking
        
        risk_return_data.append({
            'Asset': symbol_name,
            'Return': annual_return,
            'Volatility': annual_vol,
            'Size': market_cap_rank * 10  # For bubble size
        })
    
    rr_df = pd.DataFrame(risk_return_data)
    
    fig_scatter = px.scatter(
        rr_df,
        x='Volatility',
        y='Return',
        size='Size',
        text='Asset',
        title="Risk vs Return Profile",
        labels={'Volatility': 'Annual Volatility', 'Return': 'Annual Return'},
        template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
    )
    
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True, hide_index=True)
    
    # Enhanced Correlation Analysis
    st.subheader("Asset Correlation Matrix")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        corr_matrix = returns.corr()
        corr_matrix.columns = [col.replace('-USD', '') for col in corr_matrix.columns]
        corr_matrix.index = [idx.replace('-USD', '') for idx in corr_matrix.index]

        # Enhanced correlation heatmap with better styling
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            title="",
            color_continuous_scale="RdYlBu_r",
            zmin=-1, zmax=1,
            template='plotly_white'
        )
        fig_corr.update_layout(
            height=400,
            font=dict(size=11),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=60, t=20, b=60)
        )
        fig_corr.update_coloraxes(
            colorbar=dict(
                title="Correlation",
                tickmode="linear",
                tick0=-1,
                dtick=0.5
            )
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        st.markdown("**Correlation Insights**")
        
        # Find highest and lowest correlations
        corr_values = corr_matrix.values
        np.fill_diagonal(corr_values, np.nan)  # Remove self-correlations
        
        # Get asset pairs with highest/lowest correlations
        max_idx = np.nanargmax(corr_values)
        min_idx = np.nanargmin(corr_values)
        max_i, max_j = np.unravel_index(max_idx, corr_values.shape)
        min_i, min_j = np.unravel_index(min_idx, corr_values.shape)
        
        highest_corr = corr_values[max_i, max_j]
        lowest_corr = corr_values[min_i, min_j]
        
        st.metric(
            "Highest Correlation",
            f"{corr_matrix.index[max_i]} - {corr_matrix.columns[max_j]}",
            f"{highest_corr:.2f}"
        )
        
        st.metric(
            "Lowest Correlation", 
            f"{corr_matrix.index[min_i]} - {corr_matrix.columns[min_j]}",
            f"{lowest_corr:.2f}"
        )
        
        # Average correlation
        avg_corr = np.nanmean(corr_values)
        st.metric("Average Correlation", f"{avg_corr:.2f}")
        
        # Diversification insight
        if avg_corr > 0.7:
            st.warning("High correlation - Limited diversification")
        elif avg_corr > 0.3:
            st.info("Moderate correlation - Some diversification benefit")
        else:
            st.success("Low correlation - Good diversification potential")
    
    # Rolling metrics
    st.subheader("Rolling Performance Metrics")
    
    # Calculate 30-day rolling metrics
    rolling_window = 30
    rolling_returns = returns.rolling(rolling_window).mean() * 252
    rolling_vol = returns.rolling(rolling_window).std() * np.sqrt(252)
    rolling_sharpe = (rolling_returns - 0.02) / rolling_vol
    
    # Select asset for detailed analysis
    selected_asset = st.selectbox("Select asset for detailed analysis:", selected_symbols)
    asset_name = selected_asset.replace('-USD', '')
    
    if selected_asset in rolling_sharpe.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_rolling = go.Figure()
            
            fig_rolling.add_trace(go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe[selected_asset],
                name=f'{asset_name} Sharpe Ratio',
                line=dict(color='#5ac53a' if st.session_state.theme == 'dark' else '#1976d2', width=2)
            ))
            
            fig_rolling.update_layout(
                title=f"{asset_name} - 30-Day Rolling Sharpe Ratio",
                xaxis_title="Date",
                yaxis_title="Sharpe Ratio",
                height=350,
                template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
            )
            
            st.plotly_chart(fig_rolling, use_container_width=True, hide_index=True)
        
        with col2:
            fig_vol = go.Figure()
            
            fig_vol.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol[selected_asset] * 100,
                name=f'{asset_name} Volatility',
                line=dict(color='#e85555' if st.session_state.theme == 'dark' else '#d32f2f', width=2)
            ))
            
            fig_vol.update_layout(
                title=f"{asset_name} - 30-Day Rolling Volatility",
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                height=350,
                template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
            )
            
            st.plotly_chart(fig_vol, use_container_width=True, hide_index=True)
    
    # Market summary statistics
    st.subheader("Market Summary Statistics")
    
    summary_stats = []
    for symbol in selected_symbols:
        symbol_name = symbol.replace('-USD', '')
        symbol_returns = returns[symbol]
        
        summary_stats.append({
            'Asset': symbol_name,
            'Mean Daily Return': f"{symbol_returns.mean():.3%}",
            'Std Daily Return': f"{symbol_returns.std():.3%}",
            'Skewness': f"{symbol_returns.skew():.2f}",
            'Kurtosis': f"{symbol_returns.kurtosis():.2f}",
            'Min Daily Return': f"{symbol_returns.min():.2%}",
            'Max Daily Return': f"{symbol_returns.max():.2%}"
        })
    
    summary_df = pd.DataFrame(summary_stats)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

else:  # Portfolio Analysis
    st.header("Portfolio Analysis & Overview")
    
    # Get cached data
    price_data, returns = get_cached_data(
        selected_symbols, 
        start_date.strftime('%Y-%m-%d'), 
        end_date.strftime('%Y-%m-%d')
    )
    
    # Manual portfolio weight input
    st.subheader("Define Your Portfolio Weights")
    
    # Create columns for weight input
    n_cols = min(len(selected_symbols), 3)  # Max 3 columns for better layout
    cols = st.columns(n_cols)
    
    portfolio_weights = {}
    total_weight = 0
    
    for i, symbol in enumerate(selected_symbols):
        col_idx = i % n_cols
        with cols[col_idx]:
            weight = st.number_input(
                f"{symbol.replace('-USD', '')} Weight", 
                min_value=0.0, 
                max_value=1.0, 
                value=1.0/len(selected_symbols),  # Default equal weight
                step=0.05,
                key=f"portfolio_weight_{symbol}"
            )
            portfolio_weights[symbol] = weight
            total_weight += weight
    
    # Normalize weights to sum to 100%
    normalized_weights = {}
    if total_weight > 0:
        for symbol in selected_symbols:
            normalized_weights[symbol] = portfolio_weights[symbol] / total_weight
    else:
        # Fallback to equal weights
        for symbol in selected_symbols:
            normalized_weights[symbol] = 1.0 / len(selected_symbols)
    
    # Display weight summary
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Total Weight:** {total_weight:.1%}")
    with col2:
        if abs(total_weight - 1.0) > 0.01:
            st.info(f"Weights normalized to 100% (scaling factor: {1/total_weight:.2f})")
        else:
            st.success("Weights sum to 100%")
    
    # Calculate portfolio metrics based on user weights
    portfolio_returns = []
    portfolio_values = [10000]  # Start with $10,000
    
    for i in range(1, len(price_data)):
        daily_returns = {}
        for symbol in selected_symbols:
            daily_returns[symbol] = price_data[symbol].iloc[i] / price_data[symbol].iloc[i-1] - 1
        
        # Calculate weighted portfolio return
        portfolio_return = sum(normalized_weights[symbol] * daily_returns[symbol] for symbol in selected_symbols)
        portfolio_returns.append(portfolio_return)
        portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
    
    # Calculate portfolio metrics
    portfolio_returns = np.array(portfolio_returns)
    portfolio_total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)
    portfolio_annualized_return = (portfolio_values[-1] / portfolio_values[0]) ** (252 / len(portfolio_returns)) - 1
    portfolio_sharpe = (portfolio_annualized_return - 0.02) / portfolio_volatility if portfolio_volatility > 0 else 0
    
    # Maximum drawdown calculation
    portfolio_values_series = pd.Series(portfolio_values)
    running_max = portfolio_values_series.expanding().max()
    drawdown = (portfolio_values_series - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Display portfolio metrics
    st.subheader("Your Portfolio Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{portfolio_total_return:.1%}")
    
    with col2:
        st.metric("Annualized Volatility", f"{portfolio_volatility:.1%}")
    
    with col3:
        st.metric("Sharpe Ratio", f"{portfolio_sharpe:.2f}")
    
    with col4:
        st.metric("Max Drawdown", f"{max_drawdown:.1%}")
    
    # Price charts
    st.subheader("Price Performance")
    
    # Normalize prices to 100 for comparison
    normalized_prices = (price_data / price_data.iloc[0] * 100)
    
    fig = go.Figure()
    colors = ['#5ac53a', '#e85555', '#00ff88', '#f7931a', '#627eea']
    
    for i, symbol in enumerate(selected_symbols):
        fig.add_trace(go.Scatter(
            x=normalized_prices.index,
            y=normalized_prices[symbol],
            name=symbol.replace('-USD', ''),
            line=dict(color=colors[i % len(colors)], width=2),
            mode='lines'
        ))
    
    fig.update_layout(
        title="Normalized Price Performance (Base = 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        hovermode='x unified',
        height=400,
        template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True, hide_index=True)
    
    # Returns distribution and correlation
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot of returns
        returns_melted = returns.melt(var_name='Asset', value_name='Daily Return')
        returns_melted['Asset'] = returns_melted['Asset'].str.replace('-USD', '')
        
        fig_box = px.box(
            returns_melted, 
            x='Asset', 
            y='Daily Return',
            title="Daily Return Distributions",
            template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
        )
        fig_box.update_layout(height=350)
        st.plotly_chart(fig_box, use_container_width=True, hide_index=True)
    
    with col2:
        # Enhanced correlation heatmap
        corr_matrix = returns.corr()
        corr_matrix.columns = [col.replace('-USD', '') for col in corr_matrix.columns]
        corr_matrix.index = [idx.replace('-USD', '') for idx in corr_matrix.index]
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            title="Asset Correlations",
            color_continuous_scale="RdYlBu_r",
            zmin=-1, zmax=1,
            template='plotly_white'
        )
        fig_corr.update_layout(
            height=350,
            font=dict(size=10),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=40, b=50)
        )
        fig_corr.update_coloraxes(
            colorbar=dict(
                title="Correlation",
                tickmode="linear",
                tick0=-1,
                dtick=0.5
            )
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Statistics table
    st.subheader("Statistical Summary")
    
    stats_data = []
    for symbol in selected_symbols:
        symbol_returns = returns[symbol]
        symbol_name = symbol.replace('-USD', '')
        
        stats_data.append({
            'Asset': symbol_name,
            'Mean Return (Annual)': f"{symbol_returns.mean() * 252:.1%}",
            'Volatility (Annual)': f"{symbol_returns.std() * np.sqrt(252):.1%}",
            'Sharpe Ratio': f"{(symbol_returns.mean() * 252 - 0.02) / (symbol_returns.std() * np.sqrt(252)):.2f}",
            'Max Daily Return': f"{symbol_returns.max():.2%}",
            'Min Daily Return': f"{symbol_returns.min():.2%}",
            'Current Price': f"${price_data[symbol].iloc[-1]:,.2f}"
        })
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center'>
    <p><strong>Crypto Portfolio Optimizer</strong> - Advanced portfolio management for digital assets</p>
    <p><em>This uses real market data from multiple sources. Not financial advice.</em></p>
    <p>Built with Streamlit, Plotly, and Python</p>
</div>
""", unsafe_allow_html=True)