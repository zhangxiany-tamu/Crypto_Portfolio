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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import requests
import json
warnings.filterwarnings('ignore')

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
        font-size: 16px;
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
        width: 100%;
        box-sizing: border-box;
    }
    
    /* Stable sidebar styling */
    .stSidebar {
        background: var(--cb-card-bg) !important;
        border-right: 1px solid var(--cb-border) !important;
        box-shadow: var(--cb-shadow);
        width: 300px !important;
        min-width: 300px !important;
        max-width: 300px !important;
    }
    
    .stSidebar .stSidebarContent {
        background: var(--cb-card-bg) !important;
        width: 100% !important;
    }
    
    .stSidebar label {
        color: var(--cb-text-primary) !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: var(--cb-text-primary) !important;
        font-weight: 600 !important;
    }
    
    .stSidebar p {
        color: var(--cb-text-secondary) !important;
        font-size: 0.875rem !important;
    }
    
    /* Prevent content overflow */
    .main {
        overflow-x: auto;
        width: 100%;
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
    
    /* Enhanced Font Sizes for Better Readability */
    
    /* Main content text */
    .stMarkdown, .stText, p {
        font-size: 16px !important;
        line-height: 1.5 !important;
    }
    
    /* Sidebar labels and text */
    .css-1d391kg label, .stSidebar label {
        font-size: 15px !important;
    }
    
    .css-1d391kg p, .stSidebar p {
        font-size: 14px !important;
    }
    
    /* Button text */
    .stButton > button {
        font-size: 16px !important;
        padding: 0.75rem 1.5rem !important;
    }
    
    /* Input field text */
    .stNumberInput input, .stTextInput input, .stSelectbox select {
        font-size: 15px !important;
    }
    
    /* Metric values and labels */
    .stMetric [data-testid="metric-value"] {
        font-size: 1.75rem !important;
    }
    
    .stMetric [data-testid="metric-label"] {
        font-size: 14px !important;
    }
    
    /* Headers */
    h1 {
        font-size: 2.5rem !important;
    }
    
    h2 {
        font-size: 2rem !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
    }
    
    /* Subheaders */
    .stSubheader {
        font-size: 1.25rem !important;
    }
    
    /* Table text */
    .stDataFrame table {
        font-size: 14px !important;
    }
    
    .stDataFrame th {
        font-size: 13px !important;
    }
    
    /* Info, warning, success, error messages */
    .stAlert {
        font-size: 15px !important;
    }
    
    /* Multiselect items */
    .stMultiSelect span {
        font-size: 14px !important;
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
    ["Portfolio Analysis", "Portfolio Optimization", "Sample Backtest", "Market Insights", "ML Predictions", "AI Investment Advisor"]
)

# Common parameters
st.sidebar.subheader("Time Period")
end_date = st.sidebar.date_input("End Date", datetime.now() - timedelta(days=1))
start_date = st.sidebar.date_input("Start Date", end_date - timedelta(days=365))

crypto_symbols = [
    # Top 10 by market cap
    'BTC-USD', 'ETH-USD', 'XRP-USD', 'BNB-USD', 'SOL-USD',
    'DOGE-USD', 'ADA-USD', 'TRX-USD', 'SHIB-USD', 'AVAX-USD',
    
    # Major DeFi and Layer 1/2
    'LINK-USD', 'DOT-USD', 'UNI-USD', 'AAVE-USD', 'MATIC-USD',
    'NEAR-USD', 'ICP-USD', 'APT-USD', 'SUI-USD', 'ATOM-USD',
    
    # Established cryptocurrencies
    'LTC-USD', 'BCH-USD', 'XLM-USD', 'XMR-USD', 'ETC-USD',
    'HBAR-USD', 'TON-USD', 'ALGO-USD', 'VET-USD', 'FTM-USD'
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
            
            # Set Asset as index and display table without separate Asset column
            allocation_display = allocation_df.set_index('Asset')[['Weight (%)', 'Risk Contrib (%)']]
            
            # Display allocation table
            st.dataframe(allocation_display, use_container_width=True)
            
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
    
    # Generate rebalancing dates based on frequency
    def get_rebalance_dates(date_index, frequency):
        rebalance_dates = []
        if frequency == "Monthly":
            # First trading day of each month
            for year in range(date_index[0].year, date_index[-1].year + 1):
                for month in range(1, 13):
                    month_start = pd.Timestamp(year, month, 1)
                    month_dates = [d for d in date_index if d >= month_start and d.month == month and d.year == year]
                    if month_dates:
                        rebalance_dates.append(min(month_dates))
        elif frequency == "Quarterly":
            # First trading day of Jan, Apr, Jul, Oct
            for year in range(date_index[0].year, date_index[-1].year + 1):
                for month in [1, 4, 7, 10]:
                    quarter_start = pd.Timestamp(year, month, 1)
                    quarter_dates = [d for d in date_index if d >= quarter_start and d.month == month and d.year == year]
                    if quarter_dates:
                        rebalance_dates.append(min(quarter_dates))
        elif frequency == "Semi-Annual":
            # First trading day of Jan and Jul
            for year in range(date_index[0].year, date_index[-1].year + 1):
                for month in [1, 7]:
                    semi_start = pd.Timestamp(year, month, 1)
                    semi_dates = [d for d in date_index if d >= semi_start and d.month == month and d.year == year]
                    if semi_dates:
                        rebalance_dates.append(min(semi_dates))
        return set(rebalance_dates)
    
    rebalance_dates = get_rebalance_dates(price_data.index, rebalance_freq)
    trading_fee = 0.001  # 0.1% trading fee per transaction
    
    # Enhanced portfolio simulation with rebalancing
    portfolio_values = []
    btc_values = []
    eth_values = []
    equal_weight_values = []
    portfolio_shares = {symbol: 0 for symbol in selected_symbols}  # Track actual shares
    rebalance_costs = []
    
    # Equal weight benchmark
    equal_weights = {symbol: 1.0/len(selected_symbols) for symbol in selected_symbols}
    
    for i, date in enumerate(price_data.index):
        if i == 0:
            # Initial investment
            portfolio_value = initial_capital
            btc_value = initial_capital
            eth_value = initial_capital
            equal_weight_value = initial_capital
            
            # Initial allocation with trading costs
            initial_cost = initial_capital * trading_fee
            remaining_capital = initial_capital - initial_cost
            for symbol in selected_symbols:
                allocation = remaining_capital * normalized_weights[symbol]
                portfolio_shares[symbol] = allocation / price_data[symbol].iloc[i]
            
            rebalance_costs.append(initial_cost)
        else:
            # Calculate current portfolio value from shares
            current_portfolio_value = sum(portfolio_shares[symbol] * price_data[symbol].iloc[i] for symbol in selected_symbols)
            
            # Check if rebalancing is needed
            if date in rebalance_dates:
                # Rebalancing: sell all positions and buy according to target weights
                turnover = 0.5  # Assume 50% turnover
                rebalance_cost = current_portfolio_value * turnover * trading_fee
                net_value = current_portfolio_value - rebalance_cost
                
                # Reallocate shares based on target weights
                for symbol in selected_symbols:
                    allocation = net_value * normalized_weights[symbol]
                    portfolio_shares[symbol] = allocation / price_data[symbol].iloc[i]
                
                portfolio_value = net_value
                rebalance_costs.append(rebalance_cost)
            else:
                # No rebalancing - value changes with market
                portfolio_value = current_portfolio_value
                rebalance_costs.append(0)
            
            # BTC benchmark
            if 'BTC-USD' in price_data.columns:
                btc_return = price_data['BTC-USD'].iloc[i] / price_data['BTC-USD'].iloc[i-1] - 1
                btc_value = btc_values[-1] * (1 + btc_return)
            else:
                btc_value = btc_values[-1]
            
            # ETH benchmark
            if 'ETH-USD' in price_data.columns:
                eth_return = price_data['ETH-USD'].iloc[i] / price_data['ETH-USD'].iloc[i-1] - 1
                eth_value = eth_values[-1] * (1 + eth_return)
            else:
                eth_value = eth_values[-1]
            
            # Equal weight benchmark (also with rebalancing)
            equal_weight_return = sum(equal_weights[symbol] * (price_data[symbol].iloc[i] / price_data[symbol].iloc[i-1] - 1) for symbol in selected_symbols)
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
    
    # Rebalancing statistics
    total_rebalance_cost = sum(rebalance_costs)
    num_rebalances = len([cost for cost in rebalance_costs if cost > 0])
    
    st.subheader("Rebalancing Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rebalancing Frequency", rebalance_freq)
    with col2:
        st.metric("Number of Rebalances", f"{num_rebalances}")
    with col3:
        st.metric("Total Trading Costs", f"${total_rebalance_cost:,.0f}")
    
    if num_rebalances > 0:
        st.info(f"Portfolio rebalanced {num_rebalances} times during the period. Trading costs: {total_rebalance_cost/initial_capital:.2%} of initial capital.")
    
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

elif mode == "ML Predictions":
    st.header("Machine Learning Predictions")
    
    # Get cached data
    price_data, returns = get_cached_data(
        selected_symbols, 
        start_date.strftime('%Y-%m-%d'), 
        end_date.strftime('%Y-%m-%d')
    )
    
    # ML Configuration in sidebar
    st.sidebar.subheader("ML Configuration")
    prediction_days = st.sidebar.slider("Prediction Period (days)", 1, 30, 7)
    train_window = st.sidebar.slider("Training Window (days)", 30, 365, 90)
    
    # Feature engineering function
    def create_features(prices, window=20):
        features = pd.DataFrame()
        
        # Price-based features
        features['returns'] = prices.pct_change()
        features['returns_lag1'] = features['returns'].shift(1)
        features['returns_lag2'] = features['returns'].shift(2)
        features['returns_lag3'] = features['returns'].shift(3)
        
        # Moving averages
        features['ma_5'] = prices.rolling(5).mean() / prices
        features['ma_10'] = prices.rolling(10).mean() / prices
        features['ma_20'] = prices.rolling(20).mean() / prices
        
        # Volatility features
        features['volatility_5'] = features['returns'].rolling(5).std()
        features['volatility_10'] = features['returns'].rolling(10).std()
        
        # Momentum features
        features['rsi'] = calculate_rsi(prices, 14)
        features['momentum_5'] = prices / prices.shift(5) - 1
        features['momentum_10'] = prices / prices.shift(10) - 1
        
        return features.dropna()
    
    # RSI calculation
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # ML Prediction function
    def predict_returns(prices, target_days=7, train_window=90):
        features = create_features(prices)
        
        # Prepare target (future returns)
        target = prices.shift(-target_days) / prices - 1
        
        # Align features and target
        valid_idx = features.index.intersection(target.index)
        X = features.loc[valid_idx].fillna(0)
        y = target.loc[valid_idx].fillna(0)
        
        if len(X) < train_window:
            return None, None, None
        
        # Use last train_window days for training
        X_train = X.iloc[-train_window:-target_days] if len(X) > target_days else X.iloc[:-target_days]
        y_train = y.iloc[-train_window:-target_days] if len(y) > target_days else y.iloc[:-target_days]
        
        if len(X_train) < 20:  # Minimum samples required
            return None, None, None
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'Linear Regression': LinearRegression()
        }
        
        predictions = {}
        metrics = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                
                # Predict on the last available features
                last_features = X.iloc[-1:].fillna(0)
                last_features_scaled = scaler.transform(last_features)
                pred = model.predict(last_features_scaled)[0]
                predictions[name] = pred
                
                # Calculate metrics on training data
                train_pred = model.predict(X_train_scaled)
                metrics[name] = {
                    'MSE': mean_squared_error(y_train, train_pred),
                    'MAE': mean_absolute_error(y_train, train_pred),
                    'R2': r2_score(y_train, train_pred) if len(set(y_train)) > 1 else 0
                }
            except Exception as e:
                predictions[name] = 0
                metrics[name] = {'MSE': float('inf'), 'MAE': float('inf'), 'R2': 0}
        
        return predictions, metrics, X_train.index
    
    # Run predictions for each cryptocurrency
    st.subheader("Individual Asset Predictions")
    
    prediction_results = {}
    all_predictions_data = []
    
    # Collect all prediction data first
    for symbol in selected_symbols:
        prices = price_data[symbol]
        predictions, metrics, train_dates = predict_returns(prices, prediction_days, train_window)
        
        if predictions is None:
            st.warning(f"Insufficient data for {symbol.replace('-USD', '')} predictions")
            continue
        
        prediction_results[symbol] = predictions
        
        # Collect data for visualization
        asset_name = symbol.replace('-USD', '')
        for model_name, pred in predictions.items():
            all_predictions_data.append({
                'Asset': asset_name,
                'Model': model_name,
                'Prediction': pred,
                'R2_Score': metrics[model_name]['R2'],
                'MAE': metrics[model_name]['MAE']
            })
    
    if all_predictions_data:
        predictions_df = pd.DataFrame(all_predictions_data)
        
        # Create comprehensive prediction visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart for predictions by asset and model
            fig_bar = px.bar(
                predictions_df,
                x='Asset',
                y='Prediction',
                color='Model',
                title=f"Predicted Returns by Asset ({prediction_days} days)",
                labels={'Prediction': 'Predicted Return'},
                text=[f"{val:.1%}" for val in predictions_df['Prediction']],
                color_discrete_map={
                    'Random Forest': '#2E8B57',
                    'Linear Regression': '#4169E1'
                }
            )
            fig_bar.update_traces(textposition='outside')
            
            # Calculate y-axis range with generous padding for text visibility
            y_min = predictions_df['Prediction'].min()
            y_max = predictions_df['Prediction'].max()
            y_range = y_max - y_min
            
            # Ensure sufficient padding for text labels above bars
            top_padding = max(0.04, y_range * 0.5)  # At least 4% or 50% of range for top
            bottom_padding = max(0.02, y_range * 0.2)  # At least 2% or 20% of range for bottom
            
            fig_bar.update_layout(
                template='plotly_white',
                height=400,
                yaxis_tickformat='.1%',
                xaxis_title="Cryptocurrency",
                yaxis_title="Predicted Return",
                yaxis=dict(
                    range=[y_min - bottom_padding, y_max + top_padding]
                )
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Model agreement analysis
            agreement_data = []
            for symbol in selected_symbols:
                if symbol in prediction_results:
                    asset_name = symbol.replace('-USD', '')
                    rf_pred = prediction_results[symbol].get('Random Forest', 0)
                    lr_pred = prediction_results[symbol].get('Linear Regression', 0)
                    
                    # Calculate agreement metrics
                    avg_pred = (rf_pred + lr_pred) / 2
                    agreement_score = 1 - abs(rf_pred - lr_pred) / max(abs(avg_pred), 0.01)
                    agreement_score = max(0, min(1, agreement_score)) * 100  # Convert to 0-100%
                    
                    agreement_data.append({
                        'Asset': asset_name,
                        'Agreement_Score': agreement_score,
                        'Difference': abs(rf_pred - lr_pred) * 100,  # Convert to percentage points
                        'Average_Prediction': avg_pred * 100
                    })
            
            if agreement_data:
                agreement_df = pd.DataFrame(agreement_data)
                
                # Create model agreement visualization
                fig_agreement = px.bar(
                    agreement_df,
                    x='Asset',
                    y='Agreement_Score',
                    color='Agreement_Score',
                    color_continuous_scale='RdYlGn',
                    title="Model Agreement Score by Asset",
                    labels={'Agreement_Score': 'Agreement Score (%)'},
                    text=[f"{val:.0f}%" for val in agreement_df['Agreement_Score']],
                    hover_data={
                        'Difference': ':.1f',
                        'Average_Prediction': ':.1f'
                    }
                )
                
                fig_agreement.update_traces(textposition='outside')
                fig_agreement.update_layout(
                    template='plotly_white',
                    height=400,
                    xaxis_title="Cryptocurrency",
                    yaxis_title="Model Agreement (%)",
                    yaxis=dict(range=[0, 105]),
                    showlegend=False
                )
                
                st.plotly_chart(fig_agreement, use_container_width=True)
        
        # Detailed predictions table
        st.subheader("Detailed Predictions Summary")
        
        # Create a more readable table
        table_data = []
        for symbol in selected_symbols:
            if symbol in prediction_results:
                asset_name = symbol.replace('-USD', '')
                rf_pred = prediction_results[symbol].get('Random Forest', 0)
                lr_pred = prediction_results[symbol].get('Linear Regression', 0)
                avg_pred = (rf_pred + lr_pred) / 2
                
                table_data.append({
                    'Asset': asset_name,
                    'Random Forest': f"{rf_pred:.2%}",
                    'Linear Regression': f"{lr_pred:.2%}",
                    'Average': f"{avg_pred:.2%}",
                    'Confidence': "High" if abs(rf_pred - lr_pred) < 0.02 else "Medium" if abs(rf_pred - lr_pred) < 0.05 else "Low"
                })
        
        if table_data:
            summary_df = pd.DataFrame(table_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Portfolio-level predictions
    if prediction_results:
        st.subheader("Portfolio-Level Predictions")
        
        # Allow user to input portfolio weights
        st.write("**Define Portfolio Weights for Prediction**")
        portfolio_weights = {}
        cols = st.columns(min(len(selected_symbols), 3))
        
        total_weight = 0
        for i, symbol in enumerate(selected_symbols):
            with cols[i % len(cols)]:
                weight = st.number_input(
                    f"{symbol.replace('-USD', '')} Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0/len(selected_symbols),
                    step=0.05,
                    key=f"ml_weight_{symbol}"
                )
                portfolio_weights[symbol] = weight
                total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            normalized_weights = {k: v/total_weight for k, v in portfolio_weights.items()}
        else:
            normalized_weights = {k: 1.0/len(selected_symbols) for k in selected_symbols}
        
        # Calculate portfolio predictions
        portfolio_predictions = {'Random Forest': 0, 'Linear Regression': 0}
        
        for symbol in selected_symbols:
            if symbol in prediction_results:
                for model_name in portfolio_predictions.keys():
                    portfolio_predictions[model_name] += (
                        normalized_weights[symbol] * prediction_results[symbol][model_name]
                    )
        
        # Display portfolio predictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Random Forest Portfolio Prediction",
                f"{portfolio_predictions['Random Forest']:.2%}",
                delta=f"{prediction_days}-day forecast"
            )
        
        with col2:
            st.metric(
                "Linear Regression Portfolio Prediction", 
                f"{portfolio_predictions['Linear Regression']:.2%}",
                delta=f"{prediction_days}-day forecast"
            )
        
        # Ensemble prediction (average)
        ensemble_prediction = np.mean(list(portfolio_predictions.values()))
        st.metric(
            "Ensemble Portfolio Prediction",
            f"{ensemble_prediction:.2%}",
            delta="Average of all models"
        )
        
        # Visualization of predictions
        st.subheader("Prediction Visualization")
        
        # Create prediction chart
        fig_pred = go.Figure()
        
        models = list(portfolio_predictions.keys())
        predictions_vals = list(portfolio_predictions.values())
        
        fig_pred.add_trace(go.Bar(
            x=models + ['Ensemble'],
            y=predictions_vals + [ensemble_prediction],
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
            text=[f"{val:.2%}" for val in predictions_vals + [ensemble_prediction]],
            textposition='auto'
        ))
        
        fig_pred.update_layout(
            title=f"Portfolio Return Predictions ({prediction_days} days)",
            yaxis_title="Predicted Return",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Enhanced Asset Analysis
        st.subheader("Asset Analysis Dashboard")
        
        # Create comprehensive comparison visualization
        comparison_data = []
        for symbol in selected_symbols:
            if symbol in prediction_results:
                asset_name = symbol.replace('-USD', '')
                rf_pred = prediction_results[symbol].get('Random Forest', 0)
                lr_pred = prediction_results[symbol].get('Linear Regression', 0)
                avg_pred = (rf_pred + lr_pred) / 2
                weight = normalized_weights[symbol]
                
                comparison_data.append({
                    'Asset': asset_name,
                    'Weight': weight,
                    'Random_Forest': rf_pred,
                    'Linear_Regression': lr_pred,
                    'Average_Prediction': avg_pred,
                    'Weighted_Impact': weight * avg_pred,
                    'Weight_Category': 'High (>20%)' if weight > 0.2 else 'Medium (5-20%)' if weight > 0.05 else 'Low (<5%)'
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bubble chart: Weight vs Prediction with Impact as size
                fig_bubble = px.scatter(
                    comparison_df,
                    x='Weight',
                    y='Average_Prediction',
                    size=[abs(x)*100+10 for x in comparison_df['Weighted_Impact']],
                    color='Weight_Category',
                    hover_name='Asset',
                    hover_data={
                        'Weight': ':.1%',
                        'Average_Prediction': ':.2%',
                        'Weighted_Impact': ':.3%',
                        'Weight_Category': False
                    },
                    title="Portfolio Weight vs Predicted Return",
                    labels={
                        'Weight': 'Portfolio Weight',
                        'Average_Prediction': 'Average Predicted Return'
                    },
                    color_discrete_map={
                        'High (>20%)': '#FF6B6B',
                        'Medium (5-20%)': '#4ECDC4', 
                        'Low (<5%)': '#95A5A6'
                    }
                )
                
                # Add annotations for each asset
                for _, row in comparison_df.iterrows():
                    fig_bubble.add_annotation(
                        x=row['Weight'],
                        y=row['Average_Prediction'],
                        text=row['Asset'],
                        showarrow=False,
                        font=dict(size=10),
                        yshift=15
                    )
                
                fig_bubble.update_layout(
                    template='plotly_white',
                    height=450,
                    xaxis_tickformat='.1%',
                    yaxis_tickformat='.1%'
                )
                
                st.plotly_chart(fig_bubble, use_container_width=True)
            
            with col2:
                # Risk-Return prediction quadrant
                risk_return_data = []
                for _, row in comparison_df.iterrows():
                    # Calculate prediction volatility (difference between models as proxy for uncertainty)
                    pred_uncertainty = abs(row['Random_Forest'] - row['Linear_Regression'])
                    
                    risk_return_data.append({
                        'Asset': row['Asset'],
                        'Expected_Return': row['Average_Prediction'] * 100,
                        'Prediction_Uncertainty': pred_uncertainty * 100,
                        'Weight': row['Weight'],
                        'Size_Factor': row['Weight'] * 1000 + 20  # For bubble sizing
                    })
                
                risk_return_df = pd.DataFrame(risk_return_data)
                
                fig_risk_return = px.scatter(
                    risk_return_df,
                    x='Prediction_Uncertainty',
                    y='Expected_Return',
                    size='Size_Factor',
                    color='Expected_Return',
                    color_continuous_scale='RdYlGn',
                    hover_name='Asset',
                    hover_data={
                        'Weight': ':.1%',
                        'Prediction_Uncertainty': ':.2f',
                        'Expected_Return': ':.2f',
                        'Size_Factor': False
                    },
                    title="Return vs Prediction Uncertainty",
                    labels={
                        'Prediction_Uncertainty': 'Model Disagreement (%)',
                        'Expected_Return': 'Expected Return (%)'
                    }
                )
                
                # Add asset labels
                for _, row in risk_return_df.iterrows():
                    fig_risk_return.add_annotation(
                        x=row['Prediction_Uncertainty'],
                        y=row['Expected_Return'],
                        text=row['Asset'],
                        showarrow=False,
                        font=dict(size=9),
                        yshift=15
                    )
                
                fig_risk_return.update_layout(
                    template='plotly_white',
                    height=450,
                    showlegend=False
                )
                
                st.plotly_chart(fig_risk_return, use_container_width=True)
            
            # Portfolio impact summary table
            st.subheader("Portfolio Impact Analysis")
            
            impact_df = comparison_df[['Asset', 'Weight', 'Average_Prediction', 'Weighted_Impact']].copy()
            impact_df['Weight'] = impact_df['Weight'].apply(lambda x: f"{x:.1%}")
            impact_df['Average_Prediction'] = impact_df['Average_Prediction'].apply(lambda x: f"{x:.2%}")
            impact_df['Weighted_Impact'] = impact_df['Weighted_Impact'].apply(lambda x: f"{x:.3%}")
            impact_df = impact_df.rename(columns={
                'Average_Prediction': 'Predicted Return',
                'Weighted_Impact': 'Portfolio Impact'
            })
            
            # Sort by absolute portfolio impact
            impact_df['Sort_Key'] = comparison_df['Weighted_Impact'].abs()
            impact_df = impact_df.sort_values('Sort_Key', ascending=False).drop('Sort_Key', axis=1)
            
            st.dataframe(impact_df, use_container_width=True, hide_index=True)
    
    # Disclaimer
    st.subheader("Important Disclaimer")
    st.warning("""
    **Machine Learning Predictions Disclaimer:**
    - These predictions are based on historical patterns and technical indicators
    - Cryptocurrency markets are highly volatile and unpredictable
    - Past performance does not guarantee future results
    - ML models can be inaccurate, especially during market regime changes
    - Use predictions as one factor among many for decision-making
    - Never invest based solely on algorithmic predictions
    """)

elif mode == "AI Investment Advisor":
    st.header("AI Investment Advisor")
    
    # API Configuration in sidebar
    st.sidebar.subheader("AI Configuration")
    
    # LLM Provider selection
    llm_provider = st.sidebar.selectbox(
        "Select AI Provider",
        ["OpenAI (GPT-4)", "Anthropic (Claude)", "Google (Gemini)"]
    )
    
    # API Key input
    api_key = st.sidebar.text_input(
        "API Key",
        type="password",
        help="Enter your API key for the selected provider"
    )
    
    if not api_key:
        st.warning("Please enter your API key in the sidebar to get AI investment advice.")
        st.info("""
        **How to get API keys:**
        - **OpenAI**: Visit https://platform.openai.com/api-keys
        - **Anthropic**: Visit https://console.anthropic.com/
        - **Google**: Visit https://ai.google.dev/
        """)
        st.stop()
    
    # Risk tolerance for AI analysis
    risk_tolerance = st.sidebar.selectbox(
        "Risk Tolerance",
        ["Conservative", "Moderate", "Aggressive"]
    )
    
    # Investment horizon
    investment_horizon = st.sidebar.selectbox(
        "Investment Horizon",
        ["Short-term (1-3 months)", "Medium-term (3-12 months)", "Long-term (1+ years)"]
    )
    
    # Current Portfolio Definition
    st.subheader("Define Your Current Portfolio")
    
    portfolio_option = st.radio(
        "How would you like to define your current portfolio?",
        ["Equal Weight (Default)", "Custom Allocation", "Portfolio Analysis Mode Weights"],
        help="Choose how to represent your current portfolio holdings"
    )
    
    current_portfolio_weights = {}
    
    if portfolio_option == "Equal Weight (Default)":
        # Equal weight allocation
        equal_weight = 100 / len(selected_symbols)
        for symbol in selected_symbols:
            current_portfolio_weights[symbol.replace('-USD', '')] = equal_weight
        
        st.info(f"Using equal weight allocation: {equal_weight:.1f}% per asset")
        
    elif portfolio_option == "Custom Allocation":
        st.write("**Enter your current portfolio allocation (%):**")
        
        # Create columns for input
        n_cols = min(len(selected_symbols), 3)
        cols = st.columns(n_cols)
        
        total_weight = 0
        for i, symbol in enumerate(selected_symbols):
            col_idx = i % n_cols
            with cols[col_idx]:
                weight = st.number_input(
                    f"{symbol.replace('-USD', '')} %", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=100.0/len(selected_symbols),
                    step=5.0,
                    key=f"current_weight_{symbol}"
                )
                current_portfolio_weights[symbol.replace('-USD', '')] = weight
                total_weight += weight
        
        # Validation
        if abs(total_weight - 100) > 0.1:
            st.error(f"Portfolio weights must sum to 100%. Current total: {total_weight:.1f}%")
        else:
            st.success("✓ Portfolio weights sum to 100%")
            
    else:  # Portfolio Analysis Mode Weights
        st.info("Using weights from Portfolio Analysis mode (if available)")
        # Check if portfolio weights exist in session state
        if hasattr(st.session_state, 'portfolio_weights'):
            current_portfolio_weights = {symbol.replace('-USD', ''): weight*100 
                                       for symbol, weight in st.session_state.portfolio_weights.items()}
        else:
            st.warning("No Portfolio Analysis weights found. Using equal weight instead.")
            equal_weight = 100 / len(selected_symbols)
            for symbol in selected_symbols:
                current_portfolio_weights[symbol.replace('-USD', '')] = equal_weight
    
    # Display current portfolio
    if current_portfolio_weights:
        st.write("**Your Current Portfolio:**")
        current_df = pd.DataFrame({
            'Asset': list(current_portfolio_weights.keys()),
            'Current %': [f"{w:.1f}%" for w in current_portfolio_weights.values()]
        })
        st.dataframe(current_df, use_container_width=True, hide_index=True)
    
    # Get cached data
    price_data, returns = get_cached_data(
        selected_symbols, 
        start_date.strftime('%Y-%m-%d'), 
        end_date.strftime('%Y-%m-%d')
    )
    
    # LLM API functions
    def call_openai_api(messages, api_key):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-4",
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0.3
        }
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", 
                                   headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"
    
    def call_anthropic_api(message, api_key):
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": message}]
        }
        try:
            response = requests.post("https://api.anthropic.com/v1/messages",
                                   headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                return response.json()["content"][0]["text"]
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error calling Anthropic API: {str(e)}"
    
    def call_gemini_api(message, api_key):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": message}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 2000
            }
        }
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error calling Gemini API: {str(e)}"
    
    # Advanced ML prediction function for AI analysis
    def run_ml_predictions_for_ai():
        ml_results = {}
        
        # Feature engineering function (same as ML Predictions mode)
        def create_features(prices, window=20):
            features = pd.DataFrame()
            features['returns'] = prices.pct_change()
            features['returns_lag1'] = features['returns'].shift(1)
            features['returns_lag2'] = features['returns'].shift(2)
            features['ma_5'] = prices.rolling(5).mean() / prices
            features['ma_10'] = prices.rolling(10).mean() / prices
            features['ma_20'] = prices.rolling(20).mean() / prices
            features['volatility_5'] = features['returns'].rolling(5).std()
            features['volatility_10'] = features['returns'].rolling(10).std()
            
            # RSI calculation
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features['rsi'] = rsi
            
            features['momentum_5'] = prices / prices.shift(5) - 1
            features['momentum_10'] = prices / prices.shift(10) - 1
            return features.dropna()
        
        # ML prediction function (same as ML Predictions mode)
        def predict_returns(prices, target_days=7, train_window=90):
            features = create_features(prices)
            target = prices.shift(-target_days) / prices - 1
            valid_idx = features.index.intersection(target.index)
            X = features.loc[valid_idx].fillna(0)
            y = target.loc[valid_idx].fillna(0)
            
            if len(X) < train_window:
                return None
            
            X_train = X.iloc[-train_window:-target_days] if len(X) > target_days else X.iloc[:-target_days]
            y_train = y.iloc[-train_window:-target_days] if len(y) > target_days else y.iloc[:-target_days]
            
            if len(X_train) < 20:
                return None
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
                'Linear Regression': LinearRegression()
            }
            
            predictions = {}
            for name, model in models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    last_features = X.iloc[-1:].fillna(0)
                    last_features_scaled = scaler.transform(last_features)
                    pred = model.predict(last_features_scaled)[0]
                    predictions[name] = pred
                except:
                    predictions[name] = 0
            
            return predictions
        
        # Run ML predictions for each asset
        for symbol in selected_symbols:
            asset_name = symbol.replace('-USD', '')
            prices = price_data[symbol]
            predictions = predict_returns(prices, 7, 90)
            
            if predictions:
                rf_pred = predictions.get('Random Forest', 0)
                lr_pred = predictions.get('Linear Regression', 0)
                ensemble_pred = (rf_pred + lr_pred) / 2
                
                ml_results[asset_name] = {
                    "random_forest_7d": f"{rf_pred:.2%}",
                    "linear_regression_7d": f"{lr_pred:.2%}",
                    "ensemble_prediction_7d": f"{ensemble_pred:.2%}",
                    "model_agreement": "High" if abs(rf_pred - lr_pred) < 0.02 else "Medium" if abs(rf_pred - lr_pred) < 0.05 else "Low"
                }
        
        return ml_results
    
    # Portfolio optimization results for AI analysis
    def run_portfolio_optimization_for_ai():
        optimization_results = {}
        
        try:
            optimizer = PortfolioOptimizer()
            returns_array = returns.values
            
            # Run different optimization methods
            methods = {
                "mean_variance": "conservative",
                "risk_parity": None,
                "max_sharpe": None
            }
            
            for method, risk_level in methods.items():
                try:
                    if method == "mean_variance":
                        result = optimizer.optimize_portfolio(returns_array, method=method, risk_tolerance=risk_level)
                    else:
                        result = optimizer.optimize_portfolio(returns_array, method=method)
                    
                    if result.get('success', False):
                        weights = result['weights']
                        allocation = {selected_symbols[i].replace('-USD', ''): f"{weights[i]:.1%}" 
                                    for i in range(len(selected_symbols))}
                        
                        optimization_results[method] = {
                            "allocation": allocation,
                            "expected_return": f"{result.get('expected_return', 0):.2%}",
                            "volatility": f"{result.get('volatility', 0):.2%}",
                            "sharpe_ratio": f"{result.get('sharpe_ratio', 0):.2f}"
                        }
                except:
                    continue
        except:
            pass
        
        return optimization_results
    
    # Generate comprehensive analysis data
    def generate_analysis_data():
        analysis_data = {
            "selected_assets": [symbol.replace('-USD', '') for symbol in selected_symbols],
            "time_period": f"{start_date} to {end_date}",
            "market_data": {},
            "ml_predictions": {},
            "portfolio_optimization": {},
            "correlation_analysis": {}
        }
        
        # Market data analysis
        for symbol in selected_symbols:
            asset_name = symbol.replace('-USD', '')
            current_price = price_data[symbol].iloc[-1]
            start_price = price_data[symbol].iloc[0]
            total_return = (current_price / start_price - 1) * 100
            volatility = returns[symbol].std() * np.sqrt(252) * 100
            
            analysis_data["market_data"][asset_name] = {
                "total_return": f"{total_return:.1f}%",
                "volatility": f"{volatility:.1f}%",
                "current_price": f"${current_price:.2f}"
            }
        
        # Get ML predictions
        analysis_data["ml_predictions"] = run_ml_predictions_for_ai()
        
        # Get portfolio optimization results
        analysis_data["portfolio_optimization"] = run_portfolio_optimization_for_ai()
        
        # Correlation analysis
        try:
            corr_matrix = returns.corr()
            avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            
            analysis_data["correlation_analysis"] = {
                "average_correlation": f"{avg_correlation:.2f}",
                "diversification_potential": "Low" if avg_correlation > 0.7 else "Medium" if avg_correlation > 0.3 else "High"
            }
        except:
            pass
        
        return analysis_data
    
    # Generate AI analysis
    if st.button("Get AI Investment Advice", type="primary"):
        with st.spinner("Generating AI analysis..."):
            analysis_data = generate_analysis_data()
            
            # Create structured prompt for AI
            prompt = f"""
            As a crypto portfolio advisor, analyze the comprehensive data and provide a structured response in JSON format:
            
            **Portfolio Configuration:**
            - Selected Assets: {', '.join(analysis_data['selected_assets'])}
            - Risk Tolerance: {risk_tolerance}
            - Investment Horizon: {investment_horizon}
            - Analysis Period: {analysis_data['time_period']}
            
            **Market Performance Data:**
            {json.dumps(analysis_data['market_data'], indent=2)}
            
            **Machine Learning Predictions (7-day forecasts):**
            {json.dumps(analysis_data['ml_predictions'], indent=2)}
            
            **Portfolio Optimization Results (Multiple Methods):**
            {json.dumps(analysis_data['portfolio_optimization'], indent=2)}
            
            **Correlation Analysis:**
            {json.dumps(analysis_data['correlation_analysis'], indent=2)}
            
            **Current Portfolio (User Defined):**
            {json.dumps(current_portfolio_weights, indent=2)}
            
            **Instructions:**
            Analyze the data and recommend a target portfolio allocation for {investment_horizon}. Use the user's current portfolio above as starting point and provide rebalancing plan.
            
            **Required Response Format (JSON only):**
            {{
                "current_portfolio": {json.dumps(current_portfolio_weights)},
                "recommended_portfolio": {{
                    "BTC": 35,
                    "ETH": 25,
                    "provide_percentages_for_all_selected_assets": "that sum to 100"
                }},
                "rebalancing_plan": {{
                    "portfolio_value": 100000,
                    "timeframe": "{investment_horizon}",
                    "actions": [
                        {{"asset": "BTC", "action": "BUY", "amount": 8500, "from_percent": 20, "to_percent": 28.5, "reason": "Strong momentum"}},
                        {{"asset": "ETH", "action": "SELL", "amount": 3000, "from_percent": 20, "to_percent": 17, "reason": "High volatility"}}
                    ]
                }},
                "portfolio_metrics": {{
                    "expected_annual_return": "15.2%",
                    "expected_volatility": "28.5%",
                    "sharpe_ratio": 0.53,
                    "max_drawdown_estimate": "45%"
                }},
                "implementation": {{
                    "timeline": "Rebalance over 2-3 trading days",
                    "order_strategy": "Use limit orders, place 1-2% away from market",
                    "risk_controls": "Set stop losses at 15% below entry for each position"
                }}
            }}
            
            Provide ONLY the JSON response. Ensure all selected assets are included with realistic allocations that sum to 100%.
            """
            
            # Call selected LLM API
            try:
                if "OpenAI" in llm_provider:
                    messages = [
                        {"role": "system", "content": "You are a professional cryptocurrency investment advisor with expertise in portfolio analysis and risk management."},
                        {"role": "user", "content": prompt}
                    ]
                    ai_response = call_openai_api(messages, api_key)
                elif "Anthropic" in llm_provider:
                    ai_response = call_anthropic_api(prompt, api_key)
                elif "Google" in llm_provider:
                    ai_response = call_gemini_api(prompt, api_key)
                else:
                    ai_response = "Error: Invalid LLM provider selected"
                
                # Display AI analysis
                st.subheader("AI Investment Analysis")
                
                if ai_response.startswith("Error"):
                    st.error(ai_response)
                    st.info("Please check your API key and try again.")
                else:
                    try:
                        # Parse JSON response
                        import re
                        # Extract JSON from response (in case there's extra text)
                        json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            ai_data = json.loads(json_str)
                            
                            # 1. Portfolio Comparison
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Current Portfolio")
                                if "current_portfolio" in ai_data:
                                    current_data = ai_data["current_portfolio"]
                                    assets = list(current_data.keys())
                                    weights = [float(w) for w in current_data.values()]
                                    
                                    fig_current = px.pie(
                                        values=weights,
                                        names=assets,
                                        title="Current Allocation (Equal Weight)",
                                        color_discrete_sequence=['#F7931A', '#627EEA', '#F3BA2F', '#00AAE4', '#0033AD', '#9945FF']
                                    )
                                    fig_current.update_traces(textposition='inside', textinfo='percent+label')
                                    fig_current.update_layout(height=350, template='plotly_white')
                                    st.plotly_chart(fig_current, use_container_width=True)
                            
                            with col2:
                                st.subheader("AI Recommended Portfolio")
                                if "recommended_portfolio" in ai_data:
                                    rec_data = ai_data["recommended_portfolio"]
                                    assets = list(rec_data.keys())
                                    weights = [float(w) for w in rec_data.values()]
                                    
                                    fig_recommended = px.pie(
                                        values=weights,
                                        names=assets,
                                        title="Recommended Allocation",
                                        color_discrete_sequence=['#F7931A', '#627EEA', '#F3BA2F', '#00AAE4', '#0033AD', '#9945FF']
                                    )
                                    fig_recommended.update_traces(textposition='inside', textinfo='percent+label')
                                    fig_recommended.update_layout(height=350, template='plotly_white')
                                    st.plotly_chart(fig_recommended, use_container_width=True)
                            
                            # 2. Rebalancing Plan
                            st.subheader("Precise Rebalancing Plan")
                            
                            if "rebalancing_plan" in ai_data:
                                plan = ai_data["rebalancing_plan"]
                                
                                # Portfolio details
                                col1, col2 = st.columns(2)
                                with col1:
                                    portfolio_val = plan.get("portfolio_value", 100000)
                                    st.metric("Portfolio Value", f"${portfolio_val:,}")
                                with col2:
                                    timeframe = plan.get("timeframe", "Not specified")
                                    st.metric("Timeframe", timeframe)
                                
                                # Rebalancing actions
                                if "actions" in plan and plan["actions"]:
                                    st.markdown("**Required Trades:**")
                                    
                                    for action in plan["actions"]:
                                        asset = action.get("asset", "")
                                        action_type = action.get("action", "")
                                        amount = action.get("amount", 0)
                                        from_pct = action.get("from_percent", 0)
                                        to_pct = action.get("to_percent", 0)
                                        reason = action.get("reason", "")
                                        
                                        color = "green" if action_type == "BUY" else "red" if action_type == "SELL" else "blue"
                                        arrow = "↑" if action_type == "BUY" else "↓" if action_type == "SELL" else "→"
                                        
                                        st.markdown(f"""
                                        <div style='border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 8px; background-color: #f8f9fa;'>
                                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                                        <div>
                                        <strong style='font-size: 18px;'>{asset}</strong>: 
                                        <span style='color:{color}; font-weight:bold; font-size: 16px;'>{action_type} ${amount:,}</span>
                                        </div>
                                        <div style='text-align: right;'>
                                        <span style='font-size: 24px;'>{arrow}</span>
                                        </div>
                                        </div>
                                        <div style='margin-top: 8px;'>
                                        <span style='background-color: #e9ecef; padding: 2px 6px; border-radius: 4px; font-size: 12px;'>
                                        {from_pct}% → {to_pct}%
                                        </span>
                                        <br><small style='color: #6c757d; margin-top: 4px; display: block;'>{reason}</small>
                                        </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                            
                            # 3. Expected Portfolio Performance
                            st.subheader("Expected Portfolio Performance")
                            
                            if "portfolio_metrics" in ai_data:
                                metrics = ai_data["portfolio_metrics"]
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    return_val = metrics.get("expected_annual_return", "N/A")
                                    st.metric("Expected Return", return_val, delta="Annual")
                                
                                with col2:
                                    vol_val = metrics.get("expected_volatility", "N/A")
                                    st.metric("Expected Volatility", vol_val, delta="Risk Level")
                                
                                with col3:
                                    sharpe_val = metrics.get("sharpe_ratio", "N/A")
                                    st.metric("Sharpe Ratio", f"{sharpe_val}" if isinstance(sharpe_val, (int, float)) else sharpe_val)
                                
                                with col4:
                                    drawdown_val = metrics.get("max_drawdown_estimate", "N/A")
                                    st.metric("Max Drawdown", drawdown_val, delta="Worst Case")
                            
                            # 4. Implementation Guide
                            st.subheader("Implementation Guide")
                            
                            if "implementation" in ai_data:
                                impl = ai_data["implementation"]
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.info(f"**Timeline:** {impl.get('timeline', 'Not specified')}")
                                    st.info(f"**Order Strategy:** {impl.get('order_strategy', 'Market orders')}")
                                
                                with col2:
                                    st.warning(f"**Risk Controls:** {impl.get('risk_controls', 'Set appropriate stop losses')}")
                                    st.success("**Final Step:** Monitor positions daily and adjust as market conditions change")
                        
                        else:
                            st.error("Could not parse AI response as JSON. Raw response:")
                            st.text(ai_response)
                            
                    except json.JSONDecodeError as e:
                        st.error("Failed to parse AI response as JSON. Raw response:")
                        st.text(ai_response)
                        st.error(f"JSON Error: {str(e)}")
                    except Exception as e:
                        st.error(f"Error processing AI response: {str(e)}")
                        st.text(ai_response)
            
            except Exception as e:
                st.error(f"Error generating AI analysis: {str(e)}")
    
    # Add expandable section for data sources
    with st.expander("View Analysis Data Sources"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Market Performance Data**")
            if 'analysis_data' in locals():
                market_source_df = pd.DataFrame(analysis_data["market_data"]).T
                st.dataframe(market_source_df, use_container_width=True)
        
        with col2:
            st.write("**Technical Indicators**")
            if 'analysis_data' in locals() and analysis_data["ml_predictions"]:
                tech_source_df = pd.DataFrame(analysis_data["ml_predictions"]).T
                st.dataframe(tech_source_df, use_container_width=True)
            else:
                st.write("No technical data available")
    
    # Portfolio analysis section
    st.subheader("Current Portfolio Overview")
    
    # Display market metrics
    market_data = []
    for symbol in selected_symbols:
        asset_name = symbol.replace('-USD', '')
        current_price = price_data[symbol].iloc[-1]
        start_price = price_data[symbol].iloc[0]
        total_return = (current_price / start_price - 1) * 100
        volatility = returns[symbol].std() * np.sqrt(252) * 100
        
        # Risk assessment
        if volatility > 100:
            risk_level = "Very High"
        elif volatility > 70:
            risk_level = "High"
        elif volatility > 40:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        market_data.append({
            'Asset': asset_name,
            'Current Price': f"${current_price:.2f}",
            'Total Return': f"{total_return:.1f}%",
            'Volatility': f"{volatility:.1f}%",
            'Risk Level': risk_level
        })
    
    market_df = pd.DataFrame(market_data)
    st.dataframe(market_df, use_container_width=True, hide_index=True)
    
    # Important disclaimers
    st.subheader("Important Disclaimers")
    st.error("""
    **AI Investment Advice Disclaimer:**
    - AI-generated advice is for informational purposes only
    - Not personalized financial advice - consult qualified professionals
    - AI models may have biases or limitations in analysis
    - Cryptocurrency investments are highly risky and volatile
    - Past performance does not guarantee future results
    - Never invest more than you can afford to lose
    - Always do your own research before making investment decisions
    """)

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