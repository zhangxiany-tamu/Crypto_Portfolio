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
# Removed yahooquery import - using enhanced crypto loader instead
import sys
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# MLPRegressor removed due to convergence issues
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
import warnings
import requests
import json
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from portfolio_optimizer import PortfolioOptimizer
from enhanced_crypto_loader import EnhancedCryptoLoader
import importlib
import technical_analysis
importlib.reload(technical_analysis)
from technical_analysis import TechnicalAnalyzer
import elliott_wave_analyzer
importlib.reload(elliott_wave_analyzer)
from elliott_wave_analyzer import ElliottWaveAnalyzer

# Add backend/core to path for ML imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend', 'core'))
from market_predictor import MarketPredictor

# Data manager adapter for MarketPredictor
class AppDataManager:
    def __init__(self, crypto_loader):
        self.crypto_loader = crypto_loader
    
    async def get_price_data(self, symbols, start_date, end_date):
        """Adapter to get price data in the format expected by MarketPredictor"""
        try:
            # Use the crypto loader to get real data
            data = self.crypto_loader.get_real_data(symbols, start_date, end_date)
            return data
        except Exception as e:
            print(f"Error loading price data: {e}")
            return {}

# Initialize session state for theme - default to light mode
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Helper function to add transparent background to all plots
def make_plot_transparent(fig):
    """Add transparent background to plotly figures for Apple-style integration"""
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# Apply theme-specific CSS (Coinbase-inspired)
def apply_theme_css(theme):
    if theme == 'light':
        return """
<style>
    /* Import Apple-style system fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Apple-inspired Light Mode Variables */
    :root {
        --apple-bg: #ffffff;
        --apple-secondary-bg: #f5f5f7;
        --apple-card-bg: #ffffff;
        --apple-hover-bg: #f0f0f0;
        --apple-border: #d2d2d7;
        --apple-text-primary: #1d1d1f;
        --apple-text-secondary: #6e6e73;
        --apple-text-tertiary: #86868b;
        --apple-blue: #007aff;
        --apple-green: #30d158;
        --apple-red: #ff453a;
        --apple-orange: #ff9f0a;
        --apple-shadow: 0 2px 16px rgba(0, 0, 0, 0.1);
        --apple-shadow-card: 0 4px 20px rgba(0, 0, 0, 0.08);
        --apple-radius: 12px;
        --apple-radius-lg: 16px;
    }
    
    /* Global Apple-style Light Theme */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', 'Segoe UI', system-ui, sans-serif;
        background: var(--apple-secondary-bg);
        color: var(--apple-text-primary);
        font-feature-settings: 'tnum';
        font-size: 17px;
        font-weight: 400;
        line-height: 1.47;
        letter-spacing: -0.022em;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Apple-style main container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1240px;
        margin: 0 auto;
        width: 100%;
        box-sizing: border-box;
    }
    
    /* Apple-style sidebar */
    [data-testid="stSidebar"] {
        background: var(--apple-secondary-bg) !important;
        border-right: 1px solid var(--apple-border) !important;
        box-shadow: none !important;
    }
    
    [data-testid="stSidebar"] > div {
        background: var(--apple-secondary-bg) !important;
        padding: 1.5rem !important;
    }
    
    [data-testid="stSidebar"] label {
        color: var(--apple-text-primary) !important;
        font-weight: 500 !important;
        font-size: 15px !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: var(--apple-text-primary) !important;
        font-weight: 600 !important;
        letter-spacing: -0.022em;
    }
    
    [data-testid="stSidebar"] p {
        color: var(--apple-text-secondary) !important;
        font-size: 15px !important;
        line-height: 1.5;
    }
    
    /* Adaptive main content for wide layout */
    .main .block-container {
        max-width: none !important;
        width: 100% !important;
        padding: 1rem 2rem !important;
    }
    
    /* Apple-style buttons */
    .stButton > button {
        background: var(--apple-blue);
        color: white;
        border: none;
        border-radius: var(--apple-radius);
        padding: 12px 24px;
        font-weight: 500;
        font-size: 17px;
        transition: all 0.15s ease;
        box-shadow: var(--apple-shadow);
        text-transform: none;
        letter-spacing: -0.022em;
        min-height: 44px;
    }
    
    .stButton > button:hover {
        background: #0056d6;
        transform: translateY(-1px);
        box-shadow: var(--apple-shadow-card);
    }
    
    .stButton > button:focus {
        box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.25);
        outline: none;
    }
    
    /* Primary button styling (for Optimize Portfolio button) - Clean Apple style */
    button[kind="primary"],
    button[data-testid="baseButton-primary"],
    .stButton > button[kind="primary"],
    .stButton > button[data-baseweb="button"][kind="primary"],
    .stButton button[data-testid="baseButton-primary"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%) !important;
        color: var(--apple-text-primary) !important;
        border: 1px solid var(--apple-border) !important;
        border-radius: var(--apple-radius-lg) !important;
        padding: 12px 32px !important;
        font-weight: 500 !important;
        font-size: 17px !important;
        transition: all 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08), 0 0 0 1px rgba(255, 255, 255, 0.05) !important;
        text-transform: none !important;
        letter-spacing: -0.022em !important;
        min-height: 44px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-baseweb="button"][kind="primary"]:hover,
    .stButton button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%) !important;
        border-color: #c6c7c8 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12), 0 0 0 1px rgba(255, 255, 255, 0.1) !important;
        transform: translateY(-1px) !important;
    }
    
    button[kind="primary"]:active,
    button[data-testid="baseButton-primary"]:active,
    .stButton > button[kind="primary"]:active,
    .stButton > button[data-baseweb="button"][kind="primary"]:active,
    .stButton button[data-testid="baseButton-primary"]:active {
        background: linear-gradient(180deg, #e9ecef 0%, #dee2e6 100%) !important;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1) !important;
        transform: translateY(0px) !important;
    }
    
    button[kind="primary"]:focus,
    button[data-testid="baseButton-primary"]:focus,
    .stButton > button[kind="primary"]:focus,
    .stButton > button[data-baseweb="button"][kind="primary"]:focus,
    .stButton button[data-testid="baseButton-primary"]:focus {
        box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.3), 0 1px 3px rgba(0, 0, 0, 0.08) !important;
        outline: none !important;
    }
    
    /* Additional primary button overrides for any missed selectors */
    .stButton button[style*="background-color"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%) !important;
        color: var(--apple-text-primary) !important;
        border: 1px solid var(--apple-border) !important;
        border-radius: var(--apple-radius-lg) !important;
        padding: 12px 32px !important;
        font-weight: 500 !important;
        font-size: 17px !important;
        letter-spacing: -0.022em !important;
    }
    
    /* Apple-style inputs */
    .stSelectbox > div > div {
        background: var(--apple-card-bg);
        border: 1px solid var(--apple-border);
        border-radius: var(--apple-radius);
        transition: all 0.15s ease;
        font-size: 17px;
        min-height: 44px;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--apple-blue);
        box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.25);
    }
    
    .stNumberInput > div > div > input {
        background: var(--apple-card-bg);
        border: 1px solid var(--apple-border);
        border-radius: var(--apple-radius);
        transition: all 0.15s ease;
        font-size: 17px;
        font-feature-settings: 'tnum';
        min-height: 44px;
        padding: 8px 12px;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: var(--apple-blue);
        box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.25);
        outline: none;
    }
    
    .stMultiSelect > div > div {
        background: var(--apple-card-bg);
        border: 1px solid var(--apple-border);
        border-radius: var(--apple-radius);
        font-size: 17px;
        min-height: 44px;
    }
    
    .stDateInput > div > div > input {
        background: var(--apple-card-bg);
        border: 1px solid var(--apple-border);
        border-radius: var(--apple-radius);
        font-size: 17px;
        min-height: 44px;
        padding: 8px 12px;
    }
    
    .stTextInput > div > div > input {
        background: var(--apple-card-bg);
        border: 1px solid var(--apple-border);
        border-radius: var(--apple-radius);
        font-size: 17px;
        min-height: 44px;
        padding: 8px 12px;
        color: var(--apple-text-primary);
        transition: all 0.15s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--apple-blue);
        box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.25);
        outline: none;
    }
    
    /* Ensure password input fields match other inputs */
    .stTextInput > div > div > input[type="password"] {
        background: var(--apple-card-bg);
        border: 1px solid var(--apple-border);
        border-radius: var(--apple-radius);
        font-size: 17px;
        min-height: 44px;
        padding: 8px 12px;
        color: var(--apple-text-primary);
        transition: all 0.15s ease;
    }
    
    .stTextInput > div > div > input[type="password"]:focus {
        border-color: var(--apple-blue);
        box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.25);
        outline: none;
    }
    
    /* Remove inner container styling for text inputs */
    .stTextInput > div > div {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    
    /* Make the input fill the entire container and remove inner borders */
    .stTextInput > div > div > input,
    .stTextInput > div > div > input[type="password"] {
        width: 100% !important;
        box-sizing: border-box !important;
        border-radius: var(--apple-radius) !important;
        border: none !important;
        background: var(--apple-card-bg) !important;
        color: var(--apple-text-primary) !important;
        font-size: 17px !important;
        min-height: 44px !important;
        padding: 8px 12px !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextInput > div > div > input[type="password"]:focus {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* Style multiselect selected items with white background */
    .stMultiSelect div[data-baseweb="tag"],
    .stMultiSelect > div > div > div[data-baseweb="tag"],
    div[data-baseweb="tag"] {
        background-color: white !important;
        background: white !important;
        color: #050f19 !important;
        border: 1px solid var(--apple-border) !important;
        border-radius: var(--apple-radius) !important;
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
    
    /* Remove the red selection border that appears on focus */
    .stMultiSelect div[data-baseweb="select"] {
        border: none !important;
    }
    
    .stMultiSelect div[data-baseweb="select"]:focus-within {
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Clean, minimal table styling */
    .stDataFrame {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        overflow: visible;
        margin: 16px 0;
    }
    
    .stDataFrame table {
        background: transparent !important;
        color: var(--apple-text-primary) !important;
        font-size: 15px !important;
        border-collapse: collapse !important;
        border-spacing: 0 !important;
        width: 100% !important;
        border: none !important;
    }
    
    .stDataFrame th {
        background: transparent !important;
        color: var(--apple-text-secondary) !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border: none !important;
        padding: 16px 20px !important;
        text-align: left !important;
    }
    
    .stDataFrame td {
        background: transparent !important;
        color: var(--apple-text-primary) !important;
        border: none !important;
        padding: 14px 20px !important;
        font-feature-settings: 'tnum';
        font-weight: 400 !important;
        vertical-align: middle !important;
    }
    
    /* Simple alternating row backgrounds */
    .stDataFrame tbody tr:nth-child(even) td {
        background: rgba(248, 249, 250, 0.3) !important;
    }
    
    .stDataFrame tbody tr:nth-child(odd) td {
        background: transparent !important;
    }
    
    /* Subtle hover effect */
    .stDataFrame tbody tr:hover td {
        background: rgba(0, 122, 255, 0.05) !important;
    }
    
    /* Enhanced styling for first column */
    .stDataFrame td:first-child {
        font-weight: 500 !important;
        color: var(--apple-text-primary) !important;
    }
    
    /* Right-align numerical columns */
    .stDataFrame td:not(:first-child) {
        text-align: right !important;
        font-variant-numeric: tabular-nums !important;
    }
    
    /* Financial data styling */
    .positive-value {
        color: var(--apple-green) !important;
        font-weight: 600 !important;
    }
    
    .negative-value {
        color: var(--apple-red) !important;
        font-weight: 600 !important;
    }
    
    /* Apple-style metric cards */
    .stMetric {
        background: var(--apple-card-bg);
        padding: 20px;
        border-radius: var(--apple-radius-lg);
        box-shadow: var(--apple-shadow-card);
        border: 1px solid var(--apple-border);
        transition: all 0.15s ease;
    }
    
    .stMetric:hover {
        box-shadow: var(--apple-shadow-card);
        transform: translateY(-2px);
    }
    
    .stMetric [data-testid="metric-value"] {
        font-size: 28px !important;
        font-weight: 600 !important;
        color: var(--apple-text-primary) !important;
        font-feature-settings: 'tnum';
        letter-spacing: -0.022em;
    }
    
    .stMetric [data-testid="metric-label"] {
        font-size: 15px !important;
        font-weight: 500 !important;
        color: var(--apple-text-secondary) !important;
        letter-spacing: -0.022em;
        margin-bottom: 4px;
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
    
    
    /* Multiple selectors for sidebar collapse button - covers all cases */
    button[data-testid="collapsedControl"],
    button[data-testid="baseButton-header"],
    .stButton button[title*="sidebar"],
    button[aria-label*="sidebar"],
    button[title*="Sidebar"],
    button[aria-label*="Sidebar"] {
        display: block !important;
        position: fixed !important;
        top: 1rem !important;
        left: 1rem !important;
        z-index: 999999 !important;
        background: var(--cb-blue) !important;
        color: white !important;
        border: none !important;
        border-radius: 50% !important;
        width: 3rem !important;
        height: 3rem !important;
        box-shadow: var(--cb-shadow-hover) !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    button[data-testid="collapsedControl"]:hover,
    button[data-testid="baseButton-header"]:hover,
    .stButton button[title*="sidebar"]:hover,
    button[aria-label*="sidebar"]:hover,
    button[title*="Sidebar"]:hover,
    button[aria-label*="Sidebar"]:hover {
        background: #1347cc !important;
        transform: scale(1.1) !important;
    }
    
    /* Force visibility of any hidden collapse controls */
    .stApp > header button,
    [data-testid="stHeader"] button {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    /* Responsive sidebar - let Streamlit handle collapse naturally */
    [data-testid="stSidebar"] {
        transition: all 0.3s ease !important;
    }
    
    /* Responsive main content that adapts to screen size */
    .main .block-container {
        width: 100% !important;
        max-width: 100% !important;
        padding: clamp(1rem, 3vw, 3rem) !important;
    }
    
    /* Responsive breakpoints for different screen sizes */
    @media screen and (max-width: 768px) {
        .main .block-container {
            padding: 0.5rem !important;
        }
        
        [data-testid="stSidebar"] > div {
            padding: 0.5rem !important;
        }
    }
    
    @media screen and (min-width: 1200px) {
        .main .block-container {
            padding: 2rem 4rem !important;
        }
    }
    
    @media screen and (min-width: 1600px) {
        .main .block-container {
            padding: 3rem 6rem !important;
        }
    }
    
    /* Apple-style headings */
    h1, h2, h3, h4, h5, h6 {
        color: var(--apple-text-primary) !important;
        font-weight: 600 !important;
        letter-spacing: -0.022em !important;
        line-height: 1.2 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    h1 {
        font-size: 48px !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
    }
    
    h2 {
        font-size: 36px !important;
        font-weight: 600 !important;
    }
    
    h3 {
        font-size: 24px !important;
        font-weight: 600 !important;
    }
    
    /* Apple-style paragraphs */
    p {
        color: var(--apple-text-secondary) !important;
        font-size: 17px !important;
        line-height: 1.47 !important;
        letter-spacing: -0.022em !important;
    }
    
    /* Code blocks */
    code {
        background: var(--apple-secondary-bg) !important;
        border: 1px solid var(--apple-border) !important;
        border-radius: var(--apple-radius) !important;
        padding: 2px 6px !important;
        font-size: 15px !important;
    }
    
    /* Transparent plot backgrounds */
    .stPlotlyChart > div > div {
        background: transparent !important;
    }
    
    .stPlotlyChart {
        background: transparent !important;
    }
    
    /* Plotly chart container */
    .js-plotly-plot .plotly {
        background: transparent !important;
    }
    
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
    
    /* Chart paper background */
    .plotly .main-svg .bg {
        fill: transparent !important;
    }
</style>

<script>
// Ensure sidebar toggle button is always available
function ensureSidebarToggle() {
    // Check if sidebar is collapsed
    const sidebar = document.querySelector('[data-testid="stSidebar"]');
    const isSidebarCollapsed = !sidebar || sidebar.style.display === 'none' || 
                              sidebar.offsetWidth === 0 || sidebar.classList.contains('collapsed');
    
    if (isSidebarCollapsed) {
        // Remove any existing custom toggle
        const existingToggle = document.getElementById('custom-sidebar-toggle');
        if (existingToggle) existingToggle.remove();
        
        // Create custom toggle button
        const toggleButton = document.createElement('button');
        toggleButton.id = 'custom-sidebar-toggle';
        toggleButton.innerHTML = '☰';
        toggleButton.style.cssText = `
            position: fixed !important;
            top: 1rem !important;
            left: 1rem !important;
            z-index: 999999 !important;
            background: #1652f0 !important;
            color: white !important;
            border: none !important;
            border-radius: 50% !important;
            width: 3rem !important;
            height: 3rem !important;
            font-size: 1.2rem !important;
            cursor: pointer !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        `;
        
        toggleButton.onclick = function() {
            // Try multiple ways to show sidebar
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (sidebar) {
                sidebar.style.display = 'block';
                sidebar.style.width = '';
                sidebar.classList.remove('collapsed');
            }
            
            // Try clicking the original toggle if it exists
            const originalToggle = document.querySelector('button[data-testid="collapsedControl"]') ||
                                 document.querySelector('button[data-testid="baseButton-header"]');
            if (originalToggle) {
                originalToggle.click();
            }
            
            // Force Streamlit rerun to refresh sidebar state
            window.parent.postMessage({type: 'streamlit:componentReady'}, '*');
        };
        
        document.body.appendChild(toggleButton);
    } else {
        // Remove custom toggle if sidebar is visible
        const customToggle = document.getElementById('custom-sidebar-toggle');
        if (customToggle) customToggle.remove();
    }
}

// Run on load and monitor for changes
document.addEventListener('DOMContentLoaded', ensureSidebarToggle);
setInterval(ensureSidebarToggle, 500);

// Also monitor for Streamlit updates
const observer = new MutationObserver(ensureSidebarToggle);
observer.observe(document.body, { childList: true, subtree: true });
</script>
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
        background: var(--cb-card-bg) !important;
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
    
    .stTextInput > div > div > input {
        background: var(--cb-card-bg);
        border: 1px solid var(--cb-border);
        border-radius: var(--cb-radius);
        font-size: 0.875rem;
        color: var(--cb-text-primary);
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
        border: none;
        overflow: hidden;
    }
    
    .stDataFrame table {
        background: transparent !important;
        color: var(--cb-text-primary) !important;
        font-size: 0.875rem !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        border-radius: inherit !important;
    }
    
    .stDataFrame th {
        background: var(--cb-secondary-bg) !important;
        color: var(--cb-text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: none !important;
        padding: 1rem 0.75rem !important;
    }
    
    .stDataFrame td {
        background: transparent !important;
        color: var(--cb-text-primary) !important;
        border-bottom: none !important;
        padding: 1rem 0.75rem !important;
        font-feature-settings: 'tnum';
    }
    
    /* Ensure proper corner rounding for header cells */
    .stDataFrame th:first-child {
        border-top-left-radius: var(--cb-radius-lg) !important;
    }
    
    .stDataFrame th:last-child {
        border-top-right-radius: var(--cb-radius-lg) !important;
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
    
    /* Style multiselect selected items (tags) for dark theme */
    .stMultiSelect div[data-baseweb="tag"],
    .stMultiSelect > div > div > div[data-baseweb="tag"],
    div[data-baseweb="tag"] {
        background-color: var(--cb-card-bg) !important;
        background: var(--cb-card-bg) !important;
        color: var(--cb-text-primary) !important;
        border: 1px solid var(--cb-border) !important;
        border-radius: var(--cb-radius) !important;
    }
    
    .stMultiSelect div[data-baseweb="tag"] span,
    .stMultiSelect > div > div > div[data-baseweb="tag"] span,
    div[data-baseweb="tag"] span {
        color: var(--cb-text-primary) !important;
    }
</style>
"""

# Apply the CSS based on current theme
st.markdown(apply_theme_css(st.session_state.theme), unsafe_allow_html=True)

# Function to fetch live crypto news with WebSearch integration
@st.cache_data(ttl=1800, show_spinner=False)  # Cache for 30 minutes
def fetch_crypto_news_dynamic():
    """Fetch latest crypto news from real RSS feeds organized by source"""
    import xml.etree.ElementTree as ET
    from datetime import datetime
    import re
    import html
    
    # RSS feed URLs
    rss_feeds = [
        {"url": "https://www.coindesk.com/arc/outboundfeeds/rss/", "source": "CoinDesk"},
        {"url": "https://cointelegraph.com/rss", "source": "Cointelegraph"},
        {"url": "https://decrypt.co/feed", "source": "Decrypt"},
        {"url": "https://crypto.news/feed/", "source": "Crypto.news"},
        {"url": "https://cryptopotato.com/feed/", "source": "CryptoPotato"},
        {"url": "https://zycrypto.com/feed/", "source": "ZyCrypto"}
    ]
    
    def get_priority_score(title):
        """Score news based on importance keywords"""
        title_lower = title.lower()
        score = 0
        
        # High priority keywords
        if any(word in title_lower for word in ['bitcoin', 'btc', 'ethereum', 'eth']):
            score += 10
        if any(word in title_lower for word in ['regulation', 'government', 'policy', 'sec', 'fed']):
            score += 8
        if any(word in title_lower for word in ['market', 'price', 'surge', 'crash', 'rally']):
            score += 6
        if any(word in title_lower for word in ['breaking', 'major', 'significant']):
            score += 5
        
        return score
    
    organized_news = {}
    
    for feed_info in rss_feeds:
        try:
            # Fetch RSS feed
            response = requests.get(feed_info["url"], timeout=10)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            # Extract items based on RSS structure
            items = root.findall('.//item')
            
            source_news = []
            for item in items[:10]:  # Get top 10 from each source
                title_elem = item.find('title')
                link_elem = item.find('link')
                pubdate_elem = item.find('pubDate')
                
                if title_elem is not None and link_elem is not None:
                    # Extract title with proper text handling for mixed content
                    if title_elem.text:
                        title = title_elem.text.strip()
                    else:
                        # Handle CDATA or mixed content
                        title = ET.tostring(title_elem, encoding='unicode', method='text').strip()
                    
                    # Decode HTML entities and normalize whitespace
                    title = html.unescape(title)
                    # Strip HTML tags that might cause formatting issues
                    title = re.sub(r'<[^>]+>', '', title)
                    title = re.sub(r'\s+', ' ', title).strip()  # Normalize whitespace
                    
                    # Fallback if title is still empty
                    if not title:
                        title = "No title"
                    link = link_elem.text.strip() if link_elem.text else "#"
                    
                    # Extract time from pubDate if available
                    time_ago = "now"
                    if pubdate_elem is not None and pubdate_elem.text:
                        try:
                            # Parse pubDate and calculate time ago
                            from dateutil import parser
                            pub_date = parser.parse(pubdate_elem.text)
                            now = datetime.now(pub_date.tzinfo)
                            diff = now - pub_date
                            
                            if diff.days > 0:
                                time_ago = f"{diff.days}d"
                            elif diff.seconds > 3600:
                                time_ago = f"{diff.seconds // 3600}h"
                            else:
                                time_ago = f"{diff.seconds // 60}m"
                        except:
                            time_ago = "now"
                    
                    source_news.append({
                        "title": title,
                        "url": link,
                        "time_ago": time_ago,
                        "priority": get_priority_score(title)
                    })
            
            # Sort by priority and take top 4
            source_news.sort(key=lambda x: x['priority'], reverse=True)
            organized_news[feed_info["source"]] = source_news[:4]
                    
        except Exception as e:
            print(f"Error fetching {feed_info['source']}: {e}")
            organized_news[feed_info["source"]] = []
            continue
    
    return organized_news

# Initialize enhanced crypto loader (with hybrid data fetching)
@st.cache_resource
def get_data_manager():
    # Use parent directory for data path since frontend is in subdirectory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(parent_dir, "data")
    return EnhancedCryptoLoader(data_dir=data_dir)

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
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="font-size: 56px; font-weight: 700; letter-spacing: -0.025em; color: var(--apple-text-primary); margin-bottom: 0.5rem;">
        Crypto Portfolio
    </h1>
    <p style="font-size: 24px; font-weight: 300; color: var(--apple-text-secondary); margin-top: 0;">
        Intelligent cryptocurrency portfolio optimization and analysis.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("""
<h2 style="font-size: 28px; font-weight: 600; letter-spacing: -0.022em; color: var(--apple-text-primary); margin-bottom: 1.5rem;">
    Portfolio
</h2>
""", unsafe_allow_html=True)

# Theme settings removed - using light mode only

# Mode selection
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Market Insights", "Technical Analysis", "Elliott Wave Analysis", "Portfolio Analysis & Backtest", "Portfolio Optimization", "ML Predictions", "AI Investment Advisor"]
)

# Common parameters
st.sidebar.subheader("Time Period")
end_date = st.sidebar.date_input("End Date", datetime.now() - timedelta(days=1))
start_date = st.sidebar.date_input("Start Date", end_date - timedelta(days=365))

# Load all available cryptocurrencies from the enhanced database
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_available_cryptocurrencies():
    """Load all available cryptocurrencies from the enhanced database"""
    try:
        # Use parent directory for data path since frontend is in subdirectory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(parent_dir, "data")
        loader = EnhancedCryptoLoader(data_dir=data_dir)
        return loader.get_available_symbols()
    except Exception as e:
        st.error(f"Error loading cryptocurrency database: {e}")
        # Fallback to a smaller list if database is not available
        return [
            'BTC-USD', 'ETH-USD', 'XRP-USD', 'BNB-USD', 'SOL-USD',
            'DOGE-USD', 'ADA-USD', 'TRX-USD', 'SHIB-USD', 'AVAX-USD',
            'LINK-USD', 'DOT-USD', 'UNI-USD', 'AAVE-USD', 'MATIC-USD',
            'NEAR-USD', 'ICP-USD', 'APT-USD', 'SUI-USD', 'ATOM-USD'
        ]

crypto_symbols = load_available_cryptocurrencies()

# Show number of available cryptocurrencies
st.sidebar.write(f"**{len(crypto_symbols)} cryptocurrencies available**")
st.sidebar.write("*From extended database with 5+ years of data*")

selected_symbols = st.sidebar.multiselect(
    "Select Cryptocurrencies",
    crypto_symbols,
    default=['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD'],
    help=f"Choose from {len(crypto_symbols)} cryptocurrencies with 5+ years of historical data"
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
        # Use hybrid data fetching (local database + real-time API)
        # start_date and end_date are already strings in 'YYYY-MM-DD' format
        price_data = data_manager.get_hybrid_data(symbols, start_date, end_date)
        
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
        
        # Show brief success message with data source info
        data_info = f"Loaded {price_data.shape[0]} days × {price_data.shape[1]} symbols"
        loading_placeholder.markdown(f"""
        <div style='position: fixed; top: 70px; right: 20px; background: #f0f9ff; padding: 8px 12px; 
                    border-radius: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); z-index: 999;
                    border: 1px solid #00d395;'>
            <span style='font-size: 12px; color: #00d395;'>{data_info} (Hybrid: Local + API)</span>
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

if st.sidebar.button("Refresh News", help="Fetch latest crypto news headlines"):
    st.cache_data.clear()
    st.rerun()

# Performance tip
st.sidebar.markdown("**Tip**: Data is automatically cached. Only fetches when symbols/dates change!")

# Main content area
if mode == "Market Insights":
    st.header("Market Analysis & Insights")
    
    # Market News Section
    st.subheader("Market News")
    
    # Fetch live news organized by source
    with st.spinner("Loading latest crypto news..."):
        try:
            organized_news = fetch_crypto_news_dynamic()
        except Exception:
            organized_news = {}
    
    
    # Display news organized by source
    if organized_news:
        for source, articles in organized_news.items():
            if articles:  # Only show sources with articles
                st.markdown(f"**{source}** {articles[0]['time_ago']}")
                st.markdown("")
                
                for article in articles:
                    title = article.get('title', 'No title')
                    url = article.get('url', '#')
                    time_ago = article.get('time_ago', 'now')
                    
                    # Display each news item with source and time
                    # Escape markdown characters in title to prevent formatting issues
                    escaped_title = title.replace('_', r'\_').replace('*', r'\*').replace('[', r'\[').replace(']', r'\]').replace('$', r'\$')
                    st.markdown(f"[{escaped_title}]({url})")
                
                st.markdown("")
    else:
        st.markdown("No news available at this time.")
    
    # News Sources
    st.markdown("---")
    st.markdown("### News Sources")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Major News Outlets**
        - [CoinDesk](https://www.coindesk.com/)
        - [Cointelegraph](https://cointelegraph.com/)
        - [The Block](https://www.theblock.co/)
        - [Decrypt](https://decrypt.co/)
        - [CryptoSlate](https://cryptoslate.com/)
        - [CNBC Crypto](https://www.cnbc.com/cryptoworld/)
        - [Bloomberg](https://www.bloomberg.com/crypto)
        """)
    
    with col2:
        st.markdown("""
        **Analysis & Research**
        - [Messari](https://messari.io/news)
        - [Glassnode](https://glassnode.com/insights)
        - [IntoTheBlock](https://intotheblock.com/)
        - [CryptoQuant](https://cryptoquant.com/)
        - [Santiment](https://santiment.net/)
        - [TradingView](https://www.tradingview.com/ideas/)
        """)
    
    with col3:
        st.markdown("""
        **Traditional Media**
        - [Reuters](https://www.reuters.com/markets/cryptocurrency/)
        - [MarketWatch](https://www.marketwatch.com/investing/cryptocurrency)
        - [Yahoo Finance](https://finance.yahoo.com/crypto/)
        - [Financial Times](https://www.ft.com/cryptocurrencies)
        - [Wall Street Journal](https://www.wsj.com/news/types/cryptocurrency)
        """)
    
    st.markdown("---")
    
    # Get cached data
    price_data, returns = get_cached_data(
        selected_symbols, 
        start_date.strftime('%Y-%m-%d'), 
        end_date.strftime('%Y-%m-%d')
    )
    
    
    # Price Performance Section
    st.subheader("Price Performance")
    
    # View mode selection only
    price_view_mode = st.selectbox("View Mode", ["Normalized", "Raw Price", "Log Scale"])
    
    # Fixed chart height
    chart_height_px = 500
    
    if price_view_mode == "Normalized":
        chart_data = (price_data / price_data.iloc[0] * 100)
        y_title = "Normalized Price (Start = 100)"
        log_y = False
    elif price_view_mode == "Log Scale":
        chart_data = price_data
        y_title = "Price (USD) - Log Scale"
        log_y = True
    else:
        chart_data = price_data
        y_title = "Price (USD)"
        log_y = False
    
    fig = go.Figure()
    for symbol in selected_symbols:
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data[symbol],
            mode='lines',
            name=symbol.replace('-USD', ''),
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=f"Price Performance - {price_view_mode}",
        xaxis_title="Date",
        yaxis_title=y_title,
        yaxis_type="log" if log_y else "linear",
        hovermode='x unified',
        height=chart_height_px,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    # Make plot transparent for better integration
    fig = make_plot_transparent(fig)
    st.plotly_chart(fig, use_container_width=True)
    
    # Market Correlation Matrix
    st.subheader("Asset Correlation Matrix")
    
    if len(selected_symbols) > 1:
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Create correlation heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[symbol.replace('-USD', '') for symbol in corr_matrix.columns],
            y=[symbol.replace('-USD', '') for symbol in corr_matrix.index],
            colorscale=[
                [0.0, '#2166ac'],    # Dark blue for -1 (inverse correlation)
                [0.25, '#5aae61'],   # Green for -0.5
                [0.5, '#f7f7f7'],    # White for 0 (no correlation)
                [0.75, '#fdbf6f'],   # Orange for 0.5
                [1.0, '#d73027']     # Red for 1 (perfect correlation)
            ],
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 12, "color": "black"},
            hoverongaps=False,
            hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>',
            showscale=True,
            colorbar=dict(
                title="Correlation",
                thickness=15,
                len=0.7
            )
        ))
        
        fig_corr.update_layout(
            title="Asset Return Correlations (1 = Perfect Correlation, -1 = Inverse)",
            height=min(400, len(selected_symbols) * 50 + 200),
            xaxis_title="",
            yaxis_title=""
        )
        
        # Make plot transparent for better integration
        fig_corr = make_plot_transparent(fig_corr)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Correlation analysis requires at least 2 assets. Please select more cryptocurrencies in the sidebar.")
        
        # Correlation insights
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        st.info(f"Portfolio Diversification: Average correlation is {avg_corr:.2f}. Lower correlation (closer to 0) indicates better diversification.")
    
    # Detailed Asset Analysis
    st.subheader("Detailed Asset Analysis")
    
    # Asset selection for detailed analysis
    selected_coin = st.selectbox(
        "Select asset for detailed analysis:",
        options=selected_symbols,
        format_func=lambda x: f"{x.replace('-USD', '')} - ${price_data[x].iloc[-1]:,.2f}"
    )
    
    if selected_coin:
        coin_name = selected_coin.replace('-USD', '')
        coin_data = price_data[selected_coin].dropna()
        coin_returns = returns[selected_coin].dropna()
        
        # Calculate detailed metrics
        current_price = coin_data.iloc[-1]
        start_price = coin_data.iloc[0]
        total_return = (current_price / start_price - 1)
        
        # Time-based returns
        returns_1d = coin_returns.iloc[-1] if len(coin_returns) > 0 else 0
        returns_7d = (current_price / coin_data.iloc[-8] - 1) if len(coin_data) >= 8 else 0
        returns_30d = (current_price / coin_data.iloc[-31] - 1) if len(coin_data) >= 31 else 0
        returns_90d = (current_price / coin_data.iloc[-91] - 1) if len(coin_data) >= 91 else 0
        
        # Risk metrics
        volatility_daily = coin_returns.std()
        volatility_annual = volatility_daily * np.sqrt(252)
        sharpe_ratio = (coin_returns.mean() / coin_returns.std() * np.sqrt(252)) if coin_returns.std() > 0 else 0
        
        # Drawdown analysis
        rolling_max = coin_data.expanding().max()
        drawdown = (coin_data - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Price statistics
        price_high = coin_data.max()
        price_low = coin_data.min()
        current_from_high = (current_price - price_high) / price_high
        current_from_low = (current_price - price_low) / price_low
        
        # Analysis header
        st.markdown(f"### {coin_name} Analysis")
        st.markdown(f"*Analysis Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}*")
        
        # Key metrics
        st.markdown("#### Key Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${current_price:,.2f}")
            st.metric("1-Day Return", f"{returns_1d:.2%}")
            st.metric("7-Day Return", f"{returns_7d:.2%}")
        
        with col2:
            st.metric("30-Day Return", f"{returns_30d:.2%}")
            st.metric("90-Day Return", f"{returns_90d:.2%}")
            st.metric("Total Return", f"{total_return:.2%}")
        
        with col3:
            st.metric("Annual Volatility", f"{volatility_annual:.1%}")
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            st.metric("Max Drawdown", f"{max_drawdown:.1%}")
        
        with col4:
            st.metric("Period High", f"${price_high:,.2f}")
            st.metric("Period Low", f"${price_low:,.2f}")
            st.metric("From High", f"{current_from_high:.1%}")
        
        # Risk Assessment Summary
        st.markdown("#### Risk Assessment")
        
        # Risk level determination
        if volatility_annual > 0.8:
            risk_level = "Very High Risk"
            risk_color = "#e74c3c"
        elif volatility_annual > 0.6:
            risk_level = "High Risk"
            risk_color = "#f39c12"
        elif volatility_annual > 0.4:
            risk_level = "Medium Risk"
            risk_color = "#f1c40f"
        else:
            risk_level = "Low Risk"
            risk_color = "#27ae60"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Risk Level:** {risk_level}")
        with col2:
            performance_rating = "Strong" if total_return > 0.2 else "Moderate" if total_return > 0 else "Weak"
            st.markdown(f"**Performance:** {performance_rating} ({total_return:.1%})")
        with col3:
            trend_direction = "Uptrend" if returns_30d > 0.05 else "Downtrend" if returns_30d < -0.05 else "Sideways"
            st.markdown(f"**30D Trend:** {trend_direction}")
        
        # Analysis tabs
        tab1, tab2 = st.tabs(["Distribution", "Risk Metrics"])
        
        with tab1:
            # Price distribution analysis
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Calculate statistics
                price_mean = coin_data.mean()
                price_std = coin_data.std()
                price_min = coin_data.min()
                price_max = coin_data.max()
                price_median = coin_data.median()
                price_range = price_max - price_min
                price_skewness = coin_data.skew()
                price_kurtosis = coin_data.kurtosis()
                price_25 = coin_data.quantile(0.25)
                price_75 = coin_data.quantile(0.75)
                current_price = coin_data.iloc[-1]
                cv = (price_std / price_mean) * 100
                distance_from_mean = abs(current_price - price_mean) / price_std
                percentile_rank = (coin_data <= current_price).mean() * 100
                
                # Compact key metrics with shorter labels
                st.markdown("### Key Metrics")
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Price", f"${current_price:.2f}")
                    st.metric("Avg", f"${price_mean:.2f}")
                    st.metric("Range", f"${price_range:.0f}")
                with metric_col2:
                    st.metric("Mid", f"${price_median:.2f}")
                    st.metric("StdDev", f"${price_std:.0f}")
                    st.metric("CV%", f"{cv:.1f}")
                
                # Risk assessment without emojis
                if distance_from_mean > 2:
                    risk_text = "High deviation"
                elif distance_from_mean > 1:
                    risk_text = "Moderate deviation"
                else:
                    risk_text = "Normal range"
                
                st.markdown("### Assessment")
                st.markdown(f"""
                **Position**: {percentile_rank:.0f}th percentile  
                **Status**: {risk_text} ({distance_from_mean:.1f}σ)
                """)
            
            with col2:
                # Price histogram with overlays
                fig_hist = go.Figure()
                
                # Main histogram
                fig_hist.add_trace(go.Histogram(
                    x=coin_data,
                    nbinsx=50,
                    name=f"{coin_name} Price Distribution",
                    opacity=0.7,
                    marker_color='lightblue'
                ))
                
                # Add reference lines as traces for legend
                price_mean = coin_data.mean()
                price_median = coin_data.median()
                current_price = coin_data.iloc[-1]
                
                # Add mean line
                fig_hist.add_trace(go.Scatter(
                    x=[price_mean, price_mean],
                    y=[0, 1],
                    mode='lines',
                    name='Mean',
                    line=dict(color='red', dash='dash', width=2),
                    yaxis='y2',
                    showlegend=True
                ))
                
                # Add median line
                fig_hist.add_trace(go.Scatter(
                    x=[price_median, price_median],
                    y=[0, 1],
                    mode='lines',
                    name='Median',
                    line=dict(color='green', dash='dash', width=2),
                    yaxis='y2',
                    showlegend=True
                ))
                
                # Add current price line
                fig_hist.add_trace(go.Scatter(
                    x=[current_price, current_price],
                    y=[0, 1],
                    mode='lines',
                    name='Current',
                    line=dict(color='orange', dash='solid', width=2),
                    yaxis='y2',
                    showlegend=True
                ))
                
                fig_hist.update_layout(
                    title=f"{coin_name} Price Distribution Analysis",
                    xaxis_title="Price (USD)",
                    yaxis_title="Frequency",
                    height=450,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.15,
                        xanchor="center",
                        x=0.5,
                        bgcolor='rgba(0,0,0,0)',
                        borderwidth=0
                    ),
                    yaxis2=dict(
                        overlaying='y',
                        side='right',
                        showgrid=False,
                        showticklabels=False,
                        range=[0, 1]
                    )
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Violin plot for price distribution
                fig_violin = go.Figure()
                fig_violin.add_trace(go.Violin(
                    y=coin_data,
                    name=f"{coin_name}",
                    fillcolor='lightcoral',
                    line=dict(color='darkred'),
                    opacity=0.6,
                    points='outliers'
                ))
                fig_violin.update_layout(
                    title=f"{coin_name} Price Distribution (Outlier Detection)",
                    yaxis_title="Price (USD)",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig_violin, use_container_width=True)
        
        with tab2:
            # Returns distribution analysis
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Calculate statistics
                returns_mean = coin_returns.mean()
                returns_std = coin_returns.std()
                returns_min = coin_returns.min()
                returns_max = coin_returns.max()
                returns_median = coin_returns.median()
                returns_range = returns_max - returns_min
                returns_skewness = coin_returns.skew()
                returns_kurtosis = coin_returns.kurtosis()
                returns_25 = coin_returns.quantile(0.25)
                returns_75 = coin_returns.quantile(0.75)
                returns_cv = (returns_std / abs(returns_mean)) * 100 if returns_mean != 0 else 0
                
                # Compact key metrics with shorter labels
                st.markdown("### Key Metrics")
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Mean", f"{returns_mean:.3f}")
                    st.metric("Min", f"{returns_min:.2%}")
                    st.metric("Range", f"{returns_range:.3f}")
                with metric_col2:
                    st.metric("Median", f"{returns_median:.3f}")
                    st.metric("Max", f"{returns_max:.2%}")
                    st.metric("StdDev", f"{returns_std:.3f}")
                
                # Volatility assessment
                annualized_vol = returns_std * (252 ** 0.5)
                if annualized_vol > 1.0:
                    vol_text = "Very high volatility"
                elif annualized_vol > 0.5:
                    vol_text = "High volatility"
                else:
                    vol_text = "Moderate volatility"
                
                # Sharpe ratio (assuming risk-free rate of 0)
                sharpe = returns_mean / returns_std * (252 ** 0.5) if returns_std > 0 else 0
                
                st.markdown("### Assessment")
                st.markdown(f"""
                **Volatility**: {vol_text}  
                **Annualized Vol**: {annualized_vol:.1%}  
                **Sharpe Ratio**: {sharpe:.2f}
                """)
            
            with col2:
                # Returns histogram with overlays
                fig_returns = go.Figure()
                
                # Main histogram
                fig_returns.add_trace(go.Histogram(
                    x=coin_returns,
                    nbinsx=50,
                    name=f"{coin_name} Daily Returns",
                    opacity=0.7,
                    marker_color='lightcoral'
                ))
                
                # Add reference lines as traces for legend
                returns_mean = coin_returns.mean()
                returns_median = coin_returns.median()
                
                # Add mean line
                fig_returns.add_trace(go.Scatter(
                    x=[returns_mean, returns_mean],
                    y=[0, 1],
                    mode='lines',
                    name='Mean',
                    line=dict(color='red', dash='dash', width=2),
                    yaxis='y2',
                    showlegend=True
                ))
                
                # Add median line
                fig_returns.add_trace(go.Scatter(
                    x=[returns_median, returns_median],
                    y=[0, 1],
                    mode='lines',
                    name='Median',
                    line=dict(color='green', dash='dash', width=2),
                    yaxis='y2',
                    showlegend=True
                ))
                
                # Add zero line
                fig_returns.add_trace(go.Scatter(
                    x=[0, 0],
                    y=[0, 1],
                    mode='lines',
                    name='Zero',
                    line=dict(color='gray', dash='solid', width=2),
                    yaxis='y2',
                    showlegend=True
                ))
                
                fig_returns.update_layout(
                    title=f"{coin_name} Returns Distribution Analysis",
                    xaxis_title="Daily Return",
                    yaxis_title="Frequency",
                    height=450,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.15,
                        xanchor="center",
                        x=0.5,
                        bgcolor='rgba(0,0,0,0)',
                        borderwidth=0
                    ),
                    yaxis2=dict(
                        overlaying='y',
                        side='right',
                        showgrid=False,
                        showticklabels=False,
                        range=[0, 1]
                    )
                )
                st.plotly_chart(fig_returns, use_container_width=True)
                
                # Violin plot for returns distribution
                fig_violin_returns = go.Figure()
                fig_violin_returns.add_trace(go.Violin(
                    y=coin_returns,
                    name=f"{coin_name}",
                    fillcolor='lightcoral',
                    line=dict(color='darkred'),
                    opacity=0.6,
                    points='outliers'
                ))
                fig_violin_returns.update_layout(
                    title=f"{coin_name} Returns Distribution (Outlier Detection)",
                    yaxis_title="Daily Return",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig_violin_returns, use_container_width=True)

elif mode == "Technical Analysis":
    st.header("Technical Analysis")
    
    # Only show cryptocurrencies that are selected in the sidebar
    available_coins = [coin for coin in selected_symbols if coin in crypto_symbols]
    
    if not available_coins:
        st.warning("Please select at least one cryptocurrency from the sidebar to perform technical analysis.")
        st.stop()
    
    selected_coin = st.selectbox(
        "Choose a cryptocurrency for analysis",
        options=available_coins,
        key="ta_coin_selection",
        help=f"Select from {len(available_coins)} cryptocurrencies chosen in the sidebar"
    )
    
    # Get data for the selected coin
    try:
        coin_data = data_manager.get_real_data([selected_coin], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if coin_data is not None and not coin_data.empty:
            coin_series = coin_data[selected_coin].dropna()
            
            # Initialize technical analyzer
            ta = TechnicalAnalyzer(coin_series)
            
            # Calculate indicators and analyze signals
            indicators = ta.calculate_all_indicators()
            signals = ta.analyze_signals()
            summary = ta.get_signal_summary()
            
            # Signal Summary with Conflict Explanation
            st.subheader("Signal Summary")
            
            # Check for conflicting signals
            overall_bullish = signals['overall'].get('bullish_signals', 0)
            overall_bearish = signals['overall'].get('bearish_signals', 0)
            has_conflicts = overall_bullish > 0 and overall_bearish > 0
            
            # Show info box if there are conflicting signals
            if has_conflicts:
                conflict_ratio = min(overall_bullish, overall_bearish) / max(overall_bullish, overall_bearish)
                if conflict_ratio > 0.6:  # High conflict
                    st.warning(
                        "**Mixed Signals Detected**: Multiple indicators show conflicting signals. "
                        "This often occurs at market turning points or during consolidation periods. "
                        "Consider waiting for clearer signals or using smaller position sizes."
                    )
                elif conflict_ratio > 0.3:  # Moderate conflict
                    st.info(
                        "**Divergent Indicators**: Some indicators conflict with the overall trend. "
                        "The dominant signal direction is based on weighted indicator importance."
                    )
            
            # Clean metrics layout
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = coin_series.iloc[-1]
                st.metric("Current Price", f"${current_price:.4f}")
            
            with col2:
                overall_signal = signals['overall']['signal'].upper()
                st.metric("Overall Signal", overall_signal)
            
            with col3:
                confidence = signals['overall']['confidence']
                conf_delta = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
                st.metric("Confidence", f"{confidence:.1%}", delta=conf_delta)
            
            with col4:
                strength = signals['overall']['strength']
                st.metric("Signal Strength", f"{strength:.1%}")
            
            # Add Indicator Consensus Chart
            if has_conflicts or overall_bullish + overall_bearish > 3:
                with st.expander("View Indicator Consensus Details", expanded=False):
                    consensus_col1, consensus_col2 = st.columns([2, 1])
                    
                    with consensus_col1:
                        # Create a simple bar chart showing signal distribution
                        consensus_data = {
                            'Bullish Signals': overall_bullish,
                            'Bearish Signals': overall_bearish,
                            'Neutral Signals': signals['overall'].get('total_signals', 0) - overall_bullish - overall_bearish
                        }
                        
                        import plotly.graph_objects as go
                        fig_consensus = go.Figure(data=[
                            go.Bar(
                                x=list(consensus_data.keys()),
                                y=list(consensus_data.values()),
                                marker_color=['green', 'red', 'gray'],
                                text=list(consensus_data.values()),
                                textposition='auto',
                            )
                        ])
                        fig_consensus.update_layout(
                            title="Indicator Signal Distribution",
                            yaxis_title="Number of Indicators",
                            height=250,
                            showlegend=False
                        )
                        st.plotly_chart(fig_consensus, use_container_width=True)
                    
                    with consensus_col2:
                        st.markdown("**Interpretation Guide:**")
                        st.markdown("• **Unanimous**: All indicators agree")
                        st.markdown("• **Strong**: 70%+ indicators agree")  
                        st.markdown("• **Moderate**: 50-70% agreement")
                        st.markdown("• **Weak**: <50% agreement")
                        
                        # Calculate consensus strength
                        total_signals = signals['overall'].get('total_signals', 0)
                        if total_signals > 0:
                            max_signals = max(overall_bullish, overall_bearish)
                            consensus_pct = (max_signals / total_signals) * 100
                            
                            if consensus_pct >= 90:
                                st.success(f"Unanimous consensus ({consensus_pct:.0f}%)")
                            elif consensus_pct >= 70:
                                st.success(f"Strong consensus ({consensus_pct:.0f}%)")
                            elif consensus_pct >= 50:
                                st.warning(f"Moderate consensus ({consensus_pct:.0f}%)")
                            else:
                                st.error(f"Weak consensus ({consensus_pct:.0f}%)")
            
            # Timeframe Analysis with Enhanced Signal Presentation
            st.subheader("Timeframe Analysis")
            
            timeframe_cols = st.columns(3)
            timeframes = ['short_term', 'medium_term', 'long_term']
            timeframe_names = ['Short Term (1-7 days)', 'Medium Term (1-4 weeks)', 'Long Term (1-3 months)']
            
            for i, (tf, name) in enumerate(zip(timeframes, timeframe_names)):
                with timeframe_cols[i]:
                    tf_data = signals[tf]
                    signal = tf_data['signal'].upper()
                    
                    st.markdown(f"**{name}**")
                    
                    # Professional signal display
                    st.metric("Signal", signal)
                    st.metric("Strength", f"{tf_data['strength']:.1%}")
                    st.metric("Confidence", f"{tf_data['confidence']:.1%}")
                    
                    # Enhanced signal breakdown showing conflicts
                    if tf_data.get('bullish_signals', 0) > 0 or tf_data.get('bearish_signals', 0) > 0:
                        st.markdown("**Signal Breakdown:**")
                        bullish_count = tf_data.get('bullish_signals', 0)
                        bearish_count = tf_data.get('bearish_signals', 0)
                        total_count = tf_data.get('total_signals', 0)
                        
                        # Show signal distribution
                        if bullish_count > 0:
                            st.markdown(f"**Bullish**: {bullish_count}/{total_count} indicators")
                        if bearish_count > 0:
                            st.markdown(f"**Bearish**: {bearish_count}/{total_count} indicators")
                        if total_count - bullish_count - bearish_count > 0:
                            st.markdown(f"**Neutral**: {total_count - bullish_count - bearish_count}/{total_count} indicators")
                    
                    # Show key factors with better categorization
                    if tf_data['reasons']:
                        st.markdown("**Key Indicators:**")
                        # Group signals by type
                        bullish_reasons = []
                        bearish_reasons = []
                        neutral_reasons = []
                        
                        for reason in tf_data['reasons']:
                            reason_lower = reason.lower()
                            if 'bullish' in reason_lower or 'oversold' in reason_lower or 'above' in reason_lower:
                                bullish_reasons.append(reason)
                            elif 'bearish' in reason_lower or 'overbought' in reason_lower or 'below' in reason_lower:
                                bearish_reasons.append(reason)
                            else:
                                neutral_reasons.append(reason)
                        
                        # Display grouped reasons with professional indicators
                        if bullish_reasons:
                            for reason in bullish_reasons[:2]:
                                st.markdown(f"• **Bullish**: {reason}")
                        if bearish_reasons:
                            for reason in bearish_reasons[:2]:
                                st.markdown(f"• **Bearish**: {reason}")
                        if neutral_reasons and not (bullish_reasons or bearish_reasons):
                            for reason in neutral_reasons[:2]:
                                st.markdown(f"• **Neutral**: {reason}")
            
            # Risk Metrics
            st.subheader("Risk Metrics")
            
            if 'risk_metrics' in summary and not summary['risk_metrics'].get('insufficient_data', False):
                risk_metrics = summary['risk_metrics']
                
                risk_cols = st.columns(4)
                with risk_cols[0]:
                    st.metric("30-Day Volatility", f"{risk_metrics['volatility_30d']:.1%}")
                with risk_cols[1]:
                    st.metric("7-Day Volatility", f"{risk_metrics['volatility_7d']:.1%}")
                with risk_cols[2]:
                    st.metric("Max Drawdown", f"{risk_metrics['max_drawdown']:.1%}")
                with risk_cols[3]:
                    st.metric("Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.2f}")
            
            # Technical Indicators
            st.subheader("Technical Indicators")
            
            # Reorganized tabs without duplicates
            tab1, tab2, tab3, tab4 = st.tabs([
                "Trend Analysis", 
                "Momentum", 
                "Volatility", 
                "Additional Indicators"
            ])
            
            with tab1:
                # Price and Moving Averages
                fig_trend = go.Figure()
                
                # Price
                fig_trend.add_trace(go.Scatter(
                    x=coin_series.index,
                    y=coin_series,
                    mode='lines',
                    name=f'{selected_coin} Price',
                    line=dict(width=2)
                ))
                
                # Moving averages
                ma_colors = {'sma_7': 'orange', 'sma_21': 'green', 'sma_50': 'red', 'sma_200': 'purple'}
                ma_names = {'sma_7': 'SMA 7', 'sma_21': 'SMA 21', 'sma_50': 'SMA 50', 'sma_200': 'SMA 200'}
                
                for ma_key, color in ma_colors.items():
                    if ma_key in indicators:
                        fig_trend.add_trace(go.Scatter(
                            x=indicators[ma_key].index,
                            y=indicators[ma_key],
                            mode='lines',
                            name=ma_names[ma_key],
                            line=dict(width=1, color=color, dash='dash')
                        ))
                
                # Add key levels from summary
                if 'key_levels' in summary:
                    levels = summary['key_levels']
                    current_price = coin_series.iloc[-1]
                    
                    # Add support and resistance levels as invisible traces for legend
                    if 'support' in levels:
                        fig_trend.add_hline(
                            y=levels['support'], 
                            line_dash="dot", 
                            line_color="green",
                            line_width=2
                        )
                        # Add invisible trace for legend
                        fig_trend.add_trace(go.Scatter(
                            x=[coin_series.index[0]],
                            y=[levels['support']],
                            mode='lines',
                            name=f"Support: ${levels['support']:.4f}",
                            line=dict(color="green", dash="dot", width=2),
                            showlegend=True,
                            visible='legendonly'
                        ))
                    
                    if 'resistance' in levels:
                        fig_trend.add_hline(
                            y=levels['resistance'], 
                            line_dash="dot", 
                            line_color="red",
                            line_width=2
                        )
                        # Add invisible trace for legend
                        fig_trend.add_trace(go.Scatter(
                            x=[coin_series.index[0]],
                            y=[levels['resistance']],
                            mode='lines',
                            name=f"Resistance: ${levels['resistance']:.4f}",
                            line=dict(color="red", dash="dot", width=2),
                            showlegend=True,
                            visible='legendonly'
                        ))
                
                fig_trend.update_layout(
                    title=f"{selected_coin} Price with Moving Averages & Key Levels",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    height=500
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # MACD
                if all(k in indicators for k in ['macd_line', 'macd_signal', 'macd_histogram']):
                    fig_macd = go.Figure()
                    
                    fig_macd.add_trace(go.Scatter(
                        x=indicators['macd_line'].index,
                        y=indicators['macd_line'],
                        mode='lines',
                        name='MACD Line',
                        line=dict(color='blue')
                    ))
                    
                    fig_macd.add_trace(go.Scatter(
                        x=indicators['macd_signal'].index,
                        y=indicators['macd_signal'],
                        mode='lines',
                        name='Signal Line',
                        line=dict(color='red')
                    ))
                    
                    fig_macd.add_trace(go.Bar(
                        x=indicators['macd_histogram'].index,
                        y=indicators['macd_histogram'],
                        name='MACD Histogram',
                        marker_color=['green' if x > 0 else 'red' for x in indicators['macd_histogram']],
                        opacity=0.7
                    ))
                    
                    fig_macd.add_hline(y=0, line_dash="dash", line_color="gray")
                    
                    fig_macd.update_layout(
                        title="MACD",
                        xaxis_title="Date",
                        yaxis_title="MACD",
                        height=400
                    )
                    st.plotly_chart(fig_macd, use_container_width=True)
            
            with tab2:
                # RSI
                if 'rsi' in indicators:
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=indicators['rsi'].index,
                        y=indicators['rsi'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='orange', width=2)
                    ))
                    
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                    fig_rsi.add_hline(y=50, line_dash="solid", line_color="gray", annotation_text="Neutral (50)")
                    
                    fig_rsi.update_layout(
                        title="Relative Strength Index (RSI)",
                        xaxis_title="Date",
                        yaxis_title="RSI",
                        height=400,
                        yaxis=dict(range=[0, 100])
                    )
                    st.plotly_chart(fig_rsi, use_container_width=True)
                
                # Stochastic Oscillator
                if all(k in indicators for k in ['stoch_k', 'stoch_d']):
                    fig_stoch = go.Figure()
                    
                    fig_stoch.add_trace(go.Scatter(
                        x=indicators['stoch_k'].index,
                        y=indicators['stoch_k'],
                        mode='lines',
                        name='%K',
                        line=dict(color='blue')
                    ))
                    
                    fig_stoch.add_trace(go.Scatter(
                        x=indicators['stoch_d'].index,
                        y=indicators['stoch_d'],
                        mode='lines',
                        name='%D',
                        line=dict(color='red')
                    ))
                    
                    fig_stoch.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought (80)")
                    fig_stoch.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold (20)")
                    
                    fig_stoch.update_layout(
                        title="Stochastic Oscillator",
                        xaxis_title="Date",
                        yaxis_title="Stochastic",
                        height=400,
                        yaxis=dict(range=[0, 100])
                    )
                    st.plotly_chart(fig_stoch, use_container_width=True)
                
                # Williams %R
                if 'williams_r' in indicators:
                    fig_williams = go.Figure()
                    fig_williams.add_trace(go.Scatter(
                        x=indicators['williams_r'].index,
                        y=indicators['williams_r'],
                        mode='lines',
                        name='Williams %R',
                        line=dict(color='purple', width=2)
                    ))
                    
                    fig_williams.add_hline(y=-20, line_dash="dash", line_color="red", annotation_text="Overbought (-20)")
                    fig_williams.add_hline(y=-80, line_dash="dash", line_color="green", annotation_text="Oversold (-80)")
                    
                    fig_williams.update_layout(
                        title="Williams %R",
                        xaxis_title="Date",
                        yaxis_title="Williams %R",
                        height=400,
                        yaxis=dict(range=[-100, 0])
                    )
                    st.plotly_chart(fig_williams, use_container_width=True)
            
            with tab3:
                # Bollinger Bands
                if all(k in indicators for k in ['bb_upper', 'bb_middle', 'bb_lower']):
                    fig_bb = go.Figure()
                    
                    # Price
                    fig_bb.add_trace(go.Scatter(
                        x=coin_series.index,
                        y=coin_series,
                        mode='lines',
                        name=f'{selected_coin} Price',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Bollinger Bands
                    fig_bb.add_trace(go.Scatter(
                        x=indicators['bb_upper'].index,
                        y=indicators['bb_upper'],
                        mode='lines',
                        name='Upper Band',
                        line=dict(color='red', width=1),
                        fill=None
                    ))
                    
                    fig_bb.add_trace(go.Scatter(
                        x=indicators['bb_lower'].index,
                        y=indicators['bb_lower'],
                        mode='lines',
                        name='Lower Band',
                        line=dict(color='red', width=1),
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.1)'
                    ))
                    
                    fig_bb.add_trace(go.Scatter(
                        x=indicators['bb_middle'].index,
                        y=indicators['bb_middle'],
                        mode='lines',
                        name='Middle Band (SMA 20)',
                        line=dict(color='green', width=1, dash='dash')
                    ))
                    
                    # Key levels are shown in Trend Analysis tab to avoid duplication
                    
                    fig_bb.update_layout(
                        title=f"{selected_coin} Bollinger Bands",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        height=500
                    )
                    st.plotly_chart(fig_bb, use_container_width=True)
                
                # ATR (Average True Range)
                if 'atr' in indicators:
                    fig_atr = go.Figure()
                    fig_atr.add_trace(go.Scatter(
                        x=indicators['atr'].index,
                        y=indicators['atr'],
                        mode='lines',
                        name='ATR',
                        line=dict(color='brown', width=2)
                    ))
                    
                    fig_atr.update_layout(
                        title="Average True Range (ATR) - Volatility Measure",
                        xaxis_title="Date",
                        yaxis_title="ATR",
                        height=400
                    )
                    st.plotly_chart(fig_atr, use_container_width=True)
            
            with tab4:
                # Additional Indicators
                st.markdown("#### Additional Technical Indicators")
                
                # Commodity Channel Index (CCI)
                if 'cci' in indicators:
                    fig_cci = go.Figure()
                    fig_cci.add_trace(go.Scatter(
                        x=indicators['cci'].index,
                        y=indicators['cci'],
                        mode='lines',
                        name='CCI',
                        line=dict(color='purple', width=2)
                    ))
                    
                    fig_cci.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Overbought (100)")
                    fig_cci.add_hline(y=-100, line_dash="dash", line_color="green", annotation_text="Oversold (-100)")
                    fig_cci.add_hline(y=0, line_dash="solid", line_color="gray")
                    
                    fig_cci.update_layout(
                        title="Commodity Channel Index (CCI)",
                        xaxis_title="Date",
                        yaxis_title="CCI",
                        height=400
                    )
                    st.plotly_chart(fig_cci, use_container_width=True)
                
                # Rate of Change (ROC)
                if 'roc_10' in indicators and 'roc_20' in indicators:
                    fig_roc = go.Figure()
                    
                    fig_roc.add_trace(go.Scatter(
                        x=indicators['roc_10'].index,
                        y=indicators['roc_10'],
                        mode='lines',
                        name='ROC 10-day',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig_roc.add_trace(go.Scatter(
                        x=indicators['roc_20'].index,
                        y=indicators['roc_20'],
                        mode='lines',
                        name='ROC 20-day',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig_roc.add_hline(y=0, line_dash="solid", line_color="gray")
                    
                    fig_roc.update_layout(
                        title="Rate of Change (ROC)",
                        xaxis_title="Date",
                        yaxis_title="ROC (%)",
                        height=400
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
                
                # Momentum indicators
                if 'momentum_10' in indicators and 'momentum_20' in indicators:
                    fig_momentum = go.Figure()
                    
                    fig_momentum.add_trace(go.Scatter(
                        x=indicators['momentum_10'].index,
                        y=indicators['momentum_10'] * 100,  # Convert to percentage
                        mode='lines',
                        name='10-day Momentum',
                        line=dict(color='orange', width=2)
                    ))
                    
                    fig_momentum.add_trace(go.Scatter(
                        x=indicators['momentum_20'].index,
                        y=indicators['momentum_20'] * 100,  # Convert to percentage
                        mode='lines',
                        name='20-day Momentum',
                        line=dict(color='green', width=2)
                    ))
                    
                    fig_momentum.add_hline(y=0, line_dash="solid", line_color="gray")
                    
                    fig_momentum.update_layout(
                        title="Price Momentum",
                        xaxis_title="Date",
                        yaxis_title="Momentum (%)",
                        height=400
                    )
                    st.plotly_chart(fig_momentum, use_container_width=True)
            
            # Key levels are already displayed on the price charts in Trend Analysis tab
            
            # Removed Elliott Wave tab - now has its own section
            
        else:
            st.error("Unable to load data for the selected cryptocurrency. Please try again.")
    
    except Exception as e:
        st.error(f"Error in technical analysis: {str(e)}")
        st.info("Please ensure you have selected a valid cryptocurrency and time period.")

elif mode == "Elliott Wave Analysis":
    st.header("Elliott Wave Analysis")
    st.markdown("*Clean wave identification with support/resistance levels*")
    
    # Only show cryptocurrencies that are selected in the sidebar
    available_coins = [coin for coin in selected_symbols if coin in crypto_symbols]
    
    if not available_coins:
        st.warning("Please select at least one cryptocurrency from the sidebar.")
        st.stop()
    
    # User controls for wave calculation
    col1, col2 = st.columns(2)
    
    with col1:
        selected_coin = st.selectbox(
            "Choose cryptocurrency",
            options=available_coins,
            key="ew_coin_selection"
        )
    
    with col2:
        st.markdown("**Wave Calculation Settings**")
        adaptive_mode = st.toggle("Adaptive Wave Detection", value=True, help="Automatically adjust wave sensitivity based on timeframe characteristics")
    
    # Advanced controls (when adaptive mode is off)
    if not adaptive_mode:
        st.subheader("Manual Wave Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            short_swing = st.slider("Short Term Swing %", 0.1, 2.0, 0.5, 0.1, help="Minimum price move to detect short-term waves")
        with col2:
            medium_swing = st.slider("Medium Term Swing %", 1.0, 5.0, 2.0, 0.1, help="Minimum price move to detect medium-term waves")
        with col3:
            long_swing = st.slider("Long Term Swing %", 2.0, 10.0, 4.0, 0.1, help="Minimum price move to detect long-term waves")
        
        # Additional fine-tuning controls
        with st.expander("Advanced Pattern Recognition Settings"):
            col1, col2 = st.columns(2)
            with col1:
                confidence_threshold = st.slider("Pattern Confidence Threshold", 0.1, 0.9, 0.5, 0.05, 
                                               help="Higher values require more strict wave pattern validation")
                adaptive_labeling = st.checkbox("Adaptive Wave Labeling", True, 
                                               help="Adjust wave labels based on pattern complexity")
            with col2:
                analysis_depth = st.selectbox("Analysis Depth", 
                                            ["Conservative", "Standard", "Aggressive"],
                                            index=1,
                                            help="Conservative: Fewer, higher-confidence patterns. Aggressive: More patterns, lower confidence threshold")
                show_debug = st.checkbox("Show Analysis Details", False, 
                                       help="Display detailed information about wave detection process")
    else:
        # Set default values for adaptive mode
        confidence_threshold = 0.5
        adaptive_labeling = True
        analysis_depth = "Standard"
        show_debug = False
    
    # Get data for the selected coin
    try:
        coin_data = data_manager.get_real_data([selected_coin], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if coin_data is not None and not coin_data.empty:
            coin_series = coin_data[selected_coin].dropna()
            
            if len(coin_series) < 50:
                st.warning(f"Need at least 50 days of data. Current: {len(coin_series)} days")
                st.stop()
            
            # Adaptive timeframe configuration based on user settings and data characteristics
            if adaptive_mode:
                # Calculate adaptive swing thresholds based on volatility
                coin_volatility = coin_series.pct_change().std() * 100  # Daily volatility as percentage
                
                # Adaptive scaling based on volatility
                volatility_multiplier = max(0.5, min(2.0, coin_volatility / 3.0))  # Scale between 0.5x and 2x
                
                timeframes = {
                    'Short Term': {
                        'swing_pct': 0.3 * volatility_multiplier, 
                        'days_back': min(30, max(14, len(coin_series) // 10)),  # Adaptive period
                        'period': f'Last {min(30, max(14, len(coin_series) // 10))} Days'
                    },
                    'Medium Term': {
                        'swing_pct': 1.5 * volatility_multiplier, 
                        'days_back': min(90, max(30, len(coin_series) // 4)),
                        'period': f'Last {min(90, max(30, len(coin_series) // 4))} Days'
                    }, 
                    'Long Term': {
                        'swing_pct': 3.0 * volatility_multiplier, 
                        'days_back': None, 
                        'period': f'Full Period ({len(coin_series)} days)'
                    }
                }
                
                st.info(f"Adaptive mode: Detected {coin_volatility:.1f}% daily volatility. Swing thresholds scaled by {volatility_multiplier:.1f}x")
            else:
                # Use manual user-defined settings
                timeframes = {
                    'Short Term': {'swing_pct': short_swing, 'days_back': 30, 'period': 'Last 30 Days'},
                    'Medium Term': {'swing_pct': medium_swing, 'days_back': 90, 'period': 'Last 3 Months'}, 
                    'Long Term': {'swing_pct': long_swing, 'days_back': None, 'period': 'Full Period'}
                }
            
            st.subheader("Wave Analysis")
            
            # Create tabs for each timeframe
            tab_short, tab_medium, tab_long = st.tabs(["Short Term", "Medium Term", "Long Term"])
            
            for tab, (timeframe_name, config) in zip([tab_short, tab_medium, tab_long], timeframes.items()):
                with tab:
                    try:
                        # Get timeframe-specific data
                        if config['days_back'] is not None:
                            # Use only recent data for short/medium term
                            timeframe_series = coin_series.tail(config['days_back'])
                        else:
                            # Use full data for long term
                            timeframe_series = coin_series
                        
                        # Show timeframe context with Elliott Wave degree
                        degree_map = {
                            'Short Term': 'Minor Degree (waves: i,ii,iii,iv,v or a,b,c)',
                            'Medium Term': 'Intermediate Degree (waves: i,ii,iii,iv,v or a,b,c)',
                            'Long Term': 'Primary Degree (waves: i,ii,iii,iv,v or a,b,c)'
                        }
                        st.caption(f"{config['period']} | {len(timeframe_series)} days | {config['swing_pct']:.1f}% swings")
                        st.caption(f"**{degree_map[timeframe_name]}**")
                        
                        # Analyze for this timeframe
                        analyzer = ElliottWaveAnalyzer(timeframe_series, config['swing_pct'])
                        analysis = analyzer.get_clean_wave_analysis()
                        
                        # Clean wave analysis summary
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Wave Pattern", f"{analysis['wave_count']} {analysis['pattern_type']}")
                        with col2:
                            st.metric("Direction", analysis['wave_direction'])
                        with col3:
                            confidence = analysis['pattern_confidence']
                            conf_text = f"{confidence:.0%}" if confidence > 0 else "Low"
                            conf_color = "normal" if confidence > 0.5 else "inverse"
                            st.metric("Confidence", conf_text)
                        
                        # Current position with better formatting
                        st.info(f"**Current Wave Position:** {analysis['current_position']}")
                        
                        # Create clean chart focused on timeframe period
                        fig = go.Figure()
                        
                        # Add price line for the specific timeframe
                        fig.add_trace(go.Scatter(
                            x=timeframe_series.index,
                            y=timeframe_series,
                            mode='lines',
                            name=f'{selected_coin}',
                            line=dict(color='#2E86AB', width=3)
                        ))
                        
                        # Add wave points
                        wave_points = analyzer.wave_points[-8:]  # Last 8 points
                        if len(wave_points) >= 3:
                            wave_dates = [p.date for p in wave_points]
                            wave_prices = [p.price for p in wave_points]
                            wave_types = [p.point_type for p in wave_points]
                            
                            # Add wave markers
                            fig.add_trace(go.Scatter(
                                x=wave_dates,
                                y=wave_prices,
                                mode='markers+lines',
                                name='Wave Points',
                                marker=dict(
                                    size=10,
                                    color='red',
                                    symbol=['triangle-down' if t == 'high' else 'triangle-up' for t in wave_types]
                                ),
                                line=dict(color='orange', width=1, dash='dot')
                            ))
                            
                            # Add wave labels with high contrast visibility
                            labels = analysis['wave_labels']
                            for i, (point, label) in enumerate(zip(wave_points, labels)):
                                if i < len(labels):
                                    # Alternate label positions to reduce overlap
                                    y_offset = 25 if point.point_type == 'high' else -35
                                    
                                    fig.add_annotation(
                                        x=point.date,
                                        y=point.price,
                                        text=f"<b>{label}</b>",
                                        showarrow=True,
                                        arrowhead=2,
                                        arrowsize=1.5,
                                        arrowcolor="black",
                                        ax=0,
                                        ay=y_offset,
                                        font=dict(size=16, color="black", family="Arial Black"),
                                        bgcolor="yellow",
                                        bordercolor="black",
                                        borderwidth=2,
                                        opacity=1.0
                                    )
                        
                        # Add clean support/resistance levels (timeframe-specific)
                        key_levels = analysis['key_levels']
                        current_price = timeframe_series.iloc[-1]
                        
                        # Show clean support/resistance lines without price labels (shown in summary below)
                        if key_levels['support']:
                            closest_support = key_levels['support'][0]
                            fig.add_hline(
                                y=closest_support['price'],
                                line_dash="dash",
                                line_color="green",
                                line_width=2,
                                opacity=0.7,
                                annotation_text="Support",
                                annotation_position="bottom left",
                                annotation_font_size=10,
                                annotation_font_color="green"
                            )
                        
                        if key_levels['resistance']:
                            closest_resistance = key_levels['resistance'][0]
                            fig.add_hline(
                                y=closest_resistance['price'],
                                line_dash="dash",
                                line_color="red",
                                line_width=2,
                                opacity=0.7,
                                annotation_text="Resistance",
                                annotation_position="top left",
                                annotation_font_size=10,
                                annotation_font_color="red"
                            )
                        
                        fig.update_layout(
                            title=f"{timeframe_name} Analysis - {selected_coin} ({config['period']})",
                            xaxis_title="Date",
                            yaxis_title="Price (USD)",
                            height=450,
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                yanchor="top",
                                y=-0.15,
                                xanchor="center",
                                x=0.5
                            ),
                            margin=dict(l=50, r=50, t=50, b=80),
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Key levels summary
                        st.subheader("Key Trading Levels")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Current Price", f"${current_price:.0f}")
                        
                        with col2:
                            if key_levels['support']:
                                support_price = key_levels['support'][0]['price']
                                support_distance = ((support_price - current_price) / current_price) * 100
                                st.metric("Nearest Support", f"${support_price:.0f}", delta=f"{support_distance:+.1f}%")
                            else:
                                st.metric("Nearest Support", "None detected")
                        
                        with col3:
                            if key_levels['resistance']:
                                resistance_price = key_levels['resistance'][0]['price']
                                resistance_distance = ((resistance_price - current_price) / current_price) * 100
                                st.metric("Nearest Resistance", f"${resistance_price:.0f}", delta=f"{resistance_distance:+.1f}%")
                            else:
                                st.metric("Nearest Resistance", "None detected")
                    
                    except Exception as e:
                        st.error(f"Error in {timeframe_name} analysis: {str(e)}")
                        st.info("Try extending the time period for better wave identification.")
            
            # Elliott Wave Theory Reference
            st.markdown("---")
            st.subheader("Elliott Wave Theory Reference")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Impulse Waves (5-Wave Pattern)**
                - **Wave i**: First impulse move
                - **Wave ii**: Corrective retracement (never retraces >100% of wave i)
                - **Wave iii**: Strong impulse move (often longest, never shortest)
                - **Wave iv**: Corrective retracement (never overlaps wave i territory)
                - **Wave v**: Final impulse move
                
                **Rules:**
                - Wave ii never retraces more than 100% of wave i
                - Wave iii is never the shortest of waves i, iii, and v
                - Wave iv never overlaps with wave i price territory
                """)
            
            with col2:
                st.markdown("""
                **Corrective Waves (3-Wave Pattern)**
                - **Wave a**: First corrective move against main trend
                - **Wave b**: Counter-trend retracement of wave a
                - **Wave c**: Final corrective move, often extends beyond wave a
                
                **Common Patterns:**
                - **Zigzag**: Sharp corrective pattern (5-3-5 structure)
                - **Flat**: Sideways corrective pattern (3-3-5 structure)
                - **Triangle**: Contracting pattern before final move
                
                **Fibonacci Relationships:**
                - Wave ii often retraces 38.2% or 61.8% of wave i
                - Wave iv often retraces 23.6% or 38.2% of wave iii
                - Wave c often equals wave a or extends 161.8% of wave a
                """)
            
            st.info("""
            **Important Notes:**
            - Elliott Wave analysis is subjective and patterns can be interpreted differently
            - Multiple valid wave counts may exist simultaneously
            - Wave patterns work best when combined with other technical analysis
            - Short-term charts may show sub-waves within larger degree waves
            - Always consider multiple timeframes for complete perspective
            """)
        
        else:
            st.error("Unable to load data for the selected cryptocurrency.")
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

elif mode == "Portfolio Analysis & Backtest":
    st.header("Portfolio Analysis & Backtesting")
    
    # Get cached data
    price_data, returns = get_cached_data(
        selected_symbols, 
        start_date.strftime('%Y-%m-%d'), 
        end_date.strftime('%Y-%m-%d')
    )
    
    # Manual portfolio weight input (unified from both modes)
    st.subheader("Define Your Portfolio Weights")
    
    # Create columns for weight input
    n_cols = min(len(selected_symbols), 3)  # Max 3 columns for better layout
    cols = st.columns(n_cols)
    
    portfolio_weights = {}
    total_weight = 0
    
    for i, symbol in enumerate(selected_symbols):
        col_idx = i % n_cols
        with cols[col_idx]:
            # Check for optimized weights from Portfolio Optimization mode
            if 'portfolio_weights' in st.session_state and symbol in st.session_state.portfolio_weights:
                default_weight = st.session_state.portfolio_weights[symbol]
            else:
                default_weight = 1.0/len(selected_symbols)  # Default equal weight
            
            weight = st.number_input(
                f"{symbol.replace('-USD', '')} Weight", 
                min_value=0.0, 
                max_value=1.0, 
                value=default_weight,
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
    
    # Store weights in session state for other modes
    st.session_state.portfolio_weights = normalized_weights
    
    # Portfolio Allocation Chart
    st.subheader("Portfolio Allocation")
    
    # Create allocation DataFrame
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
        legend=dict(orientation="v", yanchor="middle", y=0.5),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Portfolio backtest simulation
    st.subheader("Realistic Backtest Simulation")
    
    # Common calculations for coin availability
    coin_launch_info = {}
    available_symbols_timeline = []
    
    # Track coin launches and availability
    for i in range(1, len(price_data)):
        for symbol in selected_symbols:
            prev_price = price_data[symbol].iloc[i-1]
            curr_price = price_data[symbol].iloc[i]
            
            is_launched = (not pd.isna(prev_price) and not pd.isna(curr_price) and 
                          prev_price > 0 and curr_price > 0)
            
            if is_launched and symbol not in coin_launch_info:
                first_available_date = price_data.index[i-1].strftime('%Y-%m-%d')
                analysis_start_date = price_data.index[0].strftime('%Y-%m-%d')
                
                if first_available_date == analysis_start_date:
                    coin_launch_info[symbol] = {'date': first_available_date, 'type': 'pre-existing'}
                else:
                    coin_launch_info[symbol] = {'date': first_available_date, 'type': 'launched'}
    
    # Display coin availability information in dropdown
    if coin_launch_info:
        with st.expander("Cryptocurrency Availability"):
            availability_data = []
            for symbol in selected_symbols:
                if symbol in coin_launch_info:
                    info = coin_launch_info[symbol]
                    if info['type'] == 'pre-existing':
                        status = f"Available from start"
                        launch_info = f"Launched before {info['date']}"
                    else:
                        status = f"Launched {info['date']}"
                        days_since_start = (pd.to_datetime(info['date']) - price_data.index[0]).days
                        launch_info = f"Day {days_since_start} of analysis"
                    
                    availability_data.append({
                        'Cryptocurrency': symbol.replace('-USD', ''),
                        'Status': status,
                        'Details': launch_info,
                        'Weight': f"{normalized_weights[symbol]:.1%}"
                    })
                else:
                    availability_data.append({
                        'Cryptocurrency': symbol.replace('-USD', ''),
                        'Status': 'Not available',
                        'Details': 'No valid data in period',
                        'Weight': f"{normalized_weights[symbol]:.1%}"
                    })
            
            availability_df = pd.DataFrame(availability_data)
            st.dataframe(availability_df, use_container_width=True, hide_index=True)
    
    # Backtest parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        initial_capital = st.number_input("Initial Capital ($)", value=100000, min_value=1000, step=10000)
    with col2:
        rebalance_freq = st.selectbox("Rebalancing", ["No Rebalancing", "Monthly", "Quarterly", "Semi-Annual"])
    with col3:
        trading_fee = st.number_input("Trading Fee (%)", value=0.1, min_value=0.0, max_value=1.0, step=0.05) / 100
    
    # Generate rebalancing dates
    def get_rebalance_dates(date_index, frequency):
        rebalance_dates = []
        if frequency == "No Rebalancing":
            return set()  # No rebalancing dates
        elif frequency == "Monthly":
            for year in range(date_index[0].year, date_index[-1].year + 1):
                for month in range(1, 13):
                    month_start = pd.Timestamp(year, month, 1)
                    month_dates = [d for d in date_index if d >= month_start and d.month == month and d.year == year]
                    if month_dates:
                        rebalance_dates.append(min(month_dates))
        elif frequency == "Quarterly":
            for year in range(date_index[0].year, date_index[-1].year + 1):
                for month in [1, 4, 7, 10]:
                    quarter_start = pd.Timestamp(year, month, 1)
                    quarter_dates = [d for d in date_index if d >= quarter_start and d.month == month and d.year == year]
                    if quarter_dates:
                        rebalance_dates.append(min(quarter_dates))
        else:  # Semi-Annual
            for year in range(date_index[0].year, date_index[-1].year + 1):
                for month in [1, 7]:
                    semi_start = pd.Timestamp(year, month, 1)
                    semi_dates = [d for d in date_index if d >= semi_start and d.month == month and d.year == year]
                    if semi_dates:
                        rebalance_dates.append(min(semi_dates))
        return set(rebalance_dates)
    
    rebalance_dates = get_rebalance_dates(price_data.index, rebalance_freq)
    
    # Portfolio simulation with realistic costs
    portfolio_values = []
    portfolio_shares = {symbol: 0 for symbol in selected_symbols}
    rebalance_costs = []
    
    # Equal weight benchmark for comparison
    def get_dynamic_equal_weights(symbols, price_data, date_index):
        available_symbols = []
        for symbol in symbols:
            if not pd.isna(price_data[symbol].iloc[date_index]) and price_data[symbol].iloc[date_index] > 0:
                available_symbols.append(symbol)
        
        if not available_symbols:
            return {symbol: 0 for symbol in symbols}
        
        equal_weight_per_coin = 1.0 / len(available_symbols)
        dynamic_weights = {}
        for symbol in symbols:
            if symbol in available_symbols:
                dynamic_weights[symbol] = equal_weight_per_coin
            else:
                dynamic_weights[symbol] = 0
        return dynamic_weights
    
    equal_weight_values = []
    btc_hold_values = []
    
    # BTC buy & hold calculation (if BTC data is available)
    btc_initial_shares = 0
    has_btc_data = 'BTC-USD' in price_data.columns
    if has_btc_data and not pd.isna(price_data['BTC-USD'].iloc[0]) and price_data['BTC-USD'].iloc[0] > 0:
        btc_initial_capital = initial_capital * (1 - trading_fee)  # Apply trading fee
        btc_initial_shares = btc_initial_capital / price_data['BTC-USD'].iloc[0]
    
    for i, date in enumerate(price_data.index):
            if i == 0:
                # Initial investment with trading costs
                initial_cost = initial_capital * trading_fee
                remaining_capital = initial_capital - initial_cost
                
                # Simple allocation using fixed weights
                for symbol in selected_symbols:
                    if not pd.isna(price_data[symbol].iloc[i]) and price_data[symbol].iloc[i] > 0:
                        allocation = remaining_capital * normalized_weights[symbol]
                        portfolio_shares[symbol] = allocation / price_data[symbol].iloc[i]
                    else:
                        portfolio_shares[symbol] = 0
                
                # Calculate actual portfolio value from shares (should equal remaining_capital)
                portfolio_value = sum(portfolio_shares[symbol] * price_data[symbol].iloc[i] 
                                    for symbol in selected_symbols 
                                    if not pd.isna(price_data[symbol].iloc[i]) and price_data[symbol].iloc[i] > 0)
                equal_weight_value = initial_capital
                rebalance_costs.append(initial_cost)
                
                # BTC buy & hold value
                if has_btc_data and btc_initial_shares > 0:
                    btc_hold_value = btc_initial_shares * price_data['BTC-USD'].iloc[i]
                else:
                    btc_hold_value = initial_capital
            else:
                # Calculate current portfolio value
                current_portfolio_value = 0
                for symbol in selected_symbols:
                    if not pd.isna(price_data[symbol].iloc[i]) and price_data[symbol].iloc[i] > 0:
                        current_portfolio_value += portfolio_shares[symbol] * price_data[symbol].iloc[i]
                
                # Check if rebalancing needed
                if date in rebalance_dates:
                    turnover = 0.5
                    rebalance_cost = current_portfolio_value * turnover * trading_fee
                    net_value = current_portfolio_value - rebalance_cost
                    
                    # Rebalance with dynamic weights
                    available_symbols = []
                    for symbol in selected_symbols:
                        if not pd.isna(price_data[symbol].iloc[i]) and price_data[symbol].iloc[i] > 0:
                            available_symbols.append(symbol)
                    
                    if available_symbols:
                        available_original_weights = {sym: normalized_weights.get(sym, 1.0/len(selected_symbols)) for sym in available_symbols}
                        total_available_weight = sum(available_original_weights.values())
                        unavailable_weight = 1.0 - total_available_weight
                        
                        if total_available_weight > 0:
                            rebalance_dynamic_weights = {}
                            for symbol in available_symbols:
                                proportional_share = available_original_weights[symbol] / total_available_weight
                                rebalance_dynamic_weights[symbol] = available_original_weights[symbol] + (unavailable_weight * proportional_share)
                            
                            for symbol in selected_symbols:
                                if symbol in available_symbols:
                                    allocation = net_value * rebalance_dynamic_weights[symbol]
                                    portfolio_shares[symbol] = allocation / price_data[symbol].iloc[i]
                                else:
                                    portfolio_shares[symbol] = 0
                        else:
                            for symbol in selected_symbols:
                                portfolio_shares[symbol] = 0
                    else:
                        for symbol in selected_symbols:
                            portfolio_shares[symbol] = 0
                    
                    portfolio_value = net_value
                    rebalance_costs.append(rebalance_cost)
                else:
                    portfolio_value = current_portfolio_value
                    rebalance_costs.append(0)
                
                # Equal weight benchmark
                current_equal_weights = get_dynamic_equal_weights(selected_symbols, price_data, i)
                equal_weight_return = 0
                for symbol in selected_symbols:
                    if current_equal_weights[symbol] > 0 and not pd.isna(price_data[symbol].iloc[i]) and not pd.isna(price_data[symbol].iloc[i-1]):
                        symbol_return = price_data[symbol].iloc[i] / price_data[symbol].iloc[i-1] - 1
                        equal_weight_return += current_equal_weights[symbol] * symbol_return
                equal_weight_value = equal_weight_values[-1] * (1 + equal_weight_return)
                
                # BTC buy & hold value
                if has_btc_data and btc_initial_shares > 0 and not pd.isna(price_data['BTC-USD'].iloc[i]):
                    btc_hold_value = btc_initial_shares * price_data['BTC-USD'].iloc[i]
                else:
                    btc_hold_value = btc_hold_values[-1] if btc_hold_values else initial_capital
            
            portfolio_values.append(portfolio_value)
            equal_weight_values.append(equal_weight_value)
            btc_hold_values.append(btc_hold_value)
        
    # Performance metrics
    portfolio_total_return = (portfolio_values[-1] / portfolio_values[0] - 1)
    equal_weight_total_return = (equal_weight_values[-1] / equal_weight_values[0] - 1)
    btc_total_return = (btc_hold_values[-1] / btc_hold_values[0] - 1) if btc_hold_values else 0
    total_trading_costs = sum(rebalance_costs)
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Portfolio Return", f"{portfolio_total_return:.1%}")
    with col2:
        st.metric("Equal Weight Return", f"{equal_weight_total_return:.1%}")
    with col3:
        st.metric("BTC Buy & Hold", f"{btc_total_return:.1%}")
    with col4:
        st.metric("Trading Costs", f"${total_trading_costs:,.0f}")
    with col5:
        st.metric("Cost Ratio", f"{total_trading_costs/initial_capital:.2%}")
    
    # Calculate drawdown for portfolio (needed for comparisons)
    rolling_max = pd.Series(portfolio_values).expanding().max()
    drawdown = (pd.Series(portfolio_values) - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Backtest Performance Chart
    st.subheader("Backtest Performance")
    
    backtest_fig = go.Figure()
    backtest_fig.add_trace(go.Scatter(
        x=price_data.index,
        y=portfolio_values,
        mode='lines',
        name='Your Portfolio',
        line=dict(width=3, color='#1652f0')
    ))
    backtest_fig.add_trace(go.Scatter(
        x=price_data.index,
        y=equal_weight_values,
        mode='lines',
        name='Equal Weight Benchmark',
        line=dict(width=2, color='#00d395', dash='dash')
    ))
    
    # Add BTC buy & hold if data is available
    if has_btc_data and btc_hold_values:
        backtest_fig.add_trace(go.Scatter(
            x=price_data.index,
            y=btc_hold_values,
            mode='lines',
            name='BTC Buy & Hold',
            line=dict(width=2, color='#F7931A', dash='dot')
        ))
    
    backtest_fig.update_layout(
        title="Realistic Backtest Performance (With Trading Costs)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(backtest_fig, use_container_width=True)
        
    # Rebalancing info
    num_rebalances = len([cost for cost in rebalance_costs if cost > 0]) - 1  # Exclude initial cost
    if rebalance_freq == "No Rebalancing":
        st.info(f"Buy-and-hold strategy: No rebalancing performed. Initial trading cost: {total_trading_costs/initial_capital:.2%} of capital.")
    elif num_rebalances > 0:
        st.info(f"Portfolio rebalanced {num_rebalances} times. Total trading costs: {total_trading_costs/initial_capital:.2%} of initial capital.")
    else:
        st.info("No rebalancing occurred during this period.")
    
    # Drawdown Analysis
    st.subheader("Downturn Analysis")
    
    # Display max drawdown metric (drawdown already calculated above)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Max Drawdown", f"{max_drawdown:.1%}")
    with col2:
        st.metric("Volatility", f"{pd.Series(portfolio_values).pct_change().std() * np.sqrt(252):.1%}")
    
    # Drawdown chart
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
    
    st.plotly_chart(fig_dd, use_container_width=True)

elif mode == "Portfolio Optimization":
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
            
            # Optimized Portfolio Allocation Chart
            st.subheader("Optimized Portfolio Allocation")
            
            # Create allocation DataFrame
            allocation_data = []
            for asset, weight in result['weights'].items():
                if weight > 0.001:  # Only show meaningful allocations
                    allocation_data.append({
                        'Asset': asset.replace('-USD', ''),
                        'Weight': weight
                    })
            
            allocation_df = pd.DataFrame(allocation_data)
            allocation_df = allocation_df.sort_values('Weight', ascending=False)
            
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
                title="Optimized Portfolio Weights",
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
                legend=dict(orientation="v", yanchor="middle", y=0.5),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Set optimized weights as the active portfolio weights
            st.session_state.portfolio_weights = result['weights']
            st.success("Optimized weights have been applied to your portfolio!")
            
            # Redirection message to other modes
            st.info("**Next Steps:** Go to the **Portfolio Analysis** mode to see detailed backtesting results with your optimized weights, or visit **ML Prediction** mode to get AI-powered forecasts using these optimal allocations.")
            
        
        else:
            st.error(f"Optimization failed: {result.get('error', 'Unknown error')}")
            st.info("Try adjusting your parameters or selecting different assets")
    
    else:
        st.info("Configure your preferences and click 'Optimize Portfolio' to get started")
    
    

elif mode == "Market Insights":
    # Market Analysis Section
    st.header("Market Analysis & Insights")
    
    # Get cached data
    price_data, returns = get_cached_data(
        selected_symbols, 
        start_date.strftime('%Y-%m-%d'), 
        end_date.strftime('%Y-%m-%d')
    )
    
    # Market overview with improved layout
    col1, col2 = st.columns([2, 1])
    
    
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
    
    # Check if cryptocurrencies are selected
    if not selected_symbols:
        st.warning("Please select cryptocurrencies in the sidebar to run ML predictions.")
        st.stop()
    
    
    # ML Configuration in sidebar
    st.sidebar.subheader("ML Configuration")
    prediction_days = st.sidebar.slider("Prediction Period (days)", 1, 30, 7)
    
    # Training window optimization option
    optimize_train_window = st.sidebar.checkbox(
        "Auto-optimize Training Window", 
        value=True,
        help="Use cross-validation to find the optimal training window length for the selected prediction period"
    )
    
    if optimize_train_window:
        st.sidebar.info("Training window will be optimized using ensemble of multiple models")
        # We'll determine this later based on CV optimization
        train_window = None
        optimization_method = "Robust (Ensemble)"  # Always use ensemble when optimizing
    else:
        train_window = st.sidebar.slider("Training Window (days)", 60, 500, 150)
        optimization_method = None
    
    # Calculate CV requirements and show warnings (validation window = prediction period)
    validation_window = prediction_days  # Validation window should match prediction period
    
    # Cross-validation info
    with st.sidebar.expander("Cross-Validation Info", expanded=False):
        st.write(f"**Validation window:** {validation_window} days")
        
        if optimize_train_window:
            st.write("**Training window optimization:**")
            st.write("- Will test multiple window sizes (30-180 days)")
            st.write("- Uses time-series cross-validation")
            st.write("- Optimizes individually for each cryptocurrency")
            st.write("- Selects window with best validation performance")
            st.write("- Balances model accuracy vs. overfitting")
        else:
            min_train_size = max(90, train_window // 2)
            required_min_data = min_train_size + (validation_window * 2)
            st.write(f"**Recommended minimum:** {required_min_data} days")
            
            if train_window < 90:
                st.error(f"Training window ({train_window} days) is below minimum requirement (90 days). Increase to at least 90 days for reliable results.")
            elif train_window < required_min_data:
                st.warning(f"Training window ({train_window} days) is below recommended minimum ({required_min_data} days) for robust cross-validation with {prediction_days}-day predictions.")
            elif train_window < 120:
                st.info(f"Short-term focus: {train_window} days is adequate for {prediction_days}-day predictions. Consider 120+ days for more robust validation.")
            elif train_window >= 365:
                st.info(f"Large training window ({train_window} days) selected - automatically loading extended historical data for robust training.")
            else:
                st.success(f"Training window is excellent for robust cross-validation ({train_window} days) with {prediction_days}-day validation periods.")
    
    # Calculate extended data range for ML training (after ML config is defined)
    # Need extra data to account for: feature engineering loss (~25 days) + training window + prediction buffer
    feature_engineering_buffer = 30  # Conservative buffer for technical indicators
    
    if optimize_train_window:
        # For optimization, we need to load enough data for the largest possible training window
        max_train_window = 180  # Maximum window we'll test
        ml_buffer_days = max(0, max_train_window + prediction_days + feature_engineering_buffer - (end_date - start_date).days)
        if ml_buffer_days > 0:
            st.info(f"Loading additional {ml_buffer_days} days of historical data to support training window optimization")
    else:
        ml_buffer_days = max(0, train_window + prediction_days + feature_engineering_buffer - (end_date - start_date).days)
        if ml_buffer_days > 0:
            st.info(f"Loading additional {ml_buffer_days} days of historical data to support {train_window}-day training window")
    
    # Extend start date if needed for large training windows
    ml_start_date = start_date - timedelta(days=ml_buffer_days) if ml_buffer_days > 0 else start_date
    
    # Get cached data with extended range
    try:
        price_data, returns = get_cached_data(
            selected_symbols, 
            ml_start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        
        # Check if we have data
        if price_data is None or price_data.empty:
            st.error("No price data available for the selected period and cryptocurrencies.")
            st.info("Try selecting a longer date range or different cryptocurrencies.")
            st.stop()
            
        # Display data info
        total_days_loaded = (end_date - ml_start_date).days
        st.info(f"Loaded data for {len(selected_symbols)} cryptocurrencies from {ml_start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({total_days_loaded} days)")
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()
    
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
    
    # Training window evaluation function
    def evaluate_training_window(prices, target_days, train_window, use_ensemble=False):
        """
        Evaluate a training window using time series cross-validation.
        Returns the average CV score (lower is better).
        
        Args:
            use_ensemble: If True, uses ensemble of models. If False, uses only Linear Regression.
        """
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.linear_model import LinearRegression, Lasso
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error
        
        # Create features
        features = create_features(prices)
        
        # Create target variable
        target = features['returns'].shift(-target_days)
        
        # Align features and target
        aligned_data = pd.concat([features, target.rename('target')], axis=1).dropna()
        
        if len(aligned_data) < train_window + target_days + 30:
            raise ValueError(f"Insufficient data for {train_window}-day training window")
        
        X = aligned_data.drop('target', axis=1)
        y = aligned_data['target']
        
        # Use only the most recent data for evaluation
        X = X.tail(train_window + target_days * 3)  # Use 3x prediction period for CV
        y = y.tail(train_window + target_days * 3)
        
        # Time series cross-validation
        n_splits = min(3, len(X) // (train_window // 2))  # Fewer splits for speed
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Define models based on user selection
        if use_ensemble:
            # Adaptive hyperparameters based on training window size and data characteristics
            n_features = X.shape[1]
            n_samples = min(len(X), train_window)
            
            # Lasso alpha: lighter for shorter windows (more flexibility), stronger for longer windows
            lasso_alpha = 0.05 if train_window <= 60 else (0.1 if train_window <= 120 else 0.15)
            
            # RF parameters: smaller ensemble for shorter windows, larger for longer windows
            rf_n_estimators = max(15, min(30, train_window // 3))  # Scale with window size
            rf_max_depth = max(3, min(6, int(np.log2(n_features)) + 1))  # Scale with feature count
            
            models = {
                'linear': LinearRegression(),
                'lasso': Lasso(alpha=lasso_alpha, max_iter=1000),  # Adaptive regularization
                'rf': RandomForestRegressor(
                    n_estimators=rf_n_estimators, 
                    max_depth=rf_max_depth,
                    min_samples_split=max(5, n_samples // 20),  # Prevent overfitting on small windows
                    min_samples_leaf=max(2, n_samples // 40),
                    random_state=42, 
                    n_jobs=1
                )
            }
            weights = {'linear': 0.4, 'lasso': 0.4, 'rf': 0.2}
        else:
            # Fast mode: only Linear Regression
            models = {
                'linear': LinearRegression()
            }
            weights = {'linear': 1.0}
        
        all_model_scores = {name: [] for name in models.keys()}
        scaler = StandardScaler()
        
        for train_idx, val_idx in tscv.split(X):
            try:
                X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
                
                # Only use the last train_window points for training
                if len(X_train_cv) > train_window:
                    X_train_cv = X_train_cv.tail(train_window)
                    y_train_cv = y_train_cv.tail(train_window)
                
                # Scale features
                X_train_scaled = scaler.fit_transform(X_train_cv.fillna(0))
                X_val_scaled = scaler.transform(X_val_cv.fillna(0))
                
                # Evaluate each model
                for name, model in models.items():
                    try:
                        # Train and predict
                        model.fit(X_train_scaled, y_train_cv)
                        y_pred = model.predict(X_val_scaled)
                        
                        # Calculate score
                        score = mean_squared_error(y_val_cv, y_pred)
                        all_model_scores[name].append(score)
                        
                    except Exception as e:
                        # Skip this model for this fold
                        continue
                        
            except Exception as e:
                # Skip this fold if it fails
                continue
        
        # Calculate ensemble score (average of model scores, weighted by reliability)
        model_avg_scores = {}
        for name, scores in all_model_scores.items():
            if scores:
                model_avg_scores[name] = np.mean(scores)
        
        if not model_avg_scores:
            raise ValueError("All models failed in CV")
        
        # Use weighted average based on selected models
        
        ensemble_score = 0
        total_weight = 0
        for name, score in model_avg_scores.items():
            weight = weights.get(name, 0)
            ensemble_score += score * weight
            total_weight += weight
        
        # Normalize by actual total weight (in case some models failed)
        if total_weight > 0:
            return ensemble_score / total_weight
        else:
            # Fallback to simple average if weighting fails
            return np.mean(list(model_avg_scores.values()))
    
    # Window choice interpretation function
    def interpret_window_choice(window):
        """Interpret what the optimal window choice means"""
        if window <= 30:
            return "Very short memory: Highly adaptive, captures immediate market shifts"
        elif window <= 60:
            return "Short memory: Captures recent volatility, adapts quickly to regime changes"
        elif window <= 90:
            return "Medium-short memory: Balances recent trends with some stability"
        elif window <= 120:
            return "Medium memory: Good balance of trend capture and noise reduction"
        elif window <= 150:
            return "Medium-long memory: Stable patterns, less sensitive to short-term noise"
        else:
            return "Long memory: Captures long-term patterns, very stable predictions"
    
    # ML Prediction function
    def predict_returns(prices, target_days=7, train_window=90):
        try:
            features = create_features(prices)
            
            # Prepare target (future returns) with bounds
            target = prices.shift(-target_days) / prices - 1
            
            # Cap extreme target values to improve training stability
            target = target.clip(-0.8, 2.0)  # Cap between -80% and +200%
            
            # Align features and target - remove NaN values
            valid_idx = features.index.intersection(target.index)
            X = features.loc[valid_idx]
            y = target.loc[valid_idx]
            
            # Remove rows with any NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask].fillna(0)
            y = y[mask].fillna(0)
            
            # Check if we have enough data (accounting for feature engineering losses)
            # Feature engineering typically loses ~20-25 days due to moving averages, RSI, etc.
            feature_engineering_loss = 25  # Conservative estimate
            effective_train_window = max(train_window - feature_engineering_loss, 30)
            
            if len(X) < effective_train_window:
                actual_available = len(X) + feature_engineering_loss  # Estimate original data size
                st.warning(f"Insufficient data for training. Need at least {train_window} days, got ~{actual_available} days after feature engineering.")
                st.info(f"Try reducing training window to {len(X) + feature_engineering_loss - 10} days or loading more historical data.")
                return None, None, None
            
            # Use available data for training, avoiding future data leakage
            # We need to avoid using future data that hasn't happened yet
            max_train_index = len(X) - target_days
            if max_train_index <= 0:
                st.warning("Not enough historical data for the selected prediction period")
                return None, None, None
            
            # Use the last train_window days, but not beyond max_train_index
            start_idx = max(0, max_train_index - train_window)
            end_idx = max_train_index
            
            X_train = X.iloc[start_idx:end_idx]
            y_train = y.iloc[start_idx:end_idx]
            
            # Ensure we have training data
            if len(X_train) == 0 or len(y_train) == 0:
                st.warning("No training data available after processing")
                return None, None, None
                
        except Exception as e:
            st.error(f"Error in data preparation: {str(e)}")
            return None, None, None
        
        if len(X_train) < 20:  # Minimum samples required
            return None, None, None
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Configure time series cross-validation with CV information display
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
        
        validation_window = target_days
        min_train_size = max(45, len(X_train) // 3)  # Reduced for shorter windows
        
        # Use TimeSeriesSplit for cross-validation - optimized for shorter training windows
        n_splits = min(4, max(3, len(X_train) // 25))  # Slightly more frequent splits for shorter windows
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_splits = list(tscv.split(X_train))
        
        # CV information will be displayed once before training starts
        
        # Train models with cross-validation
        models = {}
        
        # Random Forest with CV (optimized for shorter training windows)
        rf_param_grid = {
            'n_estimators': [30, 50],  # Reduced for faster training with shorter windows
            'max_depth': [6, 10],      # Slightly reduced to prevent overfitting
            'min_samples_split': [8, 15], # Increased to prevent overfitting on smaller datasets
            'min_samples_leaf': [4, 8],   # Increased for regularization
            'max_features': ['sqrt']
        }
        
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            rf_param_grid,
            cv=cv_splits,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        rf_grid.fit(X_train_scaled, y_train)
        models['Random Forest'] = rf_grid.best_estimator_
        
        # Linear Regression (no hyperparameters to tune)
        models['Linear Regression'] = LinearRegression()
        models['Linear Regression'].fit(X_train_scaled, y_train)
        
        # Lasso with CV (optimized for shorter training windows)
        from sklearn.linear_model import LassoCV
        lasso_cv = LassoCV(
            alphas=np.logspace(-1, 1, 20),  # Focused range for shorter windows
            cv=cv_splits,
            random_state=42,
            max_iter=1500,  # Reduced iterations for faster training
            selection='random'  # Random coordinate selection for sparsity
        )
        lasso_cv.fit(X_train_scaled, y_train)
        models['Lasso'] = lasso_cv
        
        # Neural Network removed due to convergence issues on small financial datasets
        
        # XGBoost/Gradient Boosting with CV (optimized for 7-day predictions)
        if XGBOOST_AVAILABLE:
            xgb_param_grid = {
                'n_estimators': [25, 50],      # Reduced for shorter training windows
                'max_depth': [2, 4],           # Slightly increased max depth for 7-day patterns
                'learning_rate': [0.02, 0.05], # Focused on moderate learning rates
                'subsample': [0.8],            # Increased subsample for stability
                'colsample_bytree': [0.7],     # Increased column sampling
                'reg_alpha': [5.0, 15.0],      # Reduced regularization for shorter windows
                'reg_lambda': [50.0, 100.0]    # More flexible lambda range
            }
            
            xgb_grid = GridSearchCV(
                xgb.XGBRegressor(random_state=42, verbosity=0),
                xgb_param_grid,
                cv=cv_splits,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            xgb_grid.fit(X_train_scaled, y_train)
            models['XGBoost'] = xgb_grid.best_estimator_
        else:
            gb_param_grid = {
                'n_estimators': [25, 50],      # Reduced for shorter training windows
                'max_depth': [2, 4],           # Slightly increased for 7-day patterns
                'learning_rate': [0.02, 0.05], # Focused learning rates
                'subsample': [0.8],            # Increased for stability
                'min_samples_split': [15, 25], # Adjusted for shorter windows
                'min_samples_leaf': [8, 12],   # Adjusted for regularization
                'max_features': ['sqrt']
            }
            
            gb_grid = GridSearchCV(
                GradientBoostingRegressor(random_state=42),
                gb_param_grid,
                cv=cv_splits,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            gb_grid.fit(X_train_scaled, y_train)
            models['Gradient Boosting'] = gb_grid.best_estimator_
        
        predictions = {}
        metrics = {}
        
        for name, model in models.items():
            try:
                # All models are already fitted with cross-validation above
                
                # Predict on the last available features
                last_features = X.iloc[-1:].fillna(0)
                last_features_scaled = scaler.transform(last_features)
                pred = model.predict(last_features_scaled)[0]
                
                # Validate prediction - cap extreme values for financial returns
                if abs(pred) > 1.0:  # More than 100% return is suspicious
                    pred = np.sign(pred) * min(abs(pred), 0.5)  # Cap at 50%
                    
                predictions[name] = pred
                
                # Calculate metrics on training data
                train_pred = model.predict(X_train_scaled)
                metrics[name] = {
                    'MSE': mean_squared_error(y_train, train_pred),
                    'MAE': mean_absolute_error(y_train, train_pred),
                    'R2': r2_score(y_train, train_pred) if len(set(y_train)) > 1 else 0
                }
                
            except Exception as e:
                st.warning(f"Model {name} failed: {str(e)}")
                predictions[name] = 0
                metrics[name] = {'MSE': float('inf'), 'MAE': float('inf'), 'R2': 0}
        
        return predictions, metrics, X_train.index
    
    # Run predictions for each cryptocurrency
    st.subheader("Individual Asset Predictions")
    
    # Add debugging information
    
    prediction_results = {}
    all_predictions_data = []
    
    # Training window optimization (if enabled)
    if optimize_train_window:
        # Define candidate training windows to test
        candidate_windows = [30, 45, 60, 90, 120, 150, 180]
        
        # Filter candidates based on available data
        max_available_data = min([len(price_data[symbol].dropna()) for symbol in selected_symbols])
        candidate_windows = [w for w in candidate_windows if w + prediction_days + 30 <= max_available_data]
        
        if len(candidate_windows) < 2:
            st.warning("Insufficient data for training window optimization. Using default 150 days.")
            train_window = 150
            # Create default per-asset mapping
            optimal_windows = {symbol: 150 for symbol in selected_symbols}
        else:
            # Optimize for each cryptocurrency individually
            optimal_windows = {}
            all_results = {}
            
            total_steps = len(selected_symbols) * len(candidate_windows)
            opt_progress = st.progress(0)
            opt_status = st.empty()
            step = 0
            
            opt_status.text(f"Optimizing training windows for {len(selected_symbols)} cryptocurrencies...")
            
            for symbol in selected_symbols:
                asset_name = symbol.replace('-USD', '')
                prices = price_data[symbol].dropna()
                
                asset_results = []
                
                for window in candidate_windows:
                    step += 1
                    opt_status.text(f"Testing {asset_name}: {window}-day window ({step}/{total_steps})")
                    opt_progress.progress(step / total_steps)
                    
                    try:
                        use_ensemble = optimization_method == "Robust (Ensemble)"
                        score = evaluate_training_window(prices, prediction_days, window, use_ensemble)
                        asset_results.append({
                            'Asset': asset_name,
                            'Training Window': window,
                            'CV Score': score,
                            'CV Score (%)': f"{score*100:.3f}%"
                        })
                    except Exception as e:
                        st.warning(f"Failed to evaluate {asset_name} with {window}-day window: {str(e)}")
                        continue
                
                if asset_results:
                    # Select best window for this asset
                    best_result = min(asset_results, key=lambda x: x['CV Score'])
                    optimal_windows[symbol] = best_result['Training Window']
                    all_results[symbol] = asset_results
                else:
                    # Fallback for this asset
                    optimal_windows[symbol] = 120
                    st.warning(f"Using fallback 120 days for {asset_name}")
            
            opt_status.empty()
            opt_progress.empty()
            
            if optimal_windows:
                # Use the most common optimal window as the default for display
                # But we'll actually use individual windows during training
                train_window = max(set(optimal_windows.values()), key=list(optimal_windows.values()).count)
            else:
                st.error("Training window optimization failed for all assets. Using default 150 days.")
                train_window = 150
                optimal_windows = {symbol: 150 for symbol in selected_symbols}
    else:
        # When optimization is disabled, create a mapping with the manual training window
        optimal_windows = {symbol: train_window for symbol in selected_symbols}
    
    # Display consolidated training information
    if optimize_train_window:
        # Show that individual windows are being used
        window_range = f"{min(optimal_windows.values())}-{max(optimal_windows.values())}" if len(set(optimal_windows.values())) > 1 else str(list(optimal_windows.values())[0])
        st.info(f"Training {len(selected_symbols)} cryptocurrencies with optimized windows ({window_range} days) using time series cross-validation | Prediction period: {prediction_days} days")
    else:
        st.info(f"Training {len(selected_symbols)} cryptocurrencies with {train_window}-day window using time series cross-validation | Prediction period: {prediction_days} days")
    
    # Store the prediction period used for training (for AI analysis)
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = {}
    st.session_state.ml_prediction_days = prediction_days
    
    # Data Summary (after training window is determined)
    with st.expander("Data Summary", expanded=False):
        st.write(f"**Selected Assets:** {len(selected_symbols)}")
        st.write(f"**Date Range:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        st.write(f"**Available Data Days:** {len(price_data)}")
        
        if optimize_train_window:
            window_range = f"{min(optimal_windows.values())}-{max(optimal_windows.values())}" if len(set(optimal_windows.values())) > 1 else str(list(optimal_windows.values())[0])
            st.write(f"**Training Windows:** {window_range} days (optimized per cryptocurrency)")
        else:
            st.write(f"**Training Window:** {train_window} days")
            
        st.write(f"**Prediction Period:** {prediction_days} days")
        
        # Show data availability for each asset
        data_summary = []
        for symbol in selected_symbols:
            prices = price_data[symbol].dropna()
            symbol_window = optimal_windows.get(symbol, train_window)
            data_summary.append({
                'Asset': symbol.replace('-USD', ''),
                'Data Points': len(prices),
                'Training Window': f"{symbol_window} days" if optimize_train_window else f"{train_window} days",
                'Latest Price': f"${prices.iloc[-1]:.2f}" if len(prices) > 0 else "No data",
                'Status': "Ready" if len(prices) >= symbol_window + prediction_days else "Insufficient"
            })
        
        if data_summary:
            st.dataframe(pd.DataFrame(data_summary), use_container_width=True, hide_index=True)
    
    # Progress bar for model training
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Collect all prediction data first
    for i, symbol in enumerate(selected_symbols):
        asset_name = symbol.replace('-USD', '')
        
        # Use individual optimal window if available, otherwise use default
        if optimize_train_window and symbol in optimal_windows:
            symbol_train_window = optimal_windows[symbol]
            status_text.text(f"Training models for {asset_name} (using {symbol_train_window}-day window)...")
        else:
            symbol_train_window = train_window
            status_text.text(f"Training models for {asset_name}...")
            
        progress_bar.progress((i + 1) / len(selected_symbols))
        
        prices = price_data[symbol]
        predictions, metrics, train_dates = predict_returns(prices, prediction_days, symbol_train_window)
        
        if predictions is None:
            st.warning(f"Skipping {symbol.replace('-USD', '')} - insufficient data or training failed")
            continue
        
        prediction_results[symbol] = predictions
        
        # Store predictions in session state for AI analysis
        st.session_state.prediction_results[symbol] = predictions
        
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
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.text("Model training completed successfully!")
    
    if not all_predictions_data:
        st.error("No successful predictions were generated for any of the selected cryptocurrencies.")
        st.info("This could be due to:")
        st.write("""
        - Insufficient historical data (need at least 30-90 days)
        - Data quality issues
        - Selected date range too short
        - Technical issues with feature calculation
        """)
        st.info("Try:")
        st.write("""
        - Selecting a longer date range (6+ months recommended)
        - Reducing the training window in ML Configuration
        - Selecting different cryptocurrencies with more historical data
        """)
    else:
        predictions_df = pd.DataFrame(all_predictions_data)
        
        # Create comprehensive prediction visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced bar chart for predictions by asset and model
            # Define professional color palette for all models (Neural Network removed)
            model_colors = {
                'Random Forest': '#2E8B57',      # Sea Green
                'Linear Regression': '#4169E1',  # Royal Blue  
                'Lasso': '#FF1493',             # Deep Pink
                'XGBoost': '#9932CC',           # Dark Orchid
                'Gradient Boosting': '#FF8C00'  # Dark Orange
            }
            
            fig_bar = px.bar(
                predictions_df,
                x='Asset',
                y='Prediction',
                color='Model',
                title=f"Predicted Returns by Asset ({prediction_days}-day forecast)",
                labels={'Prediction': 'Predicted Return', 'Asset': 'Cryptocurrency'},
                text=[f"{val:.1%}" for val in predictions_df['Prediction']],
                color_discrete_map=model_colors,
                barmode='group'
            )
            
            # Enhanced trace styling
            fig_bar.update_traces(
                textposition='outside',
                textfont_size=10,
                textfont_color='black',
                marker_line_width=0.5,
                marker_line_color='white'
            )
            
            # Smart y-axis range calculation
            y_min = predictions_df['Prediction'].min()
            y_max = predictions_df['Prediction'].max()
            y_center = (y_min + y_max) / 2
            y_range = abs(y_max - y_min)
            
            # Ensure minimum range for readability
            if y_range < 0.02:  # Less than 2% range
                y_range = 0.05  # Set minimum 5% range
            
            # Dynamic padding based on value magnitude and range
            base_padding = y_range * 0.25
            magnitude_padding = max(abs(y_min), abs(y_max)) * 0.15
            min_padding = 0.03  # Minimum 3% padding
            
            padding = max(base_padding, magnitude_padding, min_padding)
            
            # Apply asymmetric padding for better balance
            if y_max > 0:
                top_padding = padding * 1.5  # More space for positive labels
            else:
                top_padding = padding
                
            if y_min < 0:
                bottom_padding = padding * 1.5  # More space for negative labels
            else:
                bottom_padding = padding
            
            fig_bar.update_layout(
                template='plotly_white',
                height=500,
                title_font_size=16,
                title_x=0.5,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    bgcolor='rgba(0,0,0,0)',
                    borderwidth=0
                ),
                xaxis=dict(
                    title="Cryptocurrency",
                    title_font_size=14,
                    tickfont_size=12,
                    gridcolor='rgba(0,0,0,0.1)'
                ),
                yaxis=dict(
                    title="Predicted Return",
                    title_font_size=14,
                    tickfont_size=11,
                    tickformat='.1%',
                    gridcolor='rgba(0,0,0,0.1)',
                    range=[y_min - bottom_padding, y_max + top_padding],
                    zeroline=True,
                    zerolinecolor='rgba(0,0,0,0.3)',
                    zerolinewidth=1
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=80, b=60, l=60, r=40)
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
                    
                    # Calculate agreement metrics using relative difference
                    avg_pred = (rf_pred + lr_pred) / 2
                    if abs(avg_pred) > 0.001:  # Avoid division by very small numbers
                        relative_diff = abs(rf_pred - lr_pred) / abs(avg_pred)
                        agreement_score = max(0, (1 - relative_diff)) * 100  # Convert to 0-100%
                    else:
                        agreement_score = 0  # Uncertain when predictions are near zero
                    
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
                    yaxis=dict(range=[0, 115]),  # Increased upper limit to show 100% labels
                    showlegend=False,
                    margin=dict(t=40, b=40, l=60, r=20)  # Adjusted margins for better text display
                )
                
                st.plotly_chart(fig_agreement, use_container_width=True)
        
        # Detailed predictions table
        st.subheader("Detailed Predictions Summary")
        
        # Create a more readable table with all models
        table_data = []
        available_models = ['Random Forest', 'Linear Regression', 'Lasso']
        if XGBOOST_AVAILABLE:
            available_models.append('XGBoost')
        else:
            available_models.append('Gradient Boosting')
        
        for symbol in selected_symbols:
            if symbol in prediction_results:
                asset_name = symbol.replace('-USD', '')
                row_data = {'Asset': asset_name}
                
                # Add predictions for each model
                model_predictions = []
                for model_name in available_models:
                    pred = prediction_results[symbol].get(model_name, 0)
                    row_data[model_name] = f"{pred:.2%}"
                    model_predictions.append(pred)
                
                # Calculate average and standard deviation for confidence
                avg_pred = np.mean(model_predictions)
                std_pred = np.std(model_predictions)
                row_data['Average'] = f"{avg_pred:.2%}"
                
                # Enhanced confidence calculation based on model agreement
                if std_pred < 0.01:
                    confidence = "Very High"
                elif std_pred < 0.02:
                    confidence = "High"
                elif std_pred < 0.04:
                    confidence = "Medium"
                else:
                    confidence = "Low"
                    
                row_data['Confidence'] = confidence
                table_data.append(row_data)
        
        if table_data:
            summary_df = pd.DataFrame(table_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Portfolio-level predictions
    if prediction_results:
        st.subheader("Portfolio-Level Predictions")
        
        # Allow user to input portfolio weights
        st.write("**Define Portfolio Weights for Prediction**")
        
        # Check if optimized weights are available
        if 'portfolio_weights' in st.session_state:
            st.info("Using optimized weights from Portfolio Optimization mode. You can adjust them below if needed.")
        
        portfolio_weights = {}
        cols = st.columns(min(len(selected_symbols), 3))
        
        total_weight = 0
        for i, symbol in enumerate(selected_symbols):
            with cols[i % len(cols)]:
                # Use optimized weights if available, otherwise equal weights
                if 'portfolio_weights' in st.session_state and symbol in st.session_state.portfolio_weights:
                    default_weight = st.session_state.portfolio_weights[symbol]
                else:
                    default_weight = 1.0/len(selected_symbols)
                
                weight = st.number_input(
                    f"{symbol.replace('-USD', '')} Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=default_weight,
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
        
        # Calculate portfolio predictions for all available models
        available_models = ['Random Forest', 'Linear Regression', 'Lasso']
        if XGBOOST_AVAILABLE:
            available_models.append('XGBoost')
        else:
            available_models.append('Gradient Boosting')
        
        portfolio_predictions = {model: 0 for model in available_models}
        
        for symbol in selected_symbols:
            if symbol in prediction_results:
                for model_name in available_models:
                    if model_name in prediction_results[symbol]:
                        portfolio_predictions[model_name] += (
                            normalized_weights[symbol] * prediction_results[symbol][model_name]
                        )
        
        # Display portfolio predictions
        num_cols = len(available_models)
        cols = st.columns(num_cols)
        
        for i, model_name in enumerate(available_models):
            with cols[i]:
                st.metric(
                    f"{model_name} Portfolio",
                    f"{portfolio_predictions[model_name]:.2%}",
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
        
        # Create dynamic colors for all models plus ensemble
        model_colors_list = []
        for model in models:
            if model in model_colors:
                model_colors_list.append(model_colors[model])
            else:
                model_colors_list.append('#888888')  # Default gray for unknown models
        model_colors_list.append('#000000')  # Black for ensemble
        
        fig_pred.add_trace(go.Bar(
            x=models + ['Ensemble'],
            y=predictions_vals + [ensemble_prediction],
            marker_color=model_colors_list,
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
    
    # API Key validation functions
    def validate_openai_key(api_key):
        """Validate OpenAI API key"""
        if not api_key:
            return None, None
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=10)
            if response.status_code == 200:
                return True, "Valid OpenAI API key"
            else:
                return False, "Invalid OpenAI API key"
        except Exception as e:
            return False, f"Error validating OpenAI key: Connection failed"
    
    def validate_anthropic_key(api_key):
        """Validate Anthropic API key"""
        if not api_key:
            return None, None
        
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "test"}]
        }
        try:
            response = requests.post("https://api.anthropic.com/v1/messages", 
                                   headers=headers, json=data, timeout=10)
            if response.status_code == 200:
                return True, "Valid Anthropic API key"
            else:
                return False, "Invalid Anthropic API key"
        except Exception as e:
            return False, f"Error validating Anthropic key: Connection failed"
    
    def validate_google_key(api_key):
        """Validate Google API key"""
        if not api_key:
            return None, None
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": "test"}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 10
            }
        }
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            if response.status_code == 200:
                return True, "Valid Google API key"
            else:
                return False, "Invalid Google API key"
        except Exception as e:
            return False, f"Error validating Google key: Connection failed"

    # API Key input
    api_key = st.sidebar.text_input(
        "API Key",
        type="password",
        help="Enter your API key for the selected provider"
    )
    
    # Real-time API key validation
    if api_key:
        validation_placeholder = st.sidebar.empty()
        
        # Validate based on selected provider
        if llm_provider == "OpenAI (GPT-4)":
            is_valid, message = validate_openai_key(api_key)
        elif llm_provider == "Anthropic (Claude)":
            is_valid, message = validate_anthropic_key(api_key)
        else:  # Google (Gemini)
            is_valid, message = validate_google_key(api_key)
        
        # Display validation result
        if is_valid is True:
            validation_placeholder.success(message)
        elif is_valid is False:
            validation_placeholder.error(message)
        # If is_valid is None, show nothing (empty key)
    
    # Check if we should stop (no key or invalid key)
    should_stop = False
    if not api_key:
        st.warning("Please enter your API key in the sidebar to get AI investment advice.")
        st.info("""
        **How to get API keys:**
        - **OpenAI**: Visit https://platform.openai.com/api-keys
        - **Anthropic**: Visit https://console.anthropic.com/
        - **Google**: Visit https://ai.google.dev/
        """)
        should_stop = True
    elif api_key:
        # Check validation result
        if llm_provider == "OpenAI (GPT-4)":
            is_valid, _ = validate_openai_key(api_key)
        elif llm_provider == "Anthropic (Claude)":
            is_valid, _ = validate_anthropic_key(api_key)
        else:  # Google (Gemini)
            is_valid, _ = validate_google_key(api_key)
        
        if is_valid is False:
            st.error("Please enter a valid API key to continue.")
            should_stop = True
    
    if should_stop:
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
    
    # Show the actual prediction period from trained models (read-only)
    if hasattr(st.session_state, 'prediction_results') and st.session_state.prediction_results:
        # Try to extract the actual prediction period from session state
        # For now, we'll check if there's a way to get it, otherwise default to common values
        prediction_days = getattr(st.session_state, 'ml_prediction_days', 7)  # Get actual training period
        st.sidebar.info(f"Using ML models trained for **{prediction_days}-day** predictions")
        st.sidebar.caption("💡 To change prediction period, retrain models in ML Predictions section")
    else:
        prediction_days = 7  # Default fallback
        st.sidebar.warning("No ML predictions available")
        st.sidebar.caption("👉 Run ML Predictions first to get CV-tuned results")
    
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
    
    # ML predictions for AI analysis - Use CV-tuned predictions from main ML section
    def run_ml_predictions_for_ai(prediction_days):
        ml_results = {}
        
        try:
            # Check if we have CV-tuned predictions from the main ML section
            if hasattr(st.session_state, 'prediction_results') and st.session_state.prediction_results:
                # Use the CV-tuned predictions from session state
                cv_predictions = st.session_state.prediction_results
                
                for symbol in selected_symbols:
                    asset_name = symbol.replace('-USD', '')
                    if symbol in cv_predictions:
                        pred_data = cv_predictions[symbol]
                        
                        # Extract individual model predictions
                        rf_pred = pred_data.get('Random Forest', 0)
                        lr_pred = pred_data.get('Linear Regression', 0)
                        lasso_pred = pred_data.get('Lasso', 0)
                        
                        # Get boosting prediction (XGBoost or Gradient Boosting)
                        if 'XGBoost' in pred_data:
                            boost_pred = pred_data['XGBoost']
                            boost_name = f"xgboost_{prediction_days}d"
                        else:
                            boost_pred = pred_data.get('Gradient Boosting', 0)
                            boost_name = f"gradient_boosting_{prediction_days}d"
                        
                        ensemble_pred = pred_data.get('Ensemble', np.mean([rf_pred, lr_pred, boost_pred]))
                        
                        # Calculate model agreement using coefficient of variation
                        all_predictions = [rf_pred, lr_pred, boost_pred]
                        mean_pred = np.mean(all_predictions)
                        model_std = np.std(all_predictions)
                        
                        if abs(mean_pred) > 0.001:
                            cv = model_std / abs(mean_pred)
                            if cv < 0.05:
                                agreement = "Very High"
                            elif cv < 0.10:
                                agreement = "High"
                            elif cv < 0.20:
                                agreement = "Medium"
                            else:
                                agreement = "Low"
                        else:
                            agreement = "Uncertain"
                        
                        ml_results[asset_name] = {
                            f"random_forest_{prediction_days}d": f"{rf_pred:.2%}",
                            f"linear_regression_{prediction_days}d": f"{lr_pred:.2%}",
                            f"lasso_{prediction_days}d": f"{lasso_pred:.2%}",
                            boost_name: f"{boost_pred:.2%}",
                            f"ensemble_prediction_{prediction_days}d": f"{ensemble_pred:.2%}",
                            "model_agreement": agreement,
                            "prediction_std": f"{model_std:.3f}",
                            "cv_tuned": "Yes"  # Indicate these are CV-tuned predictions
                        }
            else:
                # Fallback: If no CV predictions available, indicate this clearly
                for symbol in selected_symbols:
                    asset_name = symbol.replace('-USD', '')
                    ml_results[asset_name] = {
                        "status": "CV predictions not available",
                        "note": "Run ML Predictions first to get CV-tuned results",
                        "cv_tuned": "No"
                    }
                        
        except Exception as e:
            print(f"Error accessing CV-tuned ML predictions for AI: {e}")
            # Return empty results if CV predictions fail
            ml_results = {}
        
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
    
    # Generate timeframe-specific market context
    def generate_timeframe_context(horizon, prediction_days=7):
        """Generate different data context based on investment horizon"""
        context = {}
        
        if "Short-term" in horizon:
            # Short-term: Focus on recent performance, technical indicators, momentum
            context_days = 30  # Last 30 days
            context["timeframe"] = "Short-term tactical analysis (30 days)"
            context["focus"] = "Technical indicators, momentum, short-term volatility"
        elif "Medium-term" in horizon:
            # Medium-term: Balance of recent and historical data
            context_days = 90  # Last 90 days
            context["timeframe"] = "Medium-term strategic analysis (90 days)"
            context["focus"] = "Trend analysis, seasonal patterns, medium-term fundamentals"
        else:  # Long-term
            # Long-term: Full historical data, fundamentals, macro trends
            context_days = min(365, len(price_data))  # Last year or all available data
            context["timeframe"] = "Long-term investment analysis (365 days)"
            context["focus"] = "Historical patterns, long-term growth, macro fundamentals"
        
        # Get subset of data for the timeframe
        recent_price_data = price_data.tail(context_days)
        recent_returns = returns.tail(context_days)
        
        # Calculate timeframe-specific metrics
        for symbol in selected_symbols:
            asset_name = symbol.replace('-USD', '')
            
            # Price data for the specific timeframe
            asset_prices = recent_price_data[symbol].dropna()
            asset_returns = recent_returns[symbol].dropna()
            
            if len(asset_prices) > 0:
                current_price = asset_prices.iloc[-1]
                period_start_price = asset_prices.iloc[0]
                period_return = (current_price / period_start_price - 1) * 100
                period_volatility = asset_returns.std() * np.sqrt(252) * 100
                
                # Technical indicators for the timeframe
                if len(asset_prices) >= 20:
                    sma_20 = asset_prices.rolling(20).mean().iloc[-1]
                    rsi = calculate_rsi_simple(asset_prices, 14).iloc[-1] if len(asset_prices) >= 14 else None
                    
                    # MACD
                    ema_12 = asset_prices.ewm(span=12).mean().iloc[-1]
                    ema_26 = asset_prices.ewm(span=26).mean().iloc[-1] if len(asset_prices) >= 26 else ema_12
                    macd = ema_12 - ema_26
                else:
                    sma_20 = current_price
                    rsi = None
                    macd = 0
                
                # Momentum indicators
                momentum_short = None
                momentum_medium = None
                
                if len(asset_prices) >= 7:
                    momentum_short = (current_price / asset_prices.iloc[-7] - 1) * 100  # 7-day momentum
                if len(asset_prices) >= 30:
                    momentum_medium = (current_price / asset_prices.iloc[-30] - 1) * 100  # 30-day momentum
                
                context[asset_name] = {
                    "period_return": f"{period_return:.2f}%",
                    "period_volatility": f"{period_volatility:.2f}%",
                    "current_price": f"${current_price:.2f}",
                    "sma_20_ratio": f"{(current_price / sma_20 - 1) * 100:.2f}%" if sma_20 > 0 else "N/A",
                    "rsi": f"{rsi:.1f}" if rsi is not None else "N/A",
                    "macd_signal": "Bullish" if macd > 0 else "Bearish",
                    f"momentum_{min(7, prediction_days)}d": f"{momentum_short:.2f}%" if momentum_short is not None else "N/A",
                    "momentum_30d": f"{momentum_medium:.2f}%" if momentum_medium is not None else "N/A",
                    "data_points": len(asset_prices)
                }
        
        return context
    
    # Helper function for RSI calculation
    def calculate_rsi_simple(prices, window=14):
        """Simple RSI calculation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Generate comprehensive analysis data
    def generate_analysis_data(prediction_days):
        analysis_data = {
            "selected_assets": [symbol.replace('-USD', '') for symbol in selected_symbols],
            "time_period": f"{start_date} to {end_date}",
            "investment_horizon": investment_horizon,
            "risk_tolerance": risk_tolerance,
            "timeframe_context": generate_timeframe_context(investment_horizon, prediction_days),
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
        analysis_data["ml_predictions"] = run_ml_predictions_for_ai(prediction_days)
        
        # Get portfolio optimization results
        analysis_data["portfolio_optimization"] = run_portfolio_optimization_for_ai()
        
        # Enhanced correlation analysis
        try:
            corr_matrix = returns.corr()
            # Calculate average correlation (excluding diagonal)
            avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            
            # Find highest and lowest correlations
            corr_pairs = []
            for i in range(len(selected_symbols)):
                for j in range(i+1, len(selected_symbols)):
                    asset1 = selected_symbols[i].replace('-USD', '')
                    asset2 = selected_symbols[j].replace('-USD', '')
                    correlation = corr_matrix.iloc[i, j]
                    corr_pairs.append({
                        'pair': f"{asset1}-{asset2}",
                        'correlation': correlation
                    })
            
            # Sort to find highest and lowest
            corr_pairs_sorted = sorted(corr_pairs, key=lambda x: x['correlation'])
            lowest_corr = corr_pairs_sorted[0] if corr_pairs_sorted else None
            highest_corr = corr_pairs_sorted[-1] if corr_pairs_sorted else None
            
            # Calculate correlation distribution
            corr_values = [pair['correlation'] for pair in corr_pairs]
            high_corr_count = sum(1 for corr in corr_values if corr > 0.7)
            medium_corr_count = sum(1 for corr in corr_values if 0.3 <= corr <= 0.7)
            low_corr_count = sum(1 for corr in corr_values if corr < 0.3)
            
            analysis_data["correlation_analysis"] = {
                "average_correlation": f"{avg_correlation:.3f}",
                "diversification_potential": "Low" if avg_correlation > 0.7 else "Medium" if avg_correlation > 0.3 else "High",
                "highest_correlation": {
                    "pair": highest_corr['pair'] if highest_corr else "N/A",
                    "value": f"{highest_corr['correlation']:.3f}" if highest_corr else "N/A"
                },
                "lowest_correlation": {
                    "pair": lowest_corr['pair'] if lowest_corr else "N/A", 
                    "value": f"{lowest_corr['correlation']:.3f}" if lowest_corr else "N/A"
                },
                "correlation_distribution": {
                    "high_correlation_pairs": high_corr_count,
                    "medium_correlation_pairs": medium_corr_count,
                    "low_correlation_pairs": low_corr_count,
                    "total_pairs": len(corr_pairs)
                },
                "interpretation": {
                    "diversification_level": "Poor" if avg_correlation > 0.8 else "Fair" if avg_correlation > 0.6 else "Good" if avg_correlation > 0.4 else "Excellent",
                    "risk_concentration": "High" if avg_correlation > 0.7 else "Medium" if avg_correlation > 0.5 else "Low"
                }
            }
        except:
            analysis_data["correlation_analysis"] = {
                "average_correlation": "Unable to calculate",
                "diversification_potential": "Unknown"
            }
        
        return analysis_data
    
    # Generate AI analysis
    if st.button("Get AI Investment Advice", type="primary"):
        with st.spinner("Generating AI analysis..."):
            analysis_data = generate_analysis_data(prediction_days)
            
            # Create structured prompt for AI
            prompt = f"""
            As an expert cryptocurrency investment advisor, analyze the comprehensive data and provide a structured response in JSON format:
            
            **Portfolio Configuration:**
            - Selected Assets: {', '.join(analysis_data['selected_assets'])}
            - Risk Tolerance: {risk_tolerance}
            - Investment Horizon: {investment_horizon}
            - Analysis Period: {analysis_data['time_period']}
            
            **Timeframe-Specific Analysis Context:**
            {json.dumps(analysis_data['timeframe_context'], indent=2)}
            
            **Horizon-Focused Market Data:**
            This data is filtered and calculated specifically for {investment_horizon.lower()} analysis:
            {json.dumps(analysis_data['market_data'], indent=2)}
            
            **Machine Learning Predictions ({prediction_days}-day forecasts using 4 models):**
            Note: Predictions generated using Random Forest, Linear Regression, Lasso, and XGBoost/Gradient Boosting models.
            Features include: RSI, MACD (line, signal, histogram), moving averages, momentum indicators, and volatility measures.
            
            **IMPORTANT - Weight ML predictions based on investment horizon:**
            - Short-term (1-3 months): HIGH weight on ML predictions (70% importance)
            - Medium-term (3-12 months): MODERATE weight on ML predictions (40% importance)  
            - Long-term (1+ years): LOW weight on ML predictions (20% importance)
            
            {json.dumps(analysis_data['ml_predictions'], indent=2)}
            
            **Portfolio Optimization Results (Multiple Methods):**
            Consider optimization method relevance based on investment horizon:
            - Short-term: Focus on momentum and technical factors, tactical rebalancing
            - Medium-term: Balance risk-return optimization with trend following
            - Long-term: Emphasize fundamental diversification and strategic allocation
            {json.dumps(analysis_data['portfolio_optimization'], indent=2)}
            
            **Correlation Analysis:**
            {json.dumps(analysis_data['correlation_analysis'], indent=2)}
            
            **Current Portfolio (User Defined):**
            {json.dumps(current_portfolio_weights, indent=2)}
            
            **Horizon-Specific Analysis Instructions:**
            
            {"SHORT-TERM FOCUS: Prioritize technical indicators from timeframe_context (RSI, MACD signals, momentum_{min(7, prediction_days)}d/30d), recent volatility patterns, and ML predictions. Make tactical recommendations based on current market momentum and technical analysis. Consider more frequent rebalancing." if "Short-term" in investment_horizon else ""}
            
            {"MEDIUM-TERM FOCUS: Balance technical analysis with fundamental trends. Use timeframe_context period returns and volatility. Weight ML predictions moderately. Consider seasonal patterns and medium-term momentum. Recommend strategic moves with some tactical adjustments." if "Medium-term" in investment_horizon else ""}
            
            {"LONG-TERM FOCUS: Emphasize fundamental analysis, historical performance patterns from timeframe_context, and diversification benefits from portfolio optimization. Use ML predictions as minor supporting evidence only. Focus on strategic asset allocation over tactical moves. Recommend buy-and-hold approach with periodic rebalancing." if "Long-term" in investment_horizon else ""}
            
            **Analysis Process:**
            1. First, analyze the timeframe_context data that's specifically relevant to {investment_horizon}
            2. Weight the importance of different data sources based on the investment horizon  
            3. Use horizon-appropriate metrics (short-term momentum vs long-term fundamentals)
            4. Provide portfolio allocation reasoning that matches the investment timeframe
            5. Include rebalancing frequency recommendations appropriate for the horizon
            
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
                    "risk_controls": "12% below entry for volatile assets (ETH, SOL), 8% for BTC. No single trade > 5% of portfolio value. Weekly monitoring, execute when >5% deviation"
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
                                        {from_pct:.1f}% → {to_pct:.1f}%
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
                                    # Format risk controls nicely
                                    risk_controls = impl.get('risk_controls', 'Set appropriate stop losses')
                                    if isinstance(risk_controls, dict):
                                        # Format complex risk controls structure
                                        formatted_controls = []
                                        
                                        # Stop loss levels
                                        if 'stop_loss_levels' in risk_controls:
                                            stop_losses = risk_controls['stop_loss_levels']
                                            if isinstance(stop_losses, dict):
                                                stop_loss_text = ", ".join([f"{asset}: {level}" for asset, level in stop_losses.items()])
                                                formatted_controls.append(f"Stop losses: {stop_loss_text}")
                                            else:
                                                formatted_controls.append(f"Stop losses: {stop_losses}")
                                        
                                        # Position limits
                                        if 'position_limits' in risk_controls:
                                            formatted_controls.append(f"Position limits: {risk_controls['position_limits']}")
                                        
                                        # Monitoring
                                        if 'monitoring' in risk_controls:
                                            formatted_controls.append(f"Monitoring: {risk_controls['monitoring']}")
                                        
                                        # Volatility triggers
                                        if 'volatility_triggers' in risk_controls:
                                            formatted_controls.append(f"Volatility triggers: {risk_controls['volatility_triggers']}")
                                        
                                        risk_controls_text = ". ".join(formatted_controls)
                                    else:
                                        risk_controls_text = str(risk_controls)
                                    
                                    st.warning(f"**Risk Controls:** {risk_controls_text}")
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
        if 'analysis_data' in locals():
            st.write("### Data Sources Sent to AI")
            
            # Show timeframe context first
            st.write("**1. Timeframe-Specific Analysis Context**")
            st.info(f"**Investment Horizon:** {investment_horizon}")
            if "timeframe_context" in analysis_data:
                context = analysis_data["timeframe_context"]
                st.write(f"**Analysis Period:** {context.get('timeframe', 'N/A')}")
                st.write(f"**Focus Areas:** {context.get('focus', 'N/A')}")
                
                # Show timeframe-specific metrics for each asset
                context_df_data = []
                for asset, metrics in context.items():
                    if asset not in ['timeframe', 'focus']:
                        context_df_data.append({
                            'Asset': asset,
                            'Period Return': metrics.get('period_return', 'N/A'),
                            'Period Volatility': metrics.get('period_volatility', 'N/A'),
                            'RSI': metrics.get('rsi', 'N/A'),
                            'MACD Signal': metrics.get('macd_signal', 'N/A'),
                            f'{min(7, prediction_days)}d Momentum': metrics.get(f'momentum_{min(7, prediction_days)}d', 'N/A'),
                            '30d Momentum': metrics.get('momentum_30d', 'N/A'),
                            'Data Points': metrics.get('data_points', 'N/A')
                        })
                
                if context_df_data:
                    context_df = pd.DataFrame(context_df_data)
                    st.dataframe(context_df, use_container_width=True, hide_index=True)
            
            # Show the data weight guidance
            st.write("**2. AI Guidance Based on Investment Horizon**")
            if "Short-term" in investment_horizon:
                st.success("**SHORT-TERM FOCUS:** High weight on ML predictions (70%), technical indicators, and recent momentum")
            elif "Medium-term" in investment_horizon:
                st.info("**MEDIUM-TERM FOCUS:** Moderate weight on ML predictions (40%), balanced technical/fundamental analysis")
            else:
                st.warning("**LONG-TERM FOCUS:** Low weight on ML predictions (20%), emphasis on fundamentals and diversification")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**3. Market Performance Data (Full Period)**")
                market_source_df = pd.DataFrame(analysis_data["market_data"]).T
                st.dataframe(market_source_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.write(f"**4. ML Predictions ({prediction_days}-day forecasts)**")
                if analysis_data["ml_predictions"]:
                    tech_source_df = pd.DataFrame(analysis_data["ml_predictions"]).T
                    st.dataframe(tech_source_df, use_container_width=True, hide_index=True)
                else:
                    st.write("No ML prediction data available")
            
            # Portfolio optimization and correlation data
            if analysis_data.get("portfolio_optimization"):
                st.write("**5. Portfolio Optimization Results**")
                opt_df_data = []
                for method, results in analysis_data["portfolio_optimization"].items():
                    opt_df_data.append({
                        'Method': method.replace('_', ' ').title(),
                        'Expected Return': results.get('expected_return', 'N/A'),
                        'Volatility': results.get('volatility', 'N/A'),
                        'Sharpe Ratio': results.get('sharpe_ratio', 'N/A')
                    })
                
                if opt_df_data:
                    opt_df = pd.DataFrame(opt_df_data)
                    st.dataframe(opt_df, use_container_width=True, hide_index=True)
            
            if analysis_data.get("correlation_analysis"):
                st.write("**6. Correlation Analysis**")
                corr_data = analysis_data["correlation_analysis"]
                
                # Create a clean presentation of correlation data
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_corr = corr_data.get("average_correlation", "N/A")
                    if avg_corr != "N/A" and avg_corr != "Unable to calculate":
                        corr_val = float(avg_corr)
                        color = "red" if corr_val > 0.7 else "orange" if corr_val > 0.5 else "green"
                        st.metric("Average Correlation", avg_corr, delta=None)
                        st.markdown(f"<span style='color: {color};'>**{corr_data.get('diversification_potential', 'N/A')} Diversification**</span>", unsafe_allow_html=True)
                    else:
                        st.metric("Average Correlation", "N/A")
                
                with col2:
                    if "highest_correlation" in corr_data:
                        highest = corr_data["highest_correlation"]
                        st.metric("Highest Correlation", 
                                f"{highest.get('value', 'N/A')}", 
                                delta=None)
                        st.caption(f"Pair: {highest.get('pair', 'N/A')}")
                    
                with col3:
                    if "lowest_correlation" in corr_data:
                        lowest = corr_data["lowest_correlation"]
                        st.metric("Lowest Correlation", 
                                f"{lowest.get('value', 'N/A')}", 
                                delta=None)
                        st.caption(f"Pair: {lowest.get('pair', 'N/A')}")
                
                # Show correlation distribution if available
                if "correlation_distribution" in corr_data:
                    dist = corr_data["correlation_distribution"]
                    st.write("**Correlation Distribution:**")
                    
                    dist_col1, dist_col2, dist_col3, dist_col4 = st.columns(4)
                    with dist_col1:
                        st.metric("High (>0.7)", dist.get("high_correlation_pairs", 0))
                    with dist_col2:
                        st.metric("Medium (0.3-0.7)", dist.get("medium_correlation_pairs", 0))
                    with dist_col3:
                        st.metric("Low (<0.3)", dist.get("low_correlation_pairs", 0))
                    with dist_col4:
                        st.metric("Total Pairs", dist.get("total_pairs", 0))
                
                # Show interpretation if available
                if "interpretation" in corr_data:
                    interp = corr_data["interpretation"]
                    div_level = interp.get("diversification_level", "Unknown")
                    risk_conc = interp.get("risk_concentration", "Unknown")
                    
                    # Color-code the interpretation
                    if div_level == "Excellent":
                        div_color = "green"
                    elif div_level == "Good":
                        div_color = "blue"
                    elif div_level == "Fair":
                        div_color = "orange"
                    else:
                        div_color = "red"
                    
                    if risk_conc == "Low":
                        risk_color = "green"
                    elif risk_conc == "Medium":
                        risk_color = "orange"
                    else:
                        risk_color = "red"
                    
                    st.markdown(f"""
                    **Portfolio Assessment:**
                    - **Diversification Level:** <span style='color: {div_color}; font-weight: bold;'>{div_level}</span>
                    - **Risk Concentration:** <span style='color: {risk_color}; font-weight: bold;'>{risk_conc}</span>
                    """, unsafe_allow_html=True)
        else:
            st.write("Analysis data not available. Click 'Get AI Investment Advice' to generate data.")
    
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
    
    # Track dynamic weights and coin launches
    dynamic_weights_history = []
    coin_launch_info = {}
    
    for i in range(1, len(price_data)):
        # Determine which coins are available (launched) at this date
        available_symbols = []
        daily_returns = {}
        
        for symbol in selected_symbols:
            prev_price = price_data[symbol].iloc[i-1]
            curr_price = price_data[symbol].iloc[i]
            
            # Check if coin is launched (has valid price data)
            is_launched = (not pd.isna(prev_price) and not pd.isna(curr_price) and 
                          prev_price > 0 and curr_price > 0)
            
            if is_launched:
                available_symbols.append(symbol)
                daily_returns[symbol] = (curr_price / prev_price) - 1
                
                # Track first availability in our analysis period
                if symbol not in coin_launch_info:
                    first_available_date = price_data.index[i-1].strftime('%Y-%m-%d')
                    analysis_start_date = price_data.index[0].strftime('%Y-%m-%d')
                    
                    # Determine if this is actual launch or pre-existing
                    if first_available_date == analysis_start_date:
                        coin_launch_info[symbol] = {'date': first_available_date, 'type': 'pre-existing'}
                    else:
                        coin_launch_info[symbol] = {'date': first_available_date, 'type': 'launched'}
            else:
                daily_returns[symbol] = 0  # Not yet launched or missing data
        
        # Calculate dynamic weights: redistribute unavailable coins' weights to available ones
        if available_symbols:
            # Get original weights for available symbols
            available_original_weights = {sym: normalized_weights[sym] for sym in available_symbols}
            total_available_weight = sum(available_original_weights.values())
            
            # If some coins are unavailable, redistribute their weights proportionally
            unavailable_weight = 1.0 - total_available_weight
            
            if total_available_weight > 0:
                # Redistribute unavailable weight proportionally among available coins
                dynamic_weights = {}
                for symbol in available_symbols:
                    proportional_share = available_original_weights[symbol] / total_available_weight
                    dynamic_weights[symbol] = available_original_weights[symbol] + (unavailable_weight * proportional_share)
                
                # Ensure weights sum to 1.0
                total_dynamic_weight = sum(dynamic_weights.values())
                if total_dynamic_weight > 0:
                    dynamic_weights = {sym: weight/total_dynamic_weight for sym, weight in dynamic_weights.items()}
            else:
                # Fallback: equal weights if no original weights available
                dynamic_weights = {sym: 1.0/len(available_symbols) for sym in available_symbols}
        else:
            # No coins available - use zero weights
            dynamic_weights = {}
        
        # Store dynamic weights for analysis
        dynamic_weights_history.append({
            'date': price_data.index[i],
            'available_coins': len(available_symbols),
            'weights': dynamic_weights.copy()
        })
        
        # Calculate weighted portfolio return using dynamic weights
        portfolio_return = 0
        for symbol in available_symbols:
            if symbol in daily_returns and symbol in dynamic_weights:
                portfolio_return += dynamic_weights[symbol] * daily_returns[symbol]
        
        portfolio_returns.append(portfolio_return)
        portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
    
    # Calculate portfolio metrics with robust error handling
    portfolio_returns = np.array(portfolio_returns)
    
    # Total return calculation
    if portfolio_values[0] > 0 and portfolio_values[-1] > 0:
        portfolio_total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    else:
        portfolio_total_return = 0
    
    # Volatility calculation
    if len(portfolio_returns) > 1 and not np.all(np.isnan(portfolio_returns)):
        portfolio_volatility = np.nanstd(portfolio_returns) * np.sqrt(252)
    else:
        portfolio_volatility = 0
    
    # Annualized return calculation with proper error handling
    if (len(portfolio_returns) > 0 and 
        portfolio_values[0] > 0 and 
        portfolio_values[-1] > 0 and
        not pd.isna(portfolio_values[-1]) and
        not pd.isna(portfolio_values[0])):
        
        time_factor = 252 / len(portfolio_returns)  # Convert to annual
        value_ratio = portfolio_values[-1] / portfolio_values[0]
        
        # Ensure value_ratio is positive for power calculation
        if value_ratio > 0:
            portfolio_annualized_return = (value_ratio ** time_factor) - 1
        else:
            portfolio_annualized_return = portfolio_total_return  # Fallback to total return
    else:
        portfolio_annualized_return = 0
    
    # Sharpe ratio calculation
    if portfolio_volatility > 0 and not pd.isna(portfolio_annualized_return):
        portfolio_sharpe = (portfolio_annualized_return - 0.02) / portfolio_volatility
    else:
        portfolio_sharpe = 0
    
    # Maximum drawdown calculation
    portfolio_values_series = pd.Series(portfolio_values)
    running_max = portfolio_values_series.expanding().max()
    drawdown = (portfolio_values_series - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Display portfolio metrics
    st.subheader("Your Portfolio Performance")
    
    # Check for data quality issues
    if len(portfolio_returns) < 30:
        st.warning("Limited data available - metrics may be less reliable for very short periods.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if pd.isna(portfolio_total_return) or np.isinf(portfolio_total_return):
            st.metric("Total Return", "N/A", help="Unable to calculate due to data issues")
        else:
            st.metric("Total Return", f"{portfolio_total_return:.1%}")
    
    with col2:
        if pd.isna(portfolio_volatility) or np.isinf(portfolio_volatility):
            st.metric("Annualized Volatility", "N/A", help="Unable to calculate due to data issues")
        else:
            st.metric("Annualized Volatility", f"{portfolio_volatility:.1%}")
    
    with col3:
        if pd.isna(portfolio_sharpe) or np.isinf(portfolio_sharpe):
            st.metric("Sharpe Ratio", "N/A", help="Unable to calculate due to data issues")
        else:
            st.metric("Sharpe Ratio", f"{portfolio_sharpe:.2f}")
    
    with col4:
        if pd.isna(max_drawdown) or np.isinf(max_drawdown):
            st.metric("Max Drawdown", "N/A", help="Unable to calculate due to data issues")
        else:
            st.metric("Max Drawdown", f"{max_drawdown:.1%}")
    
    # Additional portfolio metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if pd.isna(portfolio_annualized_return) or np.isinf(portfolio_annualized_return):
            st.metric("Annualized Return", "N/A", help="Unable to calculate due to data issues")
        else:
            st.metric("Annualized Return", f"{portfolio_annualized_return:.1%}")
    
    with col2:
        # Add number of trading days for reference
        st.metric("Trading Days", f"{len(portfolio_returns)}")
    
    with col3:
        # Add start and end values for context
        st.metric("Portfolio Growth", f"${portfolio_values[0]:,.0f} → ${portfolio_values[-1]:,.0f}")
    
    with col4:
        # Add data quality indicator
        missing_data_pct = np.sum(np.isnan(portfolio_returns)) / len(portfolio_returns) * 100 if len(portfolio_returns) > 0 else 0
        if missing_data_pct > 5:
            st.metric("Data Quality", f"{100-missing_data_pct:.0f}%", delta="Some gaps")
        else:
            st.metric("Data Quality", f"{100-missing_data_pct:.0f}%", delta="Good")
    
    # Dynamic weights and coin launch analysis
    if coin_launch_info:
        st.subheader("Coin Launch Timeline & Dynamic Weights")
        
        # Show coin launch dates
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Coin Availability Timeline:**")
            launch_data = []
            for symbol, info in coin_launch_info.items():
                availability_date = info['date']
                availability_type = info['type']
                analysis_start = price_data.index[0].strftime('%Y-%m-%d')
                
                if availability_type == 'pre-existing':
                    # Coin was already available at start of analysis
                    status = f"Available from start"
                    note = f"(Launched before {analysis_start})"
                    days_after = "Pre-existing"
                else:
                    # Coin launched during analysis period
                    days_since_start = (pd.to_datetime(availability_date) - price_data.index[0]).days
                    status = f"Launched {availability_date}"
                    note = f"({days_since_start} days after start)"
                    days_after = f"Day {days_since_start}"
                
                launch_data.append({
                    'Cryptocurrency': symbol.replace('-USD', ''),
                    'Availability Status': status,
                    'Timeline': days_after,
                    'Note': note
                })
            
            if launch_data:
                launch_df = pd.DataFrame(launch_data)
                st.dataframe(launch_df, hide_index=True, use_container_width=True)
                
                # Add explanatory note
                st.caption("**Note**: For pre-existing coins, we only know they launched before the analysis start date, not their actual launch date.")
        
        with col2:
            st.markdown("**Weight Redistribution Summary:**")
            
            # Show how weights evolved over time
            if dynamic_weights_history:
                # Get initial and final weights for comparison
                initial_weights = dynamic_weights_history[0]['weights'] if dynamic_weights_history else {}
                final_weights = dynamic_weights_history[-1]['weights'] if dynamic_weights_history else {}
                
                weight_evolution = []
                for symbol in selected_symbols:
                    initial_weight = initial_weights.get(symbol, 0) * 100
                    final_weight = final_weights.get(symbol, 0) * 100
                    original_weight = normalized_weights[symbol] * 100
                    
                    weight_evolution.append({
                        'Crypto': symbol.replace('-USD', ''),
                        'Target %': f"{original_weight:.1f}%",
                        'Initial %': f"{initial_weight:.1f}%",
                        'Final %': f"{final_weight:.1f}%"
                    })
                
                weight_df = pd.DataFrame(weight_evolution)
                st.dataframe(weight_df, hide_index=True, use_container_width=True)
        
        # Show dynamic weights over time
        if len(dynamic_weights_history) > 1:
            st.markdown("**Dynamic Weight Adjustment Over Time:**")
            
            # Create chart showing number of available coins over time
            dates = [entry['date'] for entry in dynamic_weights_history]
            available_coins = [entry['available_coins'] for entry in dynamic_weights_history]
            
            fig_coins = go.Figure()
            fig_coins.add_trace(go.Scatter(
                x=dates,
                y=available_coins,
                mode='lines',
                name='Available Coins',
                line=dict(color='#1f77b4', width=2),
                fill='tonexty'
            ))
            
            fig_coins.update_layout(
                title="Number of Available Cryptocurrencies Over Time",
                xaxis_title="Date",
                yaxis_title="Number of Available Coins",
                height=300,
                template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
            )
            
            st.plotly_chart(fig_coins, use_container_width=True)
            
            # Explanation
            total_coins = len(selected_symbols)
            coins_available_at_start = available_coins[0] if available_coins else 0
            coins_launched_during_period = total_coins - coins_available_at_start
            
            if coins_launched_during_period > 0:
                st.info(f"""
                **Dynamic Weight Adjustment Applied:**
                - Started with {coins_available_at_start}/{total_coins} available cryptocurrencies
                - {coins_launched_during_period} cryptocurrencies became available during the analysis period
                - Weights were automatically redistributed as new coins became available
                - This prevents NaN values from unavailable cryptocurrencies
                """)
            else:
                st.success("All selected cryptocurrencies were available throughout the entire analysis period")
    
    # Price charts
    col_header, col_toggle = st.columns([3, 1])
    
    with col_header:
        st.subheader("Price Performance")
    
    with col_toggle:
        st.write("")  # Spacer
        price_view_mode = st.selectbox(
            "View Mode",
            ["Normalized", "Raw Price"],
            key="price_view_mode",
            help="Switch between normalized comparison (base=100) and actual price values"
        )
    
    # Create price chart based on selected mode
    fig = go.Figure()
    colors = ['#5ac53a', '#e85555', '#00ff88', '#f7931a', '#627eea']
    
    if price_view_mode == "Normalized":
        # Normalize prices to 100 for comparison
        chart_data = (price_data / price_data.iloc[0] * 100)
        chart_title = "Normalized Price Performance (Base = 100)"
        y_axis_title = "Normalized Price"
        
        for i, symbol in enumerate(selected_symbols):
            fig.add_trace(go.Scatter(
                x=chart_data.index,
                y=chart_data[symbol],
                name=symbol.replace('-USD', ''),
                line=dict(color=colors[i % len(colors)], width=2),
                mode='lines',
                hovertemplate=f'<b>{symbol.replace("-USD", "")}</b><br>' +
                             'Date: %{x}<br>' +
                             'Normalized Price: %{y:.1f}<br>' +
                             '<extra></extra>'
            ))
    else:  # Raw Price mode
        chart_data = price_data
        chart_title = "Raw Price Performance (USD)"
        y_axis_title = "Price (USD)"
        
        for i, symbol in enumerate(selected_symbols):
            fig.add_trace(go.Scatter(
                x=chart_data.index,
                y=chart_data[symbol],
                name=symbol.replace('-USD', ''),
                line=dict(color=colors[i % len(colors)], width=2),
                mode='lines',
                hovertemplate=f'<b>{symbol.replace("-USD", "")}</b><br>' +
                             'Date: %{x}<br>' +
                             'Price: $%{y:,.2f}<br>' +
                             '<extra></extra>'
            ))
    
    # Update layout
    fig.update_layout(
        title=chart_title,
        xaxis_title="Date",
        yaxis_title=y_axis_title,
        hovermode='x unified',
        height=400,
        template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True, hide_index=True)
    
    # Add explanation based on mode
    if price_view_mode == "Normalized":
        st.info("**Normalized View**: All prices start at 100 for easy performance comparison regardless of actual price levels.")
    else:
        st.info("**Raw Price View**: Shows actual USD prices. Note: Cryptocurrencies with very different price levels may be hard to compare on the same scale.")
    
    # Returns distribution and correlation
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot of returns
        returns_melted = returns.melt(var_name='Asset', value_name='Daily Return')
        returns_melted['Asset'] = returns_melted['Asset'].str.replace('-USD', '')
        
        fig_violin = px.violin(
            returns_melted, 
            x='Asset', 
            y='Daily Return',
            title="Daily Return Distributions",
            template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
        )
        fig_violin.update_layout(height=350)
        fig_violin.update_traces(opacity=0.6, points='outliers')
        st.plotly_chart(fig_violin, use_container_width=True, hide_index=True)
    
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