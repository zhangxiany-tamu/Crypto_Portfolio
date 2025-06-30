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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
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
    ["Market Insights", "Portfolio Analysis & Backtest", "Portfolio Optimization", "ML Predictions", "AI Investment Advisor"]
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

# Performance tip
st.sidebar.markdown("**Tip**: Data is automatically cached. Only fetches when symbols/dates change!")

# Main content area
if mode == "Market Insights":
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
    
    # Price Performance Section
    st.subheader("Price Performance")
    price_view_mode = st.selectbox("View Mode", ["Normalized", "Raw Price"])
    
    if price_view_mode == "Normalized":
        # Normalize to 100 at start
        chart_data = (price_data / price_data.iloc[0] * 100)
        y_title = "Normalized Price (Start = 100)"
    else:
        chart_data = price_data
        y_title = "Price (USD)"
    
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
        title=f"Price Performance ({price_view_mode})",
        xaxis_title="Date",
        yaxis_title=y_title,
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Individual Coin Analysis
    st.subheader("Detailed Coin Analysis")
    
    # Allow user to select specific coin for detailed analysis
    selected_coin = st.selectbox(
        "Select coin for detailed analysis:",
        options=selected_symbols,
        format_func=lambda x: x.replace('-USD', '')
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
        
        # Display detailed metrics in columns
        st.write(f"**{coin_name} Detailed Analysis**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${current_price:,.2f}")
            st.metric("1-Day Return", f"{returns_1d:.2%}")
            st.metric("7-Day Return", f"{returns_7d:.2%}")
        
        with col2:
            st.metric("30-Day Return", f"{returns_30d:.2%}")
            st.metric("90-Day Return", f"{returns_90d:.2%}")
            st.metric("Period Return", f"{total_return:.2%}")
        
        with col3:
            st.metric("Annual Volatility", f"{volatility_annual:.1%}")
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            st.metric("Max Drawdown", f"{max_drawdown:.1%}")
        
        with col4:
            st.metric("Period High", f"${price_high:,.2f}")
            st.metric("Period Low", f"${price_low:,.2f}")
            st.metric("From High", f"{current_from_high:.1%}")
        
        # Additional analysis tabs
        tab1, tab2, tab3 = st.tabs(["Price Distribution", "Returns Analysis", "Technical Indicators"])
        
        with tab1:
            # Price distribution analysis
            col1, col2 = st.columns([2, 1])
            
            with col1:
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
                
                # Box plot for price distribution
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(
                    y=coin_data,
                    name=f"{coin_name}",
                    boxpoints='outliers',
                    marker_color='lightcoral'
                ))
                fig_box.update_layout(
                    title=f"{coin_name} Price Box Plot (Outlier Detection)",
                    yaxis_title="Price (USD)",
                    height=300
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                # Calculate statistics
                price_std = coin_data.std()
                price_min = coin_data.min()
                price_max = coin_data.max()
                price_range = price_max - price_min
                price_skewness = coin_data.skew()
                price_kurtosis = coin_data.kurtosis()
                price_25 = coin_data.quantile(0.25)
                price_75 = coin_data.quantile(0.75)
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
                
                # Risk assessment with visual indicator
                if distance_from_mean > 2:
                    risk_color = "🔴"
                    risk_text = "High deviation"
                elif distance_from_mean > 1:
                    risk_color = "🟡"
                    risk_text = "Moderate deviation"
                else:
                    risk_color = "🟢"
                    risk_text = "Normal range"
                
                st.markdown("### Assessment")
                st.markdown(f"""
                **Position**: {percentile_rank:.0f}th percentile  
                **Status**: {risk_color} {risk_text} ({distance_from_mean:.1f}σ)
                """)
        
        with tab2:
            # Returns distribution analysis
            col1, col2 = st.columns([2, 1])
            
            with col1:
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
                
                # Box plot for returns distribution
                fig_box_returns = go.Figure()
                fig_box_returns.add_trace(go.Box(
                    y=coin_returns,
                    name=f"{coin_name}",
                    boxpoints='outliers',
                    marker_color='lightcoral'
                ))
                fig_box_returns.update_layout(
                    title=f"{coin_name} Returns Box Plot (Outlier Detection)",
                    yaxis_title="Daily Return",
                    height=300
                )
                st.plotly_chart(fig_box_returns, use_container_width=True)
            
            with col2:
                # Calculate statistics
                returns_std = coin_returns.std()
                returns_min = coin_returns.min()
                returns_max = coin_returns.max()
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
                    vol_color = "🔴"
                    vol_text = "Very high volatility"
                elif annualized_vol > 0.5:
                    vol_color = "🟡"
                    vol_text = "High volatility"
                else:
                    vol_color = "🟢"
                    vol_text = "Moderate volatility"
                
                # Sharpe ratio (assuming risk-free rate of 0)
                sharpe = returns_mean / returns_std * (252 ** 0.5) if returns_std > 0 else 0
                
                st.markdown("### Assessment")
                st.markdown(f"""
                **Volatility**: {vol_color} {vol_text}  
                **Annualized Vol**: {annualized_vol:.1%}  
                **Sharpe Ratio**: {sharpe:.2f}
                """)
        
        with tab3:
            # Technical indicators
            # Moving averages
            ma_20 = coin_data.rolling(20).mean()
            ma_50 = coin_data.rolling(50).mean()
            
            # RSI calculation
            delta = coin_data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD calculation
            ema_12 = coin_data.ewm(span=12).mean()
            ema_26 = coin_data.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            macd_signal = macd_line.ewm(span=9).mean()
            macd_histogram = macd_line - macd_signal
            
            fig_tech = go.Figure()
            
            # Price and moving averages
            fig_tech.add_trace(go.Scatter(
                x=coin_data.index,
                y=coin_data,
                mode='lines',
                name=f"{coin_name} Price",
                line=dict(width=2)
            ))
            
            fig_tech.add_trace(go.Scatter(
                x=ma_20.index,
                y=ma_20,
                mode='lines',
                name='MA-20',
                line=dict(width=1, dash='dash')
            ))
            
            fig_tech.add_trace(go.Scatter(
                x=ma_50.index,
                y=ma_50,
                mode='lines',
                name='MA-50',
                line=dict(width=1, dash='dot')
            ))
            
            fig_tech.update_layout(
                title=f"{coin_name} Technical Analysis",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=400
            )
            st.plotly_chart(fig_tech, use_container_width=True)
            
            # RSI chart
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=rsi.index,
                y=rsi,
                mode='lines',
                name='RSI',
                line=dict(width=2, color='orange')
            ))
            
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            
            fig_rsi.update_layout(
                title="Relative Strength Index (RSI)",
                xaxis_title="Date",
                yaxis_title="RSI",
                height=300,
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            # MACD chart
            fig_macd = go.Figure()
            
            # MACD line
            fig_macd.add_trace(go.Scatter(
                x=macd_line.index,
                y=macd_line,
                mode='lines',
                name='MACD Line',
                line=dict(width=2, color='blue')
            ))
            
            # Signal line
            fig_macd.add_trace(go.Scatter(
                x=macd_signal.index,
                y=macd_signal,
                mode='lines',
                name='Signal Line',
                line=dict(width=2, color='red')
            ))
            
            # Histogram
            fig_macd.add_trace(go.Bar(
                x=macd_histogram.index,
                y=macd_histogram,
                name='MACD Histogram',
                marker_color=['green' if x > 0 else 'red' for x in macd_histogram],
                opacity=0.7
            ))
            
            # Zero line
            fig_macd.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero Line")
            
            fig_macd.update_layout(
                title="MACD (Moving Average Convergence Divergence)",
                xaxis_title="Date",
                yaxis_title="MACD",
                height=350,
                showlegend=True
            )
            st.plotly_chart(fig_macd, use_container_width=True)

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
    
    # Display coin availability information
    if coin_launch_info:
        st.subheader("Cryptocurrency Availability")
        
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
        st.caption("Note: Pre-existing coins were available before the analysis start date. Their exact launch date is unknown.")
    
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
    
    # Check if cryptocurrencies are selected
    if not selected_symbols:
        st.warning("⚠️ Please select cryptocurrencies in the sidebar to run ML predictions.")
        st.stop()
    
    # Get cached data
    try:
        price_data, returns = get_cached_data(
            selected_symbols, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        
        # Check if we have data
        if price_data is None or price_data.empty:
            st.error("❌ No price data available for the selected period and cryptocurrencies.")
            st.info("💡 Try selecting a longer date range or different cryptocurrencies.")
            st.stop()
            
        # Display data info
        st.info(f"Loaded data for {len(selected_symbols)} cryptocurrencies from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({len(price_data)} days)")
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()
    
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
        try:
            features = create_features(prices)
            
            # Prepare target (future returns)
            target = prices.shift(-target_days) / prices - 1
            
            # Align features and target - remove NaN values
            valid_idx = features.index.intersection(target.index)
            X = features.loc[valid_idx]
            y = target.loc[valid_idx]
            
            # Remove rows with any NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask].fillna(0)
            y = y[mask].fillna(0)
            
            # Check if we have enough data
            if len(X) < max(train_window, 30):
                st.warning(f"Insufficient data for training. Need at least {max(train_window, 30)} days, got {len(X)}")
                return None, None, None
            
            # Use last train_window days for training, but leave space for future predictions
            available_for_training = len(X) - target_days
            if available_for_training < train_window:
                train_window = max(available_for_training, 30)
            
            X_train = X.iloc[-train_window-target_days:-target_days] if len(X) > target_days else X.iloc[-train_window:]
            y_train = y.iloc[-train_window-target_days:-target_days] if len(y) > target_days else y.iloc[-train_window:]
            
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
        
        # Train models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'Linear Regression': LinearRegression(),
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
                alpha=0.01,
                learning_rate_init=0.001,
                solver='adam',
                activation='relu',
                batch_size='auto',
                beta_1=0.9,
                beta_2=0.999,
                tol=1e-4,
                n_iter_no_change=20
            )
        }
        
        # Add XGBoost if available, otherwise use sklearn GradientBoosting as fallback
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
        else:
            models['Gradient Boosting'] = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
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
    
    if not all_predictions_data:
        st.error("❌ No successful predictions were generated for any of the selected cryptocurrencies.")
        st.info("💡 This could be due to:")
        st.write("""
        - Insufficient historical data (need at least 30-90 days)
        - Data quality issues
        - Selected date range too short
        - Technical issues with feature calculation
        """)
        st.info("🔧 Try:")
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
            # Define professional color palette for all models
            model_colors = {
                'Random Forest': '#2E8B57',      # Sea Green
                'Linear Regression': '#4169E1',  # Royal Blue  
                'Neural Network': '#FF6347',     # Tomato Red
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
        
        # Create a more readable table with all models
        table_data = []
        available_models = ['Random Forest', 'Linear Regression', 'Neural Network']
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
        available_models = ['Random Forest', 'Linear Regression', 'Neural Network']
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