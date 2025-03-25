import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import io
import requests
import zipfile

# Set page configuration
st.set_page_config(page_title="Carhart Four-Factor Model", layout="wide")

# Title and description
st.title("Carhart Four-Factor Model Analysis")
st.write("""
This app implements the Carhart Four-Factor Model which extends the Fama-French model by adding a momentum factor.
The model helps estimate expected returns based on exposures to market risk, size, value, and momentum factors.
""")

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

# Read stock list from Excel
@st.cache_data
def load_stock_list():
    try:
        stock_data = pd.read_excel("stocklist.xlsx", sheet_name=None, engine='openpyxl')
        return stock_data
    except Exception as e:
        st.error(f"Error loading stock list: {e}")
        return None

stock_data = load_stock_list()

if stock_data is None:
    st.stop()

# Get available sheets
available_sheets = list(stock_data.keys())

# User selects sheet
selected_sheet = st.sidebar.selectbox("Select Stock List", available_sheets)

# Get symbols from selected sheet
symbols = stock_data[selected_sheet]['Symbol'].tolist()

# User selects stock
selected_stock = st.sidebar.selectbox("Select Stock", symbols)

# Date range selection
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)  # Default 5 years

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", start_date)
with col2:
    end_date = st.date_input("End Date", end_date)

# Convert to string for yfinance
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Improved factor data download function
@st.cache_data
def download_factor_data(start_date, end_date):
    try:
        # Fama-French 3 factors + Momentum (Carhart 4 factors)
        ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
        mom_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
        
        # Download and process Fama-French 3 factors
        ff_response = requests.get(ff_url)
        with zipfile.ZipFile(io.BytesIO(ff_response.content)) as z:
            with z.open(z.namelist()[0]) as f:
                ff_data = pd.read_csv(f, skiprows=3, index_col=0)
                # Find where the copyright notice begins
                ff_data = ff_data[~ff_data.index.astype(str).str.contains("Copyright")]
                ff_data.index = pd.to_datetime(ff_data.index, format='%Y%m%d')
                ff_data.columns = ['Mkt-RF', 'SMB', 'HML', 'RF']
        
        # Download and process Momentum factor
        mom_response = requests.get(mom_url)
        with zipfile.ZipFile(io.BytesIO(mom_response.content)) as z:
            with z.open(z.namelist()[0]) as f:
                mom_data = pd.read_csv(f, skiprows=13, index_col=0)
                # Find where the copyright notice begins
                mom_data = mom_data[~mom_data.index.astype(str).str.contains("Copyright")]
                mom_data.index = pd.to_datetime(mom_data.index, format='%Y%m%d')
                mom_data.columns = ['Mom']
        
        # Merge factors
        all_factors = ff_data.join(mom_data, how='inner')
        
        # Convert to decimal
        all_factors = all_factors / 100
        
        # Filter by date range
        all_factors = all_factors.loc[start_date:end_date]
        
        return all_factors
    except Exception as e:
        st.error(f"Error downloading factor data: {str(e)}")
        return None

# Download stock data - MODIFIED TO USE EXACT SYMBOL FROM EXCEL
@st.cache_data
def download_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if stock.empty:
            st.warning(f"No data found for {ticker}. Please try another stock.")
            return None
        return stock['Close']
    except Exception as e:
        st.error(f"Error downloading stock data for {ticker}: {str(e)}")
        return None

# Main analysis function - FIXED TYPO IN 'Excess_Return'
def run_carhart_model(stock_returns, factors):
    # Merge stock returns with factors
    merged_data = pd.concat([stock_returns, factors], axis=1).dropna()
    
    # Calculate excess returns
    merged_data['Excess_Return'] = merged_data['Stock_Return'] - merged_data['RF']
    
    # Prepare variables for regression
    X = merged_data[['Mkt-RF', 'SMB', 'HML', 'Mom']]
    X = sm.add_constant(X)  # Adds intercept term
    y = merged_data['Excess_Return']
    
    # Run regression
    model = sm.OLS(y, X).fit()
    
    return model, merged_data

# Run analysis when button is clicked
if st.sidebar.button("Run Analysis"):
    st.subheader(f"Analysis for {selected_stock}")
    
    with st.spinner("Downloading data and running analysis..."):
        # Download factor data
        factors = download_factor_data(start_date_str, end_date_str)
        
        if factors is None:
            st.error("Failed to download factor data. Please try again later.")
            st.stop()
        
        # Download stock data - USING EXACT SYMBOL FROM EXCEL
        stock_prices = download_stock_data(selected_stock, start_date_str, end_date_str)
        
        if stock_prices is None:
            st.error(f"Failed to download data for {selected_stock}. Please try another stock.")
            st.stop()
        
        # Calculate daily returns
        stock_returns = stock_prices.pct_change().dropna()
        stock_returns.name = "Stock_Return"
        
        # Ensure we have matching dates
        common_dates = stock_returns.index.intersection(factors.index)
        stock_returns = stock_returns[common_dates]
        factors = factors.loc[common_dates]
        
        if len(stock_returns) < 10 or len(factors) < 10:
            st.error("Not enough overlapping data points for analysis. Please adjust date range.")
            st.stop()
        
        # Run Carhart model
        model, merged_data = run_carhart_model(stock_returns, factors)
        
        # Display results
        st.subheader("Regression Results")
        st.text(model.summary())
        
        # Extract coefficients
        coefficients = model.params
        st.subheader("Factor Exposures")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Market Risk (Beta)", f"{coefficients['Mkt-RF']:.4f}")
        col2.metric("Size (SMB)", f"{coefficients['SMB']:.4f}")
        col3.metric("Value (HML)", f"{coefficients['HML']:.4f}")
        col4.metric("Momentum (Mom)", f"{coefficients['Mom']:.4f}")
        
        # Visualizations
        st.subheader("Visualizations")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(stock_prices)
        ax.set_title(f"{selected_stock} Price Chart")
        ax.set_ylabel("Price")
        ax.grid(True)
        st.pyplot(fig)
        
        # Plot factor exposures
        fig, ax = plt.subplots(figsize=(10, 6))
        factors_to_plot = ['Mkt-RF', 'SMB', 'HML', 'Mom']
        coefficients[factors_to_plot].plot(kind='bar', ax=ax)
        ax.set_title("Factor Exposures")
        ax.set_ylabel("Coefficient Value")
        ax.grid(True)
        st.pyplot(fig)
        
        # Plot cumulative returns
        fig, ax = plt.subplots(figsize=(10, 6))
        (1 + merged_data['Stock_Return']).cumprod().plot(ax=ax, label='Stock')
        (1 + merged_data['RF']).cumprod().plot(ax=ax, label='Risk-Free')
        ax.set_title("Cumulative Returns")
        ax.set_ylabel("Growth of ₹1")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        # Display raw data
        st.subheader("Raw Data")
        st.dataframe(merged_data.head())

# Add some information about the model
st.sidebar.markdown("""
**About the Carhart Four-Factor Model:**
- Extends the Fama-French three-factor model by adding momentum
- Factors:
  1. Market risk (Mkt-RF)
  2. Size (SMB - Small Minus Big)
  3. Value (HML - High Minus Low)
  4. Momentum (Mom)
""")
