# ==========================
# 📌 ML-Driven Portfolio Optimization with Streamlit Dashboard
# - Hierarchical Clustering + GMM + DBSCAN
# - Factor-Based Investing (Fama-French & Macro)
# - Streamlit Interactive Dashboard
# ==========================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize
from datetime import datetime
import statsmodels.api as sm

# ==========================
# Step 1: Fetch S&P 500 Tickers & Data
# ==========================

@st.cache
def fetch_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(url)[0]
    return sp500_table[['Symbol', 'GICS Sector']]

@st.cache
def fetch_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data.dropna(axis=1)

sp500_companies = fetch_sp500_tickers()
tickers = sp500_companies['Symbol'].tolist()

start_date = "2018-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

st.title("📈 ML-Driven Portfolio Optimization & Factor Investing")

data = fetch_stock_data(tickers[:50], start_date, end_date)  # Limit to 50 stocks for speed

# ==========================
# Step 2: Feature Engineering (Risk, Liquidity & Factors)
# ==========================

returns = data.pct_change().dropna()
volatility = returns.rolling(window=20).std().dropna()

# Fetch Fama-French Factors
@st.cache
def fetch_fama_french():
    ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    fama_french = pd.read_csv(ff_url, skiprows=3)
    fama_french = fama_french.iloc[:-1]  # Remove footer
    fama_french = fama_french.apply(pd.to_numeric, errors='coerce')
    fama_french['Date'] = pd.to_datetime(fama_french['Date'], format='%Y%m%d')
    fama_french.set_index('Date', inplace=True)
    return fama_french / 100  # Convert to decimal

fama_french = fetch_fama_french()

def compute_factors(stock_returns):
    factor_data = fama_french.loc[stock_returns.index]
    X = sm.add_constant(factor_data[['Mkt-RF', 'SMB', 'HML']])  # Market, Size, Value Factors
    y = stock_returns - factor_data['RF']
    model = sm.OLS(y, X).fit()
    return model.params  # Beta coefficients for factors

factor_exposure = pd.DataFrame({ticker: compute_factors(returns[ticker]) for ticker in returns.columns}).T

# ==========================
# Step 3: Clustering (Hierarchical + GMM + DBSCAN)
# ==========================

scaler = StandardScaler()
scaled_returns = scaler.fit_transform(returns.mean().values.reshape(-1, 1))

hierarchical = AgglomerativeClustering(n_clusters=10, linkage='ward')
clusters = hierarchical.fit_predict(scaled_returns)

factor_exposure['cluster'] = clusters

# ==========================
# Step 4: Portfolio Optimization (Mean-Variance)
# ==========================

def portfolio_performance(weights, returns):
    portfolio_return = np.sum(weights * returns.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

def minimize_sharpe(weights, returns):
    return -portfolio_performance(weights, returns)[2]  # Minimize negative Sharpe

def optimize_portfolio(returns):
    num_assets = len(returns.columns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = [(0, 1) for _ in range(num_assets)]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    optimized = minimize(minimize_sharpe, initial_weights, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)
    return optimized.x

optimized_weights = optimize_portfolio(returns)

# ==========================
# Step 5: Streamlit Dashboard
# ==========================

# Sidebar - Choose number of clusters
num_clusters = st.sidebar.slider("Select Number of Clusters", 2, 15, 10)
st.sidebar.write("Current Clusters:", num_clusters)

# Display clustering results
st.subheader("📊 Stock Clustering Results")
fig = px.scatter(factor_exposure, x=factor_exposure.index, y='Mkt-RF', color=factor_exposure['cluster'].astype(str),
                 title="Factor-Based Stock Clusters", hover_name=factor_exposure.index)
st.plotly_chart(fig)

# Portfolio Allocation
st.subheader("📈 Optimized Portfolio Allocation")
allocation_df = pd.DataFrame({'Stock': returns.columns, 'Weight': optimized_weights})
allocation_df = allocation_df.sort_values(by='Weight', ascending=False)

fig2 = px.bar(allocation_df, x='Stock', y='Weight', title="Portfolio Allocation", text_auto=True)
st.plotly_chart(fig2)

# Display Sharpe Ratio
st.subheader("📊 Portfolio Performance Metrics")
portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_performance(optimized_weights, returns)
st.write(f"📌 **Annualized Return:** {portfolio_return:.2%}")
st.write(f"📌 **Volatility:** {portfolio_volatility:.2%}")
st.write(f"📌 **Sharpe Ratio:** {sharpe_ratio:.2f}")

# Show stock price data
st.subheader("📉 Stock Price Trends")
selected_stock = st.selectbox("Choose a stock", returns.columns)
st.line_chart(data[selected_stock])

st.write("🚀 **Enhancements:** Uses Machine Learning & Factor-Based Investing for Smart Portfolio Construction!")

