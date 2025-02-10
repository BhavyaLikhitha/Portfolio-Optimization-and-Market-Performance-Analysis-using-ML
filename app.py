import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
from sklearn.manifold import TSNE

from data_ingestion import fetch_sp500_tickers, clean_tickers, fetch_stock_data
from feature_engineering import compute_features
from clustering import perform_clustering
from portfolio_optimization import optimize_portfolio

# # ==========================
# # 📌 Streamlit Dashboard UI
# # ==========================

# st.set_page_config(page_title="Stock Portfolio Dashboard", layout="wide")

# st.title("📈 Stock Portfolio Optimization Dashboard")

# # Sidebar Inputs
# st.sidebar.header("🔧 Select Parameters")
# start_date = st.sidebar.date_input("Start Date", datetime(2018, 1, 1))
# end_date = st.sidebar.date_input("End Date", datetime.today())

# # Fetch and Clean Data
# st.sidebar.subheader("📥 Fetching Data...")
# sp500_companies = fetch_sp500_tickers()
# tickers = clean_tickers(sp500_companies['Symbol'].tolist())

# # Load Stock Data
# st.sidebar.text("Fetching stock data... Please wait!")
# data = fetch_stock_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
# st.sidebar.success("✅ Data Fetched Successfully!")

# # Feature Engineering
# st.subheader("📊 Feature Engineering & Clustering")
# features_scaled_df = compute_features(data)
# clustered_data = perform_clustering(features_scaled_df)

# # Cluster Selection
# selected_cluster = st.sidebar.selectbox("🔎 Select a Cluster", clustered_data['hierarchical_cluster'].unique())
# selected_stocks = clustered_data[clustered_data['hierarchical_cluster'] == selected_cluster].index
# cluster_returns = data[selected_stocks].pct_change().dropna()

# # Select Top Performing Stocks
# top_n = st.sidebar.slider("📌 Select Top N Stocks (By Sharpe Ratio)", min_value=10, max_value=200, value=100)
# sharpe_ratios = (cluster_returns.mean() * 252) / (cluster_returns.std() + 1e-8)
# top_stocks = sharpe_ratios.nlargest(top_n).index
# filtered_returns = cluster_returns[top_stocks]


# ==========================
# 📌 Streamlit Dashboard UI
# ==========================
st.set_page_config(page_title="Stock Portfolio Dashboard", layout="wide")

st.title("📈 Stock Portfolio Optimization Dashboard")

# Sidebar Inputs
st.sidebar.header("🔧 Select Parameters")
start_date = st.sidebar.date_input("Start Date", datetime(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())

# Fetch and Clean Data
st.sidebar.subheader("📥 Fetching Data...")
sp500_companies = fetch_sp500_tickers()
tickers = clean_tickers(sp500_companies['Symbol'].tolist())

# ==========================
# 📌 Add Error Handling for Data Fetching
# ==========================
try:
    st.sidebar.text("Fetching stock data... Please wait!")
    data = fetch_stock_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if data.empty:
        raise ValueError("Error: Yahoo Finance returned no data. Try adjusting the date range or tickers.")

    st.sidebar.success("✅ Data Fetched Successfully!")

    # Feature Engineering
    st.subheader("📊 Feature Engineering & Clustering")
    features_scaled_df = compute_features(data)
    
    # Ensure Features Are Not Empty
    if features_scaled_df.empty:
        raise ValueError("Error: Computed features are empty. Try changing the date range.")

    clustered_data = perform_clustering(features_scaled_df)

    # Cluster Selection
    selected_cluster = st.sidebar.selectbox("🔎 Select a Cluster", clustered_data['hierarchical_cluster'].unique())
    selected_stocks = clustered_data[clustered_data['hierarchical_cluster'] == selected_cluster].index
    cluster_returns = data[selected_stocks].pct_change().dropna()

    # Select Top Performing Stocks
    top_n = st.sidebar.slider("📌 Select Top N Stocks (By Sharpe Ratio)", min_value=10, max_value=200, value=100)
    sharpe_ratios = (cluster_returns.mean() * 252) / (cluster_returns.std() + 1e-8)
    top_stocks = sharpe_ratios.nlargest(top_n).index
    filtered_returns = cluster_returns[top_stocks]

except ValueError as e:
    st.error(f"❌ Data Processing Error: {e}")
    st.stop()  # Stop execution if data fetching or feature processing fails


# ==========================
# 📌 Visualization: Cluster Distribution
# ==========================
st.subheader("📊 Cluster Distribution")
fig_cluster_dist = px.histogram(clustered_data, x="hierarchical_cluster", title="Stock Distribution Across Clusters", nbins=10)
st.plotly_chart(fig_cluster_dist, use_container_width=True)

# ==========================
# 📌 Visualization: Stock Price Trends
# ==========================
st.subheader("📈 Stock Price Trends")
selected_stock = st.selectbox("📊 Choose a Stock to View Price Trends", top_stocks)
fig_stock_trend = px.line(data[selected_stock], title=f"{selected_stock} Price Trend", labels={"value": "Stock Price"})
st.plotly_chart(fig_stock_trend, use_container_width=True)

# ==========================
# 📌 Visualization: Risk-Return Scatter Plot
# ==========================
st.subheader("⚖️ Risk-Return Comparison")
risk_return_df = pd.DataFrame({"Stock": top_stocks, "Annual Return": sharpe_ratios[top_stocks], "Volatility": cluster_returns[top_stocks].std()})
fig_risk_return = px.scatter(risk_return_df, x="Volatility", y="Annual Return", text="Stock", title="Risk vs. Return", color="Annual Return")
fig_risk_return.update_traces(textposition="top center")
st.plotly_chart(fig_risk_return, use_container_width=True)

# ==========================
# 📌 Visualization: t-SNE Cluster Visualization
# ==========================
st.subheader("🔍 t-SNE Visualization of Stock Clusters")
tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', random_state=42)
tsne_results = tsne.fit_transform(features_scaled_df)
clustered_data["tsne_1"] = tsne_results[:, 0]
clustered_data["tsne_2"] = tsne_results[:, 1]
fig_tsne = px.scatter(clustered_data, x="tsne_1", y="tsne_2", color=clustered_data["hierarchical_cluster"].astype(str), title="t-SNE Clustering Visualization", hover_name=clustered_data.index)
st.plotly_chart(fig_tsne, use_container_width=True)

# ==========================
# 📌 Portfolio Optimization
# ==========================
st.subheader("⚖️ Portfolio Optimization")
optimized_weights = optimize_portfolio(filtered_returns)

if optimized_weights is not None:
    portfolio_allocation = pd.Series(optimized_weights, index=filtered_returns.columns)

    # ==========================
    # 📌 Portfolio Allocation Bar Chart
    # ==========================
    st.subheader("📊 Portfolio Allocation")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=portfolio_allocation.index, y=portfolio_allocation.values, palette="coolwarm", ax=ax)
    ax.set_title("Optimized Portfolio Allocation")
    ax.set_xticklabels(portfolio_allocation.index, rotation=90)
    st.pyplot(fig)

    # ==========================
    # 📌 Pie Chart for Allocation
    # ==========================
    st.subheader("🍰 Portfolio Distribution")
    fig_pie = px.pie(portfolio_allocation, values=portfolio_allocation.values, names=portfolio_allocation.index, title="Stock Allocation")
    st.plotly_chart(fig_pie, use_container_width=True)

    # ==========================
    # 📌 Performance Metrics
    # ==========================
    st.subheader("📈 Portfolio Performance Metrics")
    portfolio_return = (filtered_returns.mean() * 252).sum()
    portfolio_volatility = (filtered_returns.std() * np.sqrt(252)).sum()
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Annual Return", f"{portfolio_return:.2f}%")
    col2.metric("📉 Volatility", f"{portfolio_volatility:.2f}%")
    col3.metric("⚖️ Sharpe Ratio", f"{sharpe_ratio:.2f}")

else:
    st.error("❌ Portfolio Optimization Failed. Try Adjusting the Inputs.")

st.sidebar.success("✅ Dashboard Ready!")
