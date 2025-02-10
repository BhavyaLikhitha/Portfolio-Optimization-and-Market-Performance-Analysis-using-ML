from datetime import datetime
from data_ingestion import fetch_sp500_tickers, clean_tickers, filter_valid_tickers, fetch_stock_data
from feature_engineering import compute_features
from clustering import perform_clustering
from portfolio_optimization import optimize_portfolio
from logger import get_logger
import pandas as pd

logger = get_logger(__name__)

def main():
    """Main script to orchestrate stock analysis and optimization."""
    logger.info("Starting main script")
    
    sp500_companies = fetch_sp500_tickers()
    tickers = clean_tickers(sp500_companies['Symbol'].tolist())
    
    start_date = "2018-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    valid_tickers = filter_valid_tickers(tickers, start_date, end_date)
    data = fetch_stock_data(valid_tickers, start_date, end_date)
    
    features_scaled_df = compute_features(data)
    clustered_data = perform_clustering(features_scaled_df)
    
    selected_cluster = 0  # Change as needed
    selected_stocks = clustered_data[clustered_data['hierarchical_cluster'] == selected_cluster].index
    cluster_returns = data[selected_stocks].pct_change().dropna()
    
    top_n = 100
    sharpe_ratios = (cluster_returns.mean() * 252) / (cluster_returns.std() + 1e-8)
    top_stocks = sharpe_ratios.nlargest(top_n).index
    filtered_returns = cluster_returns[top_stocks]
    
    optimized_weights = optimize_portfolio(filtered_returns)
    
    if optimized_weights is not None:
        portfolio_allocation = pd.Series(optimized_weights, index=filtered_returns.columns)
        print("Optimized Portfolio Allocation:")
        print(portfolio_allocation)
        logger.info("Complete execution done successfully")
    else:
        print("Portfolio optimization failed.")

if __name__ == "__main__":
    main()
