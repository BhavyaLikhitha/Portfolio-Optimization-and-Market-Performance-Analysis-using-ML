import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.manifold import TSNE
from datetime import datetime
from scipy.optimize import minimize
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
from logger import get_logger
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = get_logger(__name__)

def validate_data(features):
    """ Check if the dataset is empty before processing """
    if features.empty:
        logger.error("Feature set is empty. Ensure valid stock data is available before scaling.")
        raise ValueError("Feature set is empty. Cannot apply StandardScaler.")
    logger.info(f"Features DataFrame shape before scaling: {features.shape}")

def filter_valid_tickers(stock_data):
    """ Remove tickers with missing data """
    valid_tickers = [ticker for ticker in stock_data.columns if not stock_data[ticker].isna().all()]
    if not valid_tickers:
        logger.error("No valid tickers found after filtering. Check data sources.")
        raise ValueError("No valid tickers found. Ensure stock data is available.")
    return stock_data[valid_tickers]


# ==========================
# Step 1: Data Collection (Live Data Integration)
# ==========================

def fetch_sp500_tickers():
    logger.info("Executing function: fetch_sp500_tickers")
    """Fetch S&P 500 tickers from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(url)[0]
    return sp500_table[['Symbol', 'GICS Sector']]

def clean_tickers(tickers):
    logger.info("Executing function: clean_tickers")
    """Fix tickers with dot notation for Yahoo Finance."""
    return [t.replace('.', '-') for t in tickers]  # BRK.B → BRK-B

def filter_valid_tickers(tickers, start_date, end_date):
    logger.info("Executing function: filter_valid_tickers")
    """Check which tickers have valid data on Yahoo Finance."""
    valid_tickers = []
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(start=start_date, end=end_date)
            if not data.empty:
                valid_tickers.append(ticker)
            else:
                print(f"Skipping {ticker}: No data found.")
        except Exception as e:
            print(f"Skipping {ticker}: {e}")
    return valid_tickers

def fetch_stock_data(tickers, start_date, end_date):
    logger.info("Executing function: fetch_stock_data")
    """Download stock data for valid tickers."""
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # Check if 'Adj Close' exists in the returned data
    if 'Adj Close' in data:
        return data['Adj Close'].dropna(axis=1)  # Remove columns with NaN
    else:
        print("Warning: 'Adj Close' column not found in data. Returning raw data.")
        return data  # Return the whole DataFrame for debugging

# Fetch tickers and clean them
sp500_companies = fetch_sp500_tickers()
tickers = clean_tickers(sp500_companies['Symbol'].tolist())

# Define date range
start_date = "2018-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Validate tickers and fetch data
valid_tickers = filter_valid_tickers(tickers, start_date, end_date)
data = fetch_stock_data(valid_tickers, start_date, end_date)

# Display the first few rows
print(data.head())


# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# ==========================
# Step 2: Feature Engineering (Risk, Liquidity & Macro)
# ==========================
# Compute returns and volatility
returns = data.pct_change().dropna()
volatility = returns.rolling(window=20).std().dropna()

def compute_beta(stock_returns, market_returns):
    logger.info("Executing function: compute_beta")
    """Calculate Beta for each stock."""
    if stock_returns.isna().sum() > 0 or market_returns.isna().sum() > 0:
        return np.nan  # Handle missing values
    
    covariance = np.cov(stock_returns.dropna(), market_returns.dropna())[0, 1]
    market_variance = np.var(market_returns.dropna())

    return covariance / market_variance if market_variance > 0 else np.nan

# Approximate S&P 500 market return
sp500_returns = returns.mean(axis=1)
# Compute feature matrix
features = pd.DataFrame({

    "mean_return": returns.mean() * 252,
    "volatility": volatility.mean(),
    "cumulative_return": (data.iloc[-1] / data.iloc[0]) - 1,
    "sharpe_ratio": returns.mean() / (returns.std() + 1e-8),  # Avoid div-by-zero
    "sortino_ratio": returns.mean() / (returns[returns < 0].std() + 1e-8),  # Avoid div-by-zero
    "max_drawdown": (data / data.cummax() - 1).min(),
    "var_95": returns.quantile(0.05),
    "beta": [compute_beta(returns[ticker], sp500_returns) for ticker in returns.columns],
    "liquidity": data.mean(),  # Approximate by average price
})

# validate_data(features)

# Handle Inf and NaNs before scaling
features.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace Inf with NaN
features.dropna(inplace=True)  # Drop rows with NaN values

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
validate_data(features)

# Convert back to DataFrame
features_scaled_df = pd.DataFrame(scaled_features, index=features.index, columns=features.columns)

# Display first few rows
print(features_scaled_df.head())


# ==========================
# Step 3: Advanced Clustering (Hierarchical + GMM + DBSCAN)
# ==========================
# Use scaled features for clustering
X_scaled = features_scaled_df.copy()

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=10, linkage='ward')
X_scaled['hierarchical_cluster'] = hierarchical.fit_predict(features_scaled_df)

# Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=10, random_state=42)
X_scaled['gmm_cluster'] = gmm.fit_predict(features_scaled_df)

# DBSCAN Clustering (Density-based)
dbscan = DBSCAN(eps=1.5, min_samples=5)
X_scaled['dbscan_cluster'] = dbscan.fit_predict(features_scaled_df)

# Save Clustering Data
X_scaled.to_csv("advanced_clustered_data.csv")

# Display first few rows of results
print(X_scaled.head())


# import numpy as np
# from scipy.optimize import minimize


# ==========================
# Step 4: Portfolio Optimization (Mean-Variance)
# ==========================

def portfolio_performance(weights, returns):
    logger.info("Executing function: portfolio_performance")
    """Calculate portfolio return, volatility, and Sharpe ratio."""
    portfolio_return = np.sum(weights * returns.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
    # Avoid division by zero
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
    return portfolio_return, portfolio_volatility, sharpe_ratio



def minimize_sharpe(weights, returns):
    logger.info("Executing function: minimize_sharpe")
    """Objective function to minimize negative Sharpe ratio."""
    return -portfolio_performance(weights, returns)[2]  # Minimize negative Sharpe



def optimize_portfolio(returns):
    logger.info("Executing function: optimize_portfolio")
    """Optimize portfolio allocation using Mean-Variance optimization."""
    returns = returns.dropna(axis=1)  # Ensure valid returns data

    num_assets = len(returns.columns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = [(0, 1) for _ in range(num_assets)]  # Weight constraints (0 to 1)
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum of weights must be 1

    optimized = minimize(minimize_sharpe, initial_weights, args=(returns,), 
                         method='SLSQP', bounds=bounds, constraints=constraints)
    
    return optimized.x if optimized.success else None  # Return weights if optimization is successful



# Select the cluster (Change as needed)
selected_cluster = 0

# Get all stocks in the selected cluster
selected_stocks = X_scaled[X_scaled['hierarchical_cluster'] == selected_cluster].index
cluster_returns = returns[selected_stocks]  # Get returns for these stocks

# Compute Sharpe Ratio for ranking (Higher is better)
sharpe_ratios = (cluster_returns.mean() * 252) / (cluster_returns.std() + 1e-8)  # Avoid div-by-zero

# Select Top N Stocks (Changeable Parameter for Scalability)
top_n = 100  # Adjust this to change the number of optimized stocks
top_stocks = sharpe_ratios.nlargest(top_n).index  # Get top N stocks

# Use only the top stocks for portfolio optimization
filtered_returns = cluster_returns[top_stocks]


print(f"Number of stocks in Cluster {selected_cluster}: {len(selected_stocks)}")
print(f"Number of stocks with valid return data: {cluster_returns.shape[1]}")
print(len(filtered_returns))

# Optimize portfolio using only the top N selected stocks
optimized_weights = optimize_portfolio(filtered_returns)


# Display optimized portfolio weights for the selected top N stocks
portfolio_allocation = pd.Series(optimized_weights, index=filtered_returns.columns)
print("Optimized Portfolio Allocation:")
print(portfolio_allocation)


# Ensure clustering results exist
if 'hierarchical_cluster' not in X_scaled.columns:
    raise ValueError("Clustering results not found. Run Step 3 before visualization.")

# ==========================
# Dendrogram (Hierarchical Clustering for Filtered Stocks)
# ==========================

# Limit features_scaled_df to only filtered stocks
filtered_features_scaled_df = features_scaled_df.loc[filtered_returns.columns]

plt.figure(figsize=(12, 6))
dendrogram(linkage(filtered_features_scaled_df, method='ward'))
plt.title("Hierarchical Clustering Dendrogram (Filtered Stocks)")
plt.xlabel("Stocks")
plt.ylabel("Distance")
plt.show()


# ==========================
# t-SNE Visualization (Only for Selected Portfolio Stocks)
# ==========================

# ==========================
# Step 1: Ensure `filtered_returns` Uses Stock Names Instead of Dates
# ==========================
if isinstance(filtered_returns.index[0], pd.Timestamp):
    print("⚠ Warning: `filtered_returns.index` contains dates. Using `.columns` instead.")
    filtered_stocks = list(filtered_returns.columns)  # Use stock tickers
else:
    filtered_stocks = list(filtered_returns.index)  # Use index if it's stock names

# Debugging: Print stock counts before filtering
print(f"✅ Stocks in filtered_returns: {len(filtered_stocks)}")
print(f"✅ Stocks in X_scaled before filtering: {X_scaled.shape[0]}")

# ==========================
# Step 2: Filter `X_scaled` to Include Only Selected Portfolio Stocks
# ==========================
X_scaled_filtered = X_scaled.loc[filtered_stocks].reset_index()

# Debugging: Print stock count after filtering
print(f"✅ Stocks in X_scaled after filtering: {X_scaled_filtered.shape[0]}")

# ==========================
# Step 3: Ensure Only Numeric Data is Passed to t-SNE
# ==========================
# Print column data types for debugging
print("✅ Column types before t-SNE:")
print(X_scaled_filtered.dtypes)

# Drop non-numeric columns (e.g., Stock Names, Categorical Data)
X_scaled_numeric = X_scaled_filtered.select_dtypes(include=[np.number])

# Debugging: Print remaining numeric columns
print(f"✅ Columns used for t-SNE: {list(X_scaled_numeric.columns)}")

# ==========================
# Step 4: Apply t-SNE (Only If Enough Stocks Exist)
# ==========================
n_samples = X_scaled_numeric.shape[0]

if n_samples > 2:  # t-SNE requires at least 2 samples
    perplexity_value = min(30, max(2, n_samples - 1))  # Ensure valid perplexity

    tsne = TSNE(n_components=2, perplexity=perplexity_value, learning_rate='auto', random_state=42)
    tsne_results = tsne.fit_transform(X_scaled_numeric)  # Use only numeric columns

    # Add t-SNE results to the DataFrame
    X_scaled_filtered['tsne_1'] = tsne_results[:, 0]
    X_scaled_filtered['tsne_2'] = tsne_results[:, 1]

    # ==========================
    # Step 5: Interactive Scatter Plot (Filtered Stocks Only)
    # ==========================
    fig = px.scatter(X_scaled_filtered, x='tsne_1', y='tsne_2', 
                     color=X_scaled_filtered['hierarchical_cluster'].astype(str) if 'hierarchical_cluster' in X_scaled_filtered.columns else None, 
                     title="t-SNE Visualization of Selected Portfolio Stocks", 
                     hover_name="Ticker")  # Fix: Replace "Stock" with "Ticker"

    # Show the plot
    fig.show(renderer="browser")

else:
    print(f"⚠ Not enough samples for t-SNE. Only {n_samples} stocks selected.")


# ==========================
# Portfolio Allocation Visualization
# ==========================

# Ensure `optimized_weights` exist
if optimized_weights is not None:
    # Convert MultiIndex to a flat index (if needed)
    stock_names = [col[0] if isinstance(col, tuple) else col for col in filtered_returns.columns]  # Updated

    # Create allocation DataFrame
    allocation_df = pd.DataFrame({'Stock': stock_names, 'Weight': optimized_weights})
    allocation_df = allocation_df.sort_values(by='Weight', ascending=False)

    # Plot the allocation
    plt.figure(figsize=(10, 6))
    # sns.barplot(x='Weight', y='Stock', data=allocation_df, palette="coolwarm")
    sns.barplot(x='Weight', y='Stock', data=allocation_df, hue='Stock', palette="coolwarm", legend=False)

    plt.title("Optimized Portfolio Allocation")
    plt.xlabel("Allocation Weight")
    plt.ylabel("Stock")
    plt.show()

else:
    print("Portfolio optimization failed. No visualization available.")


# ==========================
# Save Stock Name, Price & Weight to CSV
# ==========================

# Ensure `optimized_weights` exist before proceeding
if optimized_weights is not None:
    # Convert MultiIndex to a flat index (if needed)
    stock_names = [col[0] if isinstance(col, tuple) else col for col in filtered_returns.columns]  

    # Create DataFrame with stock allocation
    allocation_df = pd.DataFrame({
        'Stock': stock_names,
        'Weight': optimized_weights
    })

    # Merge with stock prices from X_scaled_filtered
    stock_prices = X_scaled_filtered[['Ticker', 'Price']].set_index("Ticker")  # Ensure "Ticker" and "Price" exist
    final_df = allocation_df.merge(stock_prices, left_on="Stock", right_index=True, how="left")

    # Save to CSV
    csv_filename = "optimized_portfolio_simple.csv"
    final_df.to_csv(csv_filename, index=False)

    print(f"✅ Stock Name, Price & Weight saved to {csv_filename}")

else:
    print("❌ Portfolio optimization failed. No CSV file was created.")


# ==========================
# Print Filtered Stock Names
# ==========================

# Ensure that `filtered_returns` contains the selected stocks
filtered_stocks = list(filtered_returns.columns)  # Extract stock names

# Print the list of filtered stocks
print("✅ Filtered Stocks for Optimization:")
for stock in filtered_stocks:
    print(stock)

