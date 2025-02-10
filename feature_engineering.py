import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from logger import get_logger

logger = get_logger(__name__)

def compute_beta(stock_returns, market_returns):
    logger.info("Executing function: compute_beta")
    """Calculate Beta for each stock."""
    if stock_returns.isna().sum() > 0 or market_returns.isna().sum() > 0:
        return np.nan  # Handle missing values
    
    covariance = np.cov(stock_returns.dropna(), market_returns.dropna())[0, 1]
    market_variance = np.var(market_returns.dropna())

    return covariance / market_variance if market_variance > 0 else np.nan

def validate_data(features):
    """ Check if the dataset is empty before processing """
    if features.empty:
        logger.error("Feature set is empty. Ensure valid stock data is available before scaling.")
        raise ValueError("Feature set is empty. Cannot apply StandardScaler.")
    logger.info(f"Features DataFrame shape before scaling: {features.shape}")

def compute_features(data):
    """Compute financial features for stock analysis."""
    returns = data.pct_change().dropna()
    volatility = returns.rolling(window=20).std().dropna()
    sp500_returns = returns.mean(axis=1)
    
    features = pd.DataFrame({
        "mean_return": returns.mean() * 252,
        "volatility": volatility.mean(),
        "cumulative_return": (data.iloc[-1] / data.iloc[0]) - 1,
        "sharpe_ratio": returns.mean() / (returns.std() + 1e-8),
        "sortino_ratio": returns.mean() / (returns[returns < 0].std() + 1e-8),
        "max_drawdown": (data / data.cummax() - 1).min(),
        "var_95": returns.quantile(0.05),
        "beta": [compute_beta(returns[ticker], sp500_returns) for ticker in returns.columns],
        "liquidity": data.mean()
    })
    
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.dropna(inplace=True)
    validate_data(features)
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    features_scaled_df = pd.DataFrame(scaled_features, index=features.index, columns=features.columns)
    
    return features_scaled_df
