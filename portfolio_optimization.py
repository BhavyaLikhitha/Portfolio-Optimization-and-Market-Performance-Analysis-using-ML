import numpy as np
import pandas as pd
from scipy.optimize import minimize
from logger import get_logger

logger = get_logger(__name__)

def portfolio_performance(weights, returns):
    logger.info("Executing function: portfolio_performance")
    """Calculate portfolio return, volatility, and Sharpe ratio."""
    portfolio_return = np.sum(weights * returns.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
    return portfolio_return, portfolio_volatility, sharpe_ratio

def minimize_sharpe(weights, returns):
    logger.info("Executing function: minimize_sharpe")
    """Objective function to minimize negative Sharpe ratio."""
    return -portfolio_performance(weights, returns)[2]

def optimize_portfolio(returns):
    logger.info("Executing function: optimize_portfolio")
    """Optimize portfolio allocation using Mean-Variance optimization."""
    returns = returns.dropna(axis=1)
    
    num_assets = len(returns.columns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = [(0, 1) for _ in range(num_assets)]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    optimized = minimize(minimize_sharpe, initial_weights, args=(returns,), 
                         method='SLSQP', bounds=bounds, constraints=constraints)
    
    return optimized.x if optimized.success else None
