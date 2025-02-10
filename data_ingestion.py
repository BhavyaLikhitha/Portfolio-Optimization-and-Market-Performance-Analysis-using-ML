import pandas as pd
import yfinance as yf
from datetime import datetime
from logger import get_logger

logger = get_logger(__name__)

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
    
    if 'Adj Close' in data:
        return data['Adj Close'].dropna(axis=1)  # Remove columns with NaN
    else:
        print("Warning: 'Adj Close' column not found in data. Returning raw data.")
        return data
