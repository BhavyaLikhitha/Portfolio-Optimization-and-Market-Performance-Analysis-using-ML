import requests
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER if not available
nltk.download('vader_lexicon')

def fetch_news_data(ticker):
    """Fetch latest news articles related to a stock ticker."""
    API_KEY = "8838dccc234a49328144ac8680c8f65f"  # Replace with your API Key
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        return [article["title"] for article in articles[:10]]  # Get latest 10 headlines
    else:
        print(f"⚠ Failed to fetch news for {ticker}")
        return []

def compute_sentiment_scores(ticker):
    """Calculate average sentiment score for a stock based on news headlines."""
    sia = SentimentIntensityAnalyzer()
    headlines = fetch_news_data(ticker)

    if not headlines:
        return 0  # If no news available, return neutral sentiment

    sentiment_scores = [sia.polarity_scores(headline)["compound"] for headline in headlines]
    return sum(sentiment_scores) / len(sentiment_scores)  # Average sentiment score

def get_sentiment_for_stocks(stock_list):
    """Get sentiment scores for multiple stocks."""
    sentiment_data = {stock: compute_sentiment_scores(stock) for stock in stock_list}
    return pd.DataFrame(list(sentiment_data.items()), columns=["Stock", "Sentiment_Score"])
