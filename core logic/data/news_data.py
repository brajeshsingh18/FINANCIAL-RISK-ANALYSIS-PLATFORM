"""
Download financial news from MarketAux.

This module downloads company-specific financial news for a list of
stock tickers and stores each ticker as a CSV file under
data/raw/news/.
"""

from pathlib import Path
from typing import List
import os

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("MARKETAUX_API_KEY")

TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "AMD", "AVGO", "ORCL", "CRM",
    "JPM", "BAC", "WFC", "GS", "MS",
    "AXP", "BLK", "JNJ", "PFE", "MRK",
    "ABBV", "LLY", "UNH", "TSLA", "HD",
    "MCD", "NKE", "SBUX", "WMT", "PG",
    "KO", "PEP", "XOM", "CVX", "COP",
    "CAT", "GE", "HON", "UPS", "NFLX",
    "DIS", "TMUS", "QCOM", "MU", "TXN",
    "NEE", "DUK", "PLD"
]

OUTPUT_DIR = Path("data/raw/news")

BASE_URL = "https://api.marketaux.com/v1/news/all"

# ---------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------


def create_directory() -> None:
    """
    Create output directory.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_news(ticker: str) -> pd.DataFrame:
    """
    Download news for a single ticker.

    Parameters
    ----------
    ticker : str

    Returns
    -------
    pd.DataFrame
    """

    params = {
        "symbols": ticker,
        "language": "en",
        "limit": 100,
        "api_token": API_KEY,
    }

    response = requests.get(BASE_URL, params=params, timeout=30)

    response.raise_for_status()

    articles = response.json().get("data", [])

    if not articles:
        raise ValueError(f"No news found for {ticker}")

    rows = []

    for article in articles:

        rows.append(
            {
                "published_at": article.get("published_at"),
                "ticker": ticker,
                "title": article.get("title"),
                "description": article.get("description"),
                "source": article.get("source"),
                "url": article.get("url"),
            }
        )

    return pd.DataFrame(rows)


def save_news(df: pd.DataFrame, ticker: str) -> None:
    """
    Save dataframe.
    """

    output_path = OUTPUT_DIR / f"{ticker}.csv"

    df.to_csv(output_path, index=False)


def download_all_news(tickers: List[str]) -> None:
    """
    Download news for all tickers.
    """

    success = 0

    for ticker in tqdm(tickers, desc="Downloading News"):

        try:

            print(f"Downloading {ticker}...")

            df = fetch_news(ticker)

            save_news(df, ticker)

            print(f"Saved {ticker}.csv")

            success += 1

        except Exception as e:

            print(f"Failed : {ticker}")

            print(e)

    print(f"\nDownloaded {success}/{len(tickers)} companies.")


def main():

    create_directory()

    download_all_news(TICKERS)


if __name__ == "__main__":
    main()