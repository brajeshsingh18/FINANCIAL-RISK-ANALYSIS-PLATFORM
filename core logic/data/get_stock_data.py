# Source	                                        Purpose	                        Library/API
# Yahoo Finance	                                Historical stock prices	            yfinance
# NewsAPI (or Finnhub/Alpha Vantage News)	        Financial news	                requests
# FRED	                                        Macroeconomic indicators	        fredapi


# Sector	Number of Companies
# Technology	15
# Financial Services	12
# Healthcare	12
# Consumer Discretionary	10
# Consumer Staples	8
# Energy	8
# Industrials	10
# Communication Services	8
# Utilities	5
# Real Estate	5
# Materials	7

"""
Download historical stock market data from Yahoo Finance.

This module downloads daily OHLCV (Open, High, Low, Close, Volume)
data for a list of stock tickers and stores each ticker as a CSV file
under data/raw/stocks/.
"""

from pathlib import Path
import logging
from typing import List

import pandas as pd
import yfinance as yf
from tqdm import tqdm

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TICKERS = [
    # =========================
    # Technology (10)
    # =========================
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "META",   # Meta
    "NVDA",   # NVIDIA
    "AMD",    # AMD
    "AVGO",   # Broadcom
    "ORCL",   # Oracle
    "CRM",    # Salesforce

    # =========================
    # Financial Services (7)
    # =========================
    "JPM",    # JPMorgan Chase
    "BAC",    # Bank of America
    "WFC",    # Wells Fargo
    "GS",     # Goldman Sachs
    "MS",     # Morgan Stanley
    "AXP",    # American Express
    "BLK",    # BlackRock

    # =========================
    # Healthcare (6)
    # =========================
    "JNJ",    # Johnson & Johnson
    "PFE",    # Pfizer
    "MRK",    # Merck
    "ABBV",   # AbbVie
    "LLY",    # Eli Lilly
    "UNH",    # UnitedHealth

    # =========================
    # Consumer Discretionary (5)
    # =========================
    "TSLA",   # Tesla
    "HD",     # Home Depot
    "MCD",    # McDonald's
    "NKE",    # Nike
    "SBUX",   # Starbucks

    # =========================
    # Consumer Staples (4)
    # =========================
    "WMT",    # Walmart
    "PG",     # Procter & Gamble
    "KO",     # Coca-Cola
    "PEP",    # PepsiCo

    # =========================
    # Energy (3)
    # =========================
    "XOM",    # Exxon Mobil
    "CVX",    # Chevron
    "COP",    # ConocoPhillips

    # =========================
    # Industrials (4)
    # =========================
    "CAT",    # Caterpillar
    "GE",     # GE Aerospace
    "HON",    # Honeywell
    "UPS",    # United Parcel Service

    # =========================
    # Communication Services (3)
    # =========================
    "NFLX",   # Netflix
    "DIS",    # Disney
    "TMUS",   # T-Mobile

    # =========================
    # Semiconductors (3)
    # =========================
    "QCOM",   # Qualcomm
    "MU",     # Micron
    "TXN",    # Texas Instruments

    # =========================
    # Utilities (2)
    # =========================
    "NEE",    # NextEra Energy
    "DUK",    # Duke Energy

    # =========================
    # Real Estate (1)
    # =========================
    "PLD"      # Prologis
]

START_DATE = "2018-01-01"
END_DATE = None  # Downloads until today
RAW_DATA_DIR = Path("data/raw/stocks")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

def create_directory() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_stock_data(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df.reset_index(inplace=True)
    return df


def save_stock_data(df: pd.DataFrame, ticker: str) -> None:
    output_path = RAW_DATA_DIR / f"{ticker}.csv"
    df.to_csv(output_path, index=False)


def download_all_stocks(tickers: List[str]) -> None:
    success = 0
    for ticker in tqdm(tickers, desc="Downloading Stocks"):
        try:
            logger.info("Downloading %s...", ticker)
            df = download_stock_data(ticker)
            save_stock_data(df, ticker)
            logger.info("Saved %s.csv (%d rows)", ticker, len(df))
            success += 1
        except Exception as e:
            logger.error("Failed to download %s : %s", ticker, e)

    logger.info(
        "Download Complete (%d/%d successful)",
        success,
        len(tickers),
    )

def main() -> None:
    create_directory()
    download_all_stocks(TICKERS)

if __name__ == "__main__":
    main()