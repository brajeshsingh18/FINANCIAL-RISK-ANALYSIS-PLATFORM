# Indicator	                    FRED Code	      Why
# Federal Funds Rate	        FEDFUNDS	    Interest rates
# CPI	                        CPIAUCSL	    Inflation
# Unemployment  Rate	        UNRATE	        Labor market
# GDP	                        GDP	            Economic growth
# 10-Year Treasury Yield	    DGS10	        Bond market
# 3-Month Treasury Bill	        TB3MS	        Short-term interest rates
# Consumer Sentiment	        UMCSENT	        Investor confidence
# Industrial Production	        INDPRO	        Manufacturing activity
# Retail Sales	                RSAFS	        Consumer spending
# Personal Consumption Expenditure	PCE	        Consumer demand


"""
Download macroeconomic indicators from FRED.

This module downloads important US macroeconomic indicators and stores
each indicator as a CSV file under data/raw/macro/.
"""

from pathlib import Path
from typing import Dict
import os
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()

API_KEY = os.getenv("FRED_API_KEY")
START_DATE = "2015-01-01"
OUTPUT_DIR = Path("data/raw/macro")

MACRO_INDICATORS: Dict[str, str] = {
    "fed_funds_rate": "FEDFUNDS",
    "inflation_cpi": "CPIAUCSL",
    "unemployment_rate": "UNRATE",
    "gdp": "GDP",
    "treasury_10y": "DGS10",
    "treasury_3m": "TB3MS",
    "consumer_sentiment": "UMCSENT",
    "industrial_production": "INDPRO",
    "retail_sales": "RSAFS",
    "pce": "PCE",
}

fred = Fred(api_key=API_KEY)
def create_directory() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_macro_data(series_id: str,start_date=START_DATE) -> pd.DataFrame:
    data = fred.get_series(series_id, observation_start=start_date)
    if data.empty:
        raise ValueError(f"No data found for {series_id}")
    df = data.reset_index()
    df.columns = ["Date", "Value"]
    return df

def save_macro_data(df: pd.DataFrame, indicator_name: str) -> None:
    output_path = OUTPUT_DIR / f"{indicator_name}.csv"
    df.to_csv(output_path, index=False)


def download_all_macro(indicators: Dict[str, str]) -> None:
    success = 0
    for indicator_name, series_id in tqdm(indicators.items(),desc="Downloading Macro Data",):
        try:
            print(f"Downloading {indicator_name}...")
            df = fetch_macro_data(series_id)
            save_macro_data(df, indicator_name)
            print(f"Saved {indicator_name}.csv")
            success += 1

        except Exception as e:
            print(f"Failed: {indicator_name}")
            print(e)

    print(f"\nDownloaded {success}/{len(indicators)} indicators.")

def main() -> None:
    create_directory()
    download_all_macro(MACRO_INDICATORS)

if __name__ == "__main__":
    main()