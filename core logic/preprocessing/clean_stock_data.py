"""
Clean and preprocess historical stock data.

This module reads all stock CSV files from data/raw/stocks/,
validates them, cleans them, merges them into a single dataframe,
and saves the cleaned dataset.
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

RAW_DATA_DIR = Path("data/raw/stocks")
OUTPUT_DIR = Path("data/processed/stocks")
OUTPUT_FILE = OUTPUT_DIR / "cleaned_stock_data.csv"
REQUIRED_COLUMNS = ["Date","Open","High","Low","Close","Volume",]

def create_directory():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_stock_file(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)

def validate_columns(df: pd.DataFrame, file_name: str):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"{file_name} missing columns : {missing}"
        )

def clean_dataframe(df: pd.DataFrame,ticker: str,) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df.drop_duplicates(subset=["Date"],inplace=True,)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    numeric_cols = ["Open","High","Low","Close","Volume",]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["Open"] > 0)& (df["High"] > 0)& (df["Low"] > 0)& (df["Close"] > 0)& (df["Volume"] >= 0)]
    df["Ticker"] = ticker
    return df

def merge_all_files() -> pd.DataFrame:
    all_data = []
    files = sorted(RAW_DATA_DIR.glob("*.csv"))
    for file in tqdm(files, desc="Cleaning Stocks"):
        ticker = file.stem
        df = load_stock_file(file)
        validate_columns(df, file.name)
        df = clean_dataframe(df, ticker)
        all_data.append(df)
    merged_df = pd.concat(all_data,ignore_index=True,)
    merged_df.sort_values(["Ticker", "Date"],inplace=True,)
    return merged_df

def save_dataframe(df: pd.DataFrame):
    df.to_csv(OUTPUT_FILE,index=False,)

def main():
    create_directory()
    df = merge_all_files()
    save_dataframe(df)
    print("=" * 50)
    print("Stock preprocessing completed.")
    print(f"Rows : {len(df)}")
    print(f"Companies : {df['Ticker'].nunique()}")
    print("=" * 50)

if __name__ == "__main__":
    main()