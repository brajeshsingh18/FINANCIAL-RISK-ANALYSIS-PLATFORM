"""
Clean and preprocess financial news data.

This module reads all news CSV files from data/raw/news/,
validates them, cleans them, merges them into a single dataframe,
and saves the cleaned dataset.
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RAW_DATA_DIR = Path("data/raw/news")
OUTPUT_DIR = Path("data/processed/news")

OUTPUT_FILE = OUTPUT_DIR / "cleaned_news_data.csv"

REQUIRED_COLUMNS = [
    "published_at",
    "ticker",
    "title",
    "description",
    "source",
    "url",
]

# ---------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------


def create_directory() -> None:
    """
    Create output directory.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_news_file(file_path: Path) -> pd.DataFrame:
    """
    Load one news CSV.
    """
    return pd.read_csv(file_path)


def validate_columns(df: pd.DataFrame, file_name: str) -> None:
    """
    Validate required columns.
    """
    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{file_name} missing columns: {missing}"
        )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean news dataframe.
    """

    # Convert datetime
    df["published_at"] = pd.to_datetime(
        df["published_at"],
        errors="coerce",
    )

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Remove duplicate articles
    df.drop_duplicates(
        subset=["title", "published_at"],
        inplace=True,
    )

    # Remove rows without title/date
    df.dropna(
        subset=["title", "published_at"],
        inplace=True,
    )

    # Fill optional columns
    df["description"] = df["description"].fillna("")
    df["source"] = df["source"].fillna("Unknown")
    df["url"] = df["url"].fillna("")

    # Remove blank titles
    df = df[
        df["title"].str.strip() != ""
    ]

    # Sort chronologically
    df.sort_values(
        "published_at",
        inplace=True,
    )

    return df


def merge_all_files() -> pd.DataFrame:
    """
    Merge all news files.
    """

    all_data = []

    files = sorted(RAW_DATA_DIR.glob("*.csv"))

    for file in tqdm(files, desc="Cleaning News"):

        df = load_news_file(file)

        validate_columns(df, file.name)

        df = clean_dataframe(df)

        all_data.append(df)

    merged_df = pd.concat(
        all_data,
        ignore_index=True,
    )

    merged_df.sort_values(
        ["ticker", "published_at"],
        inplace=True,
    )

    merged_df.reset_index(
        drop=True,
        inplace=True,
    )

    return merged_df


def save_dataframe(df: pd.DataFrame) -> None:
    """
    Save cleaned dataframe.
    """
    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


def main() -> None:

    create_directory()

    df = merge_all_files()

    save_dataframe(df)

    print("=" * 50)
    print("News preprocessing completed.")
    print(f"Articles : {len(df)}")
    print(f"Companies : {df['ticker'].nunique()}")
    print("=" * 50)


if __name__ == "__main__":
    main()