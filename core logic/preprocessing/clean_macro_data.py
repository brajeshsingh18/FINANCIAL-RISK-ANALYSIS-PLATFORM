"""
Clean and preprocess macroeconomic data.

This module reads all macroeconomic CSV files from data/raw/macro/,
validates them, cleans them, merges them into a single dataframe,
and saves the cleaned dataset.
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RAW_DATA_DIR = Path("data/raw/macro")
OUTPUT_DIR = Path("data/processed/macro")

OUTPUT_FILE = OUTPUT_DIR / "cleaned_macro_data.csv"

REQUIRED_COLUMNS = [
    "Date",
    "Value",
]

# ---------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------


def create_directory() -> None:
    """
    Create processed directory.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_macro_file(file_path: Path) -> pd.DataFrame:
    """
    Load one macroeconomic csv.
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


def clean_dataframe(
    df: pd.DataFrame,
    indicator_name: str,
) -> pd.DataFrame:
    """
    Clean macroeconomic dataframe.
    """

    # Convert date
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    # Convert Value to numeric
    df["Value"] = pd.to_numeric(
        df["Value"],
        errors="coerce",
    )

    # Remove duplicates
    df.drop_duplicates(
        subset=["Date"],
        inplace=True,
    )

    # Remove missing values
    df.dropna(inplace=True)

    # Sort by date
    df.sort_values(
        "Date",
        inplace=True,
    )

    # Rename Value column
    df.rename(
        columns={
            "Value": indicator_name
        },
        inplace=True,
    )

    return df


def merge_all_files() -> pd.DataFrame:
    """
    Merge all macroeconomic indicators.
    """

    files = sorted(
        RAW_DATA_DIR.glob("*.csv")
    )

    merged_df = None

    for file in tqdm(
        files,
        desc="Cleaning Macro Data",
    ):

        indicator_name = file.stem

        df = load_macro_file(file)

        validate_columns(df, file.name)

        df = clean_dataframe(
            df,
            indicator_name,
        )

        if merged_df is None:

            merged_df = df

        else:

            merged_df = pd.merge(
                merged_df,
                df,
                on="Date",
                how="outer",
            )

    merged_df.sort_values(
        "Date",
        inplace=True,
    )

    merged_df.ffill(inplace=True)

    merged_df.bfill(inplace=True)

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
    print("Macro preprocessing completed.")
    print(f"Rows : {len(df)}")
    print(f"Indicators : {len(df.columns)-1}")
    print("=" * 50)


if __name__ == "__main__":
    main()