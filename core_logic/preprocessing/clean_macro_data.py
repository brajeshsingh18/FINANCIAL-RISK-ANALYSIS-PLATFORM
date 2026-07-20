from pathlib import Path
import pandas as pd
from tqdm import tqdm
RAW_DATA_DIR = Path("data/raw/macro")
OUTPUT_DIR = Path("data/processed/macro")
OUTPUT_FILE = OUTPUT_DIR / "cleaned_macro_data.csv"
REQUIRED_COLUMNS = ["Date","Value",]

def create_directory() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_macro_file(df: pd.DataFrame | None = None,file_path: Path | None = None) -> pd.DataFrame: #just manipulating the logic of providing inputs to include the inference pipeline here
    if df is not None:
        return df
    return pd.read_csv(file_path)

def validate_columns(df: pd.DataFrame, file_name: str) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"{file_name} missing columns: {missing}"
        )

def clean_dataframe(df: pd.DataFrame,indicator_name: str,) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"],errors="coerce",)
    df["Value"] = pd.to_numeric(df["Value"],errors="coerce",)
    df.drop_duplicates(subset=["Date"],inplace=True,)
    df.dropna(inplace=True)
    df.sort_values("Date",inplace=True,)
    df.rename(columns={"Value": indicator_name},inplace=True,)
    return df


def merge_all_files(macro_dfs: dict[str, pd.DataFrame] | None = None,) -> pd.DataFrame:
    merged_df = None
    if macro_dfs is None:
        iterator = ((file.stem, load_macro_file(file_path=file)) for file in sorted(RAW_DATA_DIR.glob("*.csv")))
    else:
        iterator = macro_dfs.items()

    for indicator_name, df in tqdm(iterator, desc="Cleaning Macro Data"):
        validate_columns(df, indicator_name)
        df = clean_dataframe(df.copy(),indicator_name,)
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df,df,on="Date",how="outer",)

    merged_df.sort_values("Date", inplace=True)
    merged_df.ffill(inplace=True)
    merged_df.bfill(inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df


def save_dataframe(df: pd.DataFrame) -> None:
    df.to_csv(OUTPUT_FILE,index=False,)


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