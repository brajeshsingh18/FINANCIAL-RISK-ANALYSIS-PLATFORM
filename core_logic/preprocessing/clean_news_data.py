from pathlib import Path
import pandas as pd
from tqdm import tqdm

RAW_DATA_DIR = Path("data/raw/news")
OUTPUT_DIR = Path("data/processed/news")
OUTPUT_FILE = OUTPUT_DIR / "cleaned_news_data.csv"
REQUIRED_COLUMNS = ["published_at","ticker","title","description","source","url",]

def create_directory() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_news_file(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)


def validate_columns(df: pd.DataFrame, file_name: str) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{file_name} missing columns: {missing}")

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce",)

    df.drop_duplicates(inplace=True)
    df.drop_duplicates(subset=["title", "published_at"],inplace=True,)
    df.dropna(
        subset=["title", "published_at"],
        inplace=True,
    )
    df["description"] = df["description"].fillna("")
    df["source"] = df["source"].fillna("Unknown")
    df["url"] = df["url"].fillna("")
    df = df[
        df["title"].str.strip() != ""
    ]
    df.sort_values("published_at",inplace=True,)
    return df

def preprocess_news_data(df:pd.DataFrame,ticker:str)-> pd.DataFrame:
    validate_columns(df,ticker)
    return clean_dataframe(df.copy())

def merge_all_files() -> pd.DataFrame:
    all_data = []
    files = sorted(RAW_DATA_DIR.glob("*.csv"))
    for file in tqdm(files, desc="Cleaning News"):
        df = load_news_file(file)
        validate_columns(df, file.name)
        df = clean_dataframe(df)
        all_data.append(df)

    merged_df = pd.concat(all_data,ignore_index=True,)
    merged_df.sort_values(["ticker", "published_at"],inplace=True,)
    merged_df.reset_index(drop=True,inplace=True,)
    return merged_df

def save_dataframe(df: pd.DataFrame) -> None:
    df.to_csv(OUTPUT_FILE,index=False,)

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