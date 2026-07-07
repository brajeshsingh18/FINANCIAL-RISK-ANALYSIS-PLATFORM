from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "stocks" / "cleaned_stock_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "feature_engineered" / "stocks"
OUTPUT_FILE = OUTPUT_DIR / "stock_features.parquet"

WARMUP_COLUMNS = ["RSI_14", "MACD", "ATR_14"]


def load_data():
    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df = df.set_index(["Ticker", "Date"])
    return df


def price_features(df):
    g = df.groupby(level="Ticker")
    df["Prev_Close"] = g["Close"].shift(1)
    df["Daily_Return"] = g["Close"].pct_change()
    df["Log_Return"] = np.where(
        df["Prev_Close"] > 0,
        np.log(df["Close"] / df["Prev_Close"]),
        np.nan,
    )
    df["Gap"] = df["Open"] - df["Prev_Close"]
    df["Open_Close_Diff"] = df["Close"] - df["Open"]
    df["High_Low_Spread"] = df["High"] - df["Low"]
    df["Price_Range_Pct"] = (df["High"] - df["Low"]) / df["Close"]
    return df


def indicator_features(df):
    g_close = df.groupby(level="Ticker")["Close"]
    df["SMA_5"] = g_close.transform(lambda x: ta.sma(x, length=5))
    df["SMA_20"] = g_close.transform(lambda x: ta.sma(x, length=20))
    df["SMA_50"] = g_close.transform(lambda x: ta.sma(x, length=50))
    df["EMA_12"] = g_close.transform(lambda x: ta.ema(x, length=12))
    df["EMA_26"] = g_close.transform(lambda x: ta.ema(x, length=26))
    df["RSI_14"] = g_close.transform(lambda x: ta.rsi(x, length=14))
    df["ROC_10"] = g_close.transform(lambda x: ta.roc(x, length=10))
    df["Rolling_STD_20"] = (
        df.groupby(level="Ticker")["Daily_Return"]
        .transform(lambda x: x.rolling(20).std())
    )
    df["Volume_SMA20"] = (
        df.groupby(level="Ticker")["Volume"]
        .transform(lambda x: x.rolling(20).mean())
    )
    df["Relative_Volume"] = df["Volume"] / df["Volume_SMA20"]
    return df


def add_macd(group):
    macd = ta.macd(group["Close"].reset_index(drop=True))
    group = group.copy()
    group["MACD"] = macd["MACD_12_26_9"].values
    group["MACD_SIGNAL"] = macd["MACDs_12_26_9"].values
    group["MACD_HIST"] = macd["MACDh_12_26_9"].values
    return group


def add_bbands(group):
    bb = ta.bbands(group["Close"].reset_index(drop=True), length=20)
    group = group.copy()
    group["BB_UPPER"] = bb["BBU_20_2.0_2.0"].values
    group["BB_MIDDLE"] = bb["BBM_20_2.0_2.0"].values
    group["BB_LOWER"] = bb["BBL_20_2.0_2.0"].values
    group["BB_WIDTH"] = group["BB_UPPER"] - group["BB_LOWER"]
    return group


def add_atr_obv(df):
    pieces = []
    for _ticker, group in df.groupby(level="Ticker", sort=False):
        group = group.copy()
        group["ATR_14"] = ta.atr(
            group["High"], group["Low"], group["Close"], length=14
        ).values
        group["OBV"] = ta.obv(group["Close"], group["Volume"]).values
        pieces.append(group)
    return pd.concat(pieces).sort_index()


def build_features(df):
    df = price_features(df)
    df = indicator_features(df)
    df = df.groupby(level="Ticker", group_keys=False).apply(add_macd)
    df = df.groupby(level="Ticker", group_keys=False).apply(add_bbands)
    df = add_atr_obv(df)
    df = df.drop(columns=["Prev_Close"])
    df = df.dropna(subset=WARMUP_COLUMNS)
    df = df.reset_index()
    return df


def save_features(df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)


def main():
    df = load_data()
    features = build_features(df)
    save_features(features)

    print("=" * 50)
    print("Stock feature engineering completed.")
    print(f"Rows: {len(features)}")
    print(f"Tickers: {features['Ticker'].nunique()}")
    print(f"Columns: {len(features.columns)}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    main()
