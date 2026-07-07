import pandas as pd
df = pd.read_csv("data/processed/macro/cleaned_macro_data.csv")
print(df.columns)
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)
macro_cols = [
    "consumer_sentiment","fed_funds_rate","gdp","industrial_production","inflation_cpi",
    "pce","retail_sales","treasury_10y","treasury_3m","unemployment_rate"]

for col in macro_cols:
    df[f"{col}_lag1"] = df[col].shift(1)
    df[f"{col}_pct_change"] = df[col].pct_change()
    df[f"{col}_diff"] = df[col].diff()
    df[f"{col}_rolling3"] = (df[col].rolling(3).mean())
    df[f"{col}_rolling_std3"] = (df[col].rolling(3).std())

df["yield_curve"] = (df["treasury_10y"] -df["treasury_3m"])
df = df.dropna().reset_index(drop=True)
df.to_parquet("data/feature_engineered/macro/macro_features.parquet",index=False)