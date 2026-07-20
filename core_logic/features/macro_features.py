import pandas as pd
import numpy as np
df = pd.read_csv("data/processed/macro/cleaned_macro_data.csv")
print(df.columns)

def feature_engineering_macro(df:pd.DataFrame)->pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    macro_cols = [
        "consumer_sentiment","fed_funds_rate","gdp","industrial_production","inflation_cpi",
        "pce","retail_sales","treasury_10y","treasury_3m","unemployment_rate"]

    df["yield_curve"] = (df["treasury_10y"] -df["treasury_3m"])
    for col in macro_cols:
        for lag in [3, 6, 12]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        df[f"{col}_ewm6"] = (df[col].ewm(span=6,adjust=False).mean())
        df[f"{col}_ewm12"] = (df[col].ewm(span=12,adjust=False).mean())
                            
        df[f"{col}_pct_change"] = df[col].pct_change(fill_method=None)
        df[f"{col}_diff"] = df[col].diff()
        df[f"{col}_rolling3"] = (df[col].rolling(3).mean())
        df[f"{col}_rolling6"] = (df[col].rolling(6).mean())
        df[f"{col}_rolling12"] = (df[col].rolling(12).mean())
        df[f"{col}_rolling_std6"] = (df[col].rolling(6).std())
        df[f"{col}_rolling_std12"] = (df[col].rolling(12).std())
        df[f"{col}_rolling_std3"] = (df[col].rolling(3).std())
        rolling_mean = df[col].rolling(12).mean()
        rolling_std = df[col].rolling(12).std().replace(0,np.nan)
        
        df[f"{col}_zscore"] = ((df[col] - rolling_mean)/ rolling_std)
        df[f"{col}_momentum3"] = (df[col] - df[col].shift(3))
        df[f"{col}_momentum6"] = (df[col] - df[col].shift(6))
        
        
    df["yield_curve_inverted"] = (df["yield_curve"] < 0).astype(int)
    df["high_inflation"] = (df["inflation_cpi"] > 3).astype(int)
    df["high_unemployment"] = (df["unemployment_rate"] > 6).astype(int)

    df["real_interest_rate"] = (df["fed_funds_rate"]- df["inflation_cpi"])
    df["consumer_vs_inflation"] = (df["consumer_sentiment"]/ df["inflation_cpi"]+1e-8)
    df["growth_minus_inflation"] = (df["gdp"] - df["inflation_cpi"])
    df["retail_vs_unemployment"] = (df["retail_sales"]/ df["unemployment_rate"]+1e-8)

    df["gdp_up"] = (df["gdp_diff"] > 0).astype(int)
    df["inflation_up"] = (df["inflation_cpi_diff"] > 0).astype(int)
    df["rates_up"] = (df["fed_funds_rate_diff"] > 0).astype(int)
    df["economic_stress"] = (df["inflation_cpi_zscore"]+ df["unemployment_rate_zscore"]- df["consumer_sentiment_zscore"])
    df["yield_curve_change"] = (df["yield_curve"].diff())
    df["yield_curve_ma6"] = (df["yield_curve"].rolling(6).mean())

    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df.dropna(inplace=True)
    return df

df=feature_engineering_macro(df)
df.to_parquet("data/feature_engineered/macro/macro_features.parquet",index=False)
print("saved")