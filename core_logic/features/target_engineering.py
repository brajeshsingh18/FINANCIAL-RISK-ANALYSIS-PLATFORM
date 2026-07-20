import pandas as pd
import numpy as np

df=pd.read_parquet("data/final_dataset.parquet")

def adding_target(df:pd.DataFrame)->pd.DataFrame:
    df=df.copy()
    df=df.sort_values(["ticker","date"])
    df["future_return"]=(df.groupby("ticker")["close"].shift(-1)/df['close']-1)
    df.dropna(subset=['future_return'],inplace=True)
    return df

df=adding_target(df)
df.to_parquet("data/model_data.parquet",index=False)