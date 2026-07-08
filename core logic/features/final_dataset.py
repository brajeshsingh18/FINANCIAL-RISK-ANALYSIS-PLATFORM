import pandas as pd
import numpy as np

df_stocks=pd.read_parquet("data/feature_engineered/stocks/stock_features.parquet")
df_news=pd.read_parquet("data/feature_engineered/news/news_features.parquet")
df_macro=pd.read_parquet("data/feature_engineered/macro/macro_features.parquet")
print(df_stocks["Date"].dtype)
print(df_news["Date"].dtype)  #since news data has timezone utc convert it into local 
print(df_macro["Date"].dtype)

for df in [df_stocks, df_news, df_macro]:
    df.columns =(df.columns.str.strip().str.lower().str.replace(" ", "_"))

df_news["date"]=(pd.to_datetime(df_news["date"]).dt.tz_localize(None))

final_df=df_stocks.merge(df_news,on=["date","ticker"],how="left").merge(df_macro,on="date",how="left")
final_df.fillna(0,inplace=True)
final_df.to_parquet("data/final_dataset.parquet")
print(df_stocks.shape,df_macro.shape,df_news.shape)
print(final_df.shape)
print(final_df.columns)