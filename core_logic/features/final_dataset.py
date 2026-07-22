import pandas as pd
import numpy as np

df_stocks=pd.read_parquet("data/feature_engineered/stocks/stock_features.parquet")
df_news=pd.read_parquet("data/feature_engineered/news/news_features.parquet")
df_macro=pd.read_parquet("data/feature_engineered/macro/macro_features.parquet")
print(df_stocks["Date"].dtype)
print(df_news["Date"].dtype)  #since news data has timezone utc convert it into local 
print(df_macro["Date"].dtype)

# def merge_dfs(df_stocks,df_news,df_macro):
    # for df in [df_stocks, df_news, df_macro]:
    #     df.columns =(df.columns.str.strip().str.lower().str.replace(" ", "_"))

    # df_news["date"]=(pd.to_datetime(df_news["date"]).dt.tz_localize(None))
    # final_df=df_stocks.merge(df_news,on=["date","ticker"],how="left").merge(df_macro,on="date",how="left")
    # print("Stock last date :", df_stocks["date"].max())
    # print("Macro last date :", df_macro["date"].max())

    # print("\nStock sample dates")
    # print(df_stocks["date"].tail())

    # print("\nMacro sample dates")
    # print(df_macro["date"].tail())

    # print("\nMacro values after merge (before fillna)")
    # cols = [
    #     "fed_funds_rate",
    #     "inflation_cpi",
    #     "gdp",
    #     "yield_curve",
    #     "yield_curve_ma6"
    # ]
    # print(final_df[cols].tail())
    # print(final_df[cols].isna().sum())
    # final_df.fillna(0,inplace=True)
    # return final_df

def merge_dfs(df_stocks, df_news, df_macro):
    for df in [df_stocks, df_news, df_macro]:
        df.columns = (
            df.columns.str.strip()
                     .str.lower()
                     .str.replace(" ", "_")
        )

    df_news["date"] = pd.to_datetime(df_news["date"]).dt.tz_localize(None)

    # Merge stocks and news
    final_df = df_stocks.merge(
        df_news,
        on=["date", "ticker"],
        how="left"
    )

    # Sort before merge_asof
    final_df = final_df.sort_values("date")
    df_macro = df_macro.sort_values("date")

    # final_df["date"] = pd.to_datetime(final_df["date"])
    # df_macro["date"] = pd.to_datetime(df_macro["date"])
    final_df["date"] = (
    pd.to_datetime(final_df["date"])
      .astype("datetime64[ns]")
)

    df_macro["date"] = (
        pd.to_datetime(df_macro["date"])
        .astype("datetime64[ns]")
    )
    print(final_df["date"].dtype)
    print(df_macro["date"].dtype)
    # Merge latest available macro data
    final_df = pd.merge_asof(
        final_df,
        df_macro,
        on="date",
        direction="backward"
    )

    final_df.fillna(0, inplace=True)

    return final_df

final_df=merge_dfs(df_stocks,df_news,df_macro)
final_df.to_parquet("data/final_dataset.parquet")
print(df_stocks.shape,df_macro.shape,df_news.shape)
print(final_df.shape)
print(final_df.columns)
print("Done")