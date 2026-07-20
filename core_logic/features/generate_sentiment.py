from finbert_sentiment import predict_sentiment
import pandas as pd
import numpy as np
from pathlib import Path
df=pd.read_parquet("data/feature_engineered/news/news_preprocessed.parquet")
print(df.columns.tolist())
print(df.head())

def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    sentiment_df = pd.DataFrame(df.apply(predict_sentiment, axis=1).tolist())
    return pd.concat([df, sentiment_df], axis=1)

def aggrigate_daily_news(df:pd.DataFrame)->pd.DataFrame:
    df=df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"])
    df["Date"] = df["published_at"].dt.normalize()
    return (
        df.groupby(["Date", "ticker"])
        .agg(
            News_Count=("sentiment_score", "count"),

            Avg_Sentiment=("sentiment_score", "mean"),
            Max_Sentiment=("sentiment_score", "max"),
            Min_Sentiment=("sentiment_score", "min"),
            Std_Sentiment=("sentiment_score", "std"),

            Avg_Confidence=("confidence", "mean"),

            Positive_Count=("sentiment", lambda x: (x == "positive").sum()),
            Negative_Count=("sentiment", lambda x: (x == "negative").sum()),
            Neutral_Count=("sentiment", lambda x: (x == "neutral").sum()),
        )
        .reset_index()
    )

def adding_features(news_features:pd.DataFrame)->pd.DataFrame:
    news_features["Std_Sentiment"] = (news_features["Std_Sentiment"].fillna(0))

    news_features["Positive_Ratio"] = (news_features["Positive_Count"]/ news_features["News_Count"])
    news_features["Negative_Ratio"] = (news_features["Negative_Count"]/ news_features["News_Count"])
    news_features["Neutral_Ratio"] = (news_features["Neutral_Count"]/ news_features["News_Count"])

    news_features = news_features.sort_values(["ticker", "Date"])
    g = news_features.groupby("ticker")

    news_features["Sentiment_3D"] = (
        g["Avg_Sentiment"].transform(lambda x: x.rolling(3, min_periods=1).mean()))
    news_features["Sentiment_7D"] = (
        g["Avg_Sentiment"].transform(lambda x: x.rolling(7, min_periods=1).mean()))
    news_features["Sentiment_14D"] = (
        g["Avg_Sentiment"].transform(lambda x: x.rolling(14, min_periods=1).mean()))

    news_features["News_Count_3D"] = (g["News_Count"].transform(lambda x: x.rolling(3, min_periods=1).sum()))
    news_features["News_Count_7D"] = (g["News_Count"].transform(lambda x: x.rolling(7, min_periods=1).sum()))
    news_features["News_Count_14D"] = (g["News_Count"].transform(lambda x: x.rolling(14, min_periods=1).sum()))

    news_features["Sentiment_Change"] = (g["Avg_Sentiment"].diff())
    news_features["Sentiment_Pct_Change"] = (g["Avg_Sentiment"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan))

    news_features["Sentiment_STD_7D"] = (g["Avg_Sentiment"].transform(lambda x: x.rolling(7, min_periods=1).std()))
    news_features["Sentiment_STD_14D"] = (g["Avg_Sentiment"].transform(lambda x: x.rolling(14, min_periods=1).std()))

    news_features["Positive_Streak"] = ((news_features["Positive_Ratio"] > 0.6).astype(int))
    news_features["Negative_Streak"] = ((news_features["Negative_Ratio"] > 0.6).astype(int))

    rolling_news = (g["News_Count"].transform(lambda x: x.rolling(20, min_periods=5).mean()))
    news_features["News_Surprise"] = (news_features["News_Count"]- rolling_news)

    news_features["Confidence_STD_7D"] = (g["Avg_Confidence"].transform(lambda x: x.rolling(7, min_periods=1).std()))
    news_features["Weighted_Sentiment"] = (news_features["Avg_Sentiment"]* news_features["News_Count"])

    news_features["Positive_Intensity"] = (news_features["Positive_Count"]* news_features["Avg_Confidence"])
    news_features["Negative_Intensity"] = (news_features["Negative_Count"]* news_features["Avg_Confidence"])

    news_features["EWM_Sentiment"] = (g["Avg_Sentiment"].transform(lambda x: x.ewm(span=7).mean()))

    news_features = news_features.fillna(0)
    
    return news_features

def feature_engineering_news(df:pd.DataFrame)->pd.DataFrame:
    df=df.copy()
    df=add_sentiment(df)
    # df.to_parquet("data/feature_engineered/news/news_sentiment.parquet",index=False)  used only while training
    df=aggrigate_daily_news(df)
    df=adding_features(df)
    return df

news_features=feature_engineering_news(df)
Path("data/feature_engineered/news").mkdir(
    parents=True,
    exist_ok=True
)
news_features.to_parquet(
    "data/feature_engineered/news/news_features.parquet",
    index=False
)
