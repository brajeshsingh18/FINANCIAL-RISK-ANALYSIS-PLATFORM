from finbert_sentiment import predict_sentiment
import pandas as pd
import numpy as np
from pathlib import Path
df=pd.read_parquet("data/feature_engineered/news/news_preprocessed.parquet")
print(df.columns.tolist())
print(df.head())

sentiment_df = pd.DataFrame(df.apply(predict_sentiment, axis=1).tolist())
df = pd.concat([df, sentiment_df], axis=1)
print(df.columns.tolist())

df.to_parquet("data/feature_engineered/news/news_sentiment.parquet",index=False)

df["published_at"] = pd.to_datetime(df["published_at"])
df["Date"] = df["published_at"].dt.normalize()

news_features = (
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
news_features["Std_Sentiment"] = (news_features["Std_Sentiment"].fillna(0))

news_features["Positive_Ratio"] = (news_features["Positive_Count"]/ news_features["News_Count"])
news_features["Negative_Ratio"] = (news_features["Negative_Count"]/ news_features["News_Count"])
news_features["Neutral_Ratio"] = (news_features["Neutral_Count"]/ news_features["News_Count"])


Path("data/feature_engineered/news").mkdir(
    parents=True,
    exist_ok=True
)
news_features.to_parquet(
    "data/feature_engineered/news/news_features.parquet",
    index=False
)
