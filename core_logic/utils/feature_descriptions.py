import joblib 
from typing import Dict

features=joblib.load("models/classificaion_feature_columns.pkl")
# print(features)

BASE_FEATURES: Dict[str, str] = {
    "adj_close": "Adjusted closing stock price",
    "close": "Closing stock price",
    "open": "Opening stock price",
    "high": "Highest trading price during the session",
    "low": "Lowest trading price during the session",
    "volume": "Total number of shares traded",

    "daily_return": "One-day percentage return",
    "log_return": "Daily logarithmic return",
    "return_5d": "Five-day percentage return",
    "return_10d": "Ten-day percentage return",
    "return_20d": "Twenty-day percentage return",
    "return_60d": "Sixty-day percentage return",

    "gap": "Difference between today's opening price and the previous closing price",
    "open_close_diff": "Difference between opening and closing prices",
    "high_low_spread": "Difference between the day's highest and lowest prices",
    "price_range_pct": "Daily trading range as a percentage of the closing price",
    "price_position_20": "Position of the closing price within the 20-day trading range",
    "drawdown": "Percentage decline from the highest closing price reached so far",

    "sma_5": "5-day Simple Moving Average",
    "sma_20": "20-day Simple Moving Average",
    "sma_50": "50-day Simple Moving Average",
    "ema_12": "12-day Exponential Moving Average",
    "ema_26": "26-day Exponential Moving Average",

    "dist_sma5": "Distance from the 5-day moving average",
    "dist_sma20": "Distance from the 20-day moving average",
    "dist_sma50": "Distance from the 50-day moving average",
    "dist_ema12": "Distance from the 12-day exponential moving average",
    "dist_ema26": "Distance from the 26-day exponential moving average",
    "ema_diff": "Difference between the 12-day and 26-day exponential moving averages",
    "ema_ratio": "Ratio of the 12-day to the 26-day exponential moving average",

    "rsi_14": "14-day Relative Strength Index",
    "roc_10": "10-day Rate of Change",
    "macd": "Moving Average Convergence Divergence",
    "macd_signal": "MACD Signal Line",
    "macd_hist": "MACD Histogram",
    "adx_14": "14-day Average Directional Index",

    "rolling_std_5": "5-day rolling standard deviation of daily returns",
    "rolling_std_10": "10-day rolling standard deviation of daily returns",
    "rolling_std_20": "20-day rolling standard deviation of daily returns",
    "rolling_std_60": "60-day rolling standard deviation of daily returns",
    "ewma_vol_20": "20-day exponentially weighted volatility",
    "atr_14": "14-day Average True Range",
    "rolling_sharpe_20": "20-day rolling Sharpe Ratio",

    "volume_sma20": "20-day average trading volume",
    "relative_volume": "Current trading volume relative to its 20-day average",
    "volume_zscore": "Standardized trading volume relative to recent history",
    "obv": "On-Balance Volume",

    "bb_upper": "Upper Bollinger Band",
    "bb_middle": "Middle Bollinger Band",
    "bb_lower": "Lower Bollinger Band",
    "bb_width": "Width of the Bollinger Bands",

    "rsi_above70": "Indicator that RSI is above the overbought threshold",
    "rsi_below30": "Indicator that RSI is below the oversold threshold",

    "news_count": "Number of news articles",
    "avg_sentiment": "Average FinBERT sentiment score",
    "max_sentiment": "Highest FinBERT sentiment score",
    "min_sentiment": "Lowest FinBERT sentiment score",
    "std_sentiment": "Variation in FinBERT sentiment scores",
    "avg_confidence": "Average confidence of sentiment predictions",

    "positive_count": "Number of positive news articles",
    "negative_count": "Number of negative news articles",
    "neutral_count": "Number of neutral news articles",

    "positive_ratio": "Proportion of positive news articles",
    "negative_ratio": "Proportion of negative news articles",
    "neutral_ratio": "Proportion of neutral news articles",

    "sentiment_3d": "3-day average news sentiment",
    "sentiment_7d": "7-day average news sentiment",
    "sentiment_14d": "14-day average news sentiment",

    "news_count_3d": "News articles published over the last 3 days",
    "news_count_7d": "News articles published over the last 7 days",
    "news_count_14d": "News articles published over the last 14 days",

    "sentiment_change": "Change in average news sentiment",
    "sentiment_pct_change": "Percentage change in average news sentiment",

    "sentiment_std_7d": "7-day rolling sentiment volatility",
    "sentiment_std_14d": "14-day rolling sentiment volatility",

    "positive_streak": "Indicator that positive news has recently dominated",
    "negative_streak": "Indicator that negative news has recently dominated",

    "news_surprise": "Difference between today's news volume and its recent average",

    "confidence_std_7d": "7-day variation in sentiment confidence",

    "weighted_sentiment": "News sentiment weighted by article count",

    "positive_intensity": "Strength of positive news considering confidence",
    "negative_intensity": "Strength of negative news considering confidence",

    "ewm_sentiment": "Exponentially weighted moving average of news sentiment",
}

BASE_FEATURES.update({
    "consumer_sentiment": "Consumer Sentiment Index",
    "fed_funds_rate": "Federal Funds Rate",
    "gdp": "Gross Domestic Product",
    "industrial_production": "Industrial Production Index",
    "inflation_cpi": "Consumer Price Index (CPI)",
    "pce": "Personal Consumption Expenditures Price Index",
    "retail_sales": "Retail Sales",
    "treasury_10y": "10-Year Treasury Yield",
    "treasury_3m": "3-Month Treasury Yield",
    "unemployment_rate": "Unemployment Rate",
    "yield_curve": "Difference between the 10-Year and 3-Month Treasury Yields",

    "yield_curve_inverted": "Indicator showing whether the yield curve is inverted",
    "high_inflation": "Indicator showing whether CPI exceeds the high inflation threshold",
    "high_unemployment": "Indicator showing whether unemployment exceeds the high unemployment threshold",

    "real_interest_rate": "Difference between the Federal Funds Rate and CPI",
    "consumer_vs_inflation": "Ratio of Consumer Sentiment to CPI",
    "growth_minus_inflation": "Difference between GDP and CPI",
    "retail_vs_unemployment": "Ratio of Retail Sales to the Unemployment Rate",

    "gdp_up": "Indicator showing whether GDP increased from the previous period",
    "inflation_up": "Indicator showing whether CPI increased from the previous period",
    "rates_up": "Indicator showing whether the Federal Funds Rate increased from the previous period",

    "economic_stress": "Composite indicator combining inflation, unemployment and consumer sentiment",
    "yield_curve_change": "Change in the yield curve from the previous period",
    "yield_curve_ma6": "6-period moving average of the yield curve",
})

def describe_feature(feature: str) -> str:

    if feature in BASE_FEATURES:
        return BASE_FEATURES[feature]

    suffix_rules = {
        "_lag3": "3-period lag of {}",
        "_lag6": "6-period lag of {}",
        "_lag12": "12-period lag of {}",

        "_ewm6": "6-period exponentially weighted moving average of {}",
        "_ewm12": "12-period exponentially weighted moving average of {}",

        "_pct_change": "Percentage change in {}",
        "_diff": "First difference of {}",

        "_rolling3": "3-period rolling average of {}",
        "_rolling6": "6-period rolling average of {}",
        "_rolling12": "12-period rolling average of {}",

        "_rolling_std3": "3-period rolling standard deviation of {}",
        "_rolling_std6": "6-period rolling standard deviation of {}",
        "_rolling_std12": "12-period rolling standard deviation of {}",

        "_momentum3": "3-period momentum of {}",
        "_momentum6": "6-period momentum of {}",

        "_zscore": "12-period rolling z-score of {}",
    }

    for suffix, template in suffix_rules.items():
        if feature.endswith(suffix):
            base = feature[:-len(suffix)]
            if base in BASE_FEATURES:
                return template.format(BASE_FEATURES[base])

    return feature.replace("_", " ")


