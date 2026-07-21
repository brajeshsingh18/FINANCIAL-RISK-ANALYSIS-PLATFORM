import numpy as np
import pandas as pd

def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except:
        return None


def percentage_change(current, previous):
    if previous is None:
        return None
    if previous == 0:
        return None
    return ((current - previous) / abs(previous)) * 100


def confidence_level(probability):
    if probability >= 0.95:
        return "Extremely High"
    if probability >= 0.90:
        return "Very High"
    if probability >= 0.80:
        return "High"
    if probability >= 0.65:
        return "Moderate"
    return "Low"


def classify_probability(probability):
    if probability >= 0.90:
        return "Very Strong Risk Signal"
    if probability >= 0.75:
        return "Strong Risk Signal"
    if probability >= 0.60:
        return "Moderate Risk Signal"
    if probability >= 0.50:
        return "Weak Risk Signal"
    return "Very Weak Risk Signal"


def interpret_rsi(rsi):
    if rsi is None:
        return {}

    if rsi >= 80:
        state = "Extremely Overbought"
        implication = "Very high bullish momentum but elevated probability of correction."
    elif rsi >= 70:
        state = "Overbought"
        implication = "Strong buying pressure with possible pullback."
    elif rsi <= 20:
        state = "Extremely Oversold"
        implication = "Very strong selling pressure with rebound potential."
    elif rsi <= 30:
        state = "Oversold"
        implication = "Weak price momentum with recovery potential."
    else:
        state = "Neutral"
        implication = "Momentum remains balanced."
    return {
        "Value": safe_float(rsi),
        "State": state,
        "Interpretation": implication
    }


def interpret_macd(macd, signal):
    if macd is None or signal is None:
        return {}
    
    diff = macd - signal
    if diff > 0:
        trend = "Bullish Crossover"
        implication = "Positive momentum dominates."
    elif diff < 0:
        trend = "Bearish Crossover"
        implication = "Negative momentum dominates."
    else:
        trend = "Neutral"
        implication = "Momentum is balanced."
    return {
        "MACD": safe_float(macd),
        "Signal": safe_float(signal),
        "Difference": safe_float(diff),
        "State": trend,
        "Interpretation": implication
    }


def interpret_adx(adx):
    if adx is None:
        return {}

    if adx >= 50:
        strength = "Extremely Strong Trend"
    elif adx >= 40:
        strength = "Strong Trend"
    elif adx >= 25:
        strength = "Moderate Trend"
    elif adx >= 20:
        strength = "Weak Trend"
    else:
        strength = "No Significant Trend"

    return {
        "Value": safe_float(adx),
        "Trend Strength": strength
    }


def interpret_atr(current, history):
    if current is None:
        return {}
    history = [x for x in history if pd.notna(x)]
    if len(history) == 0:
        return {}

    avg = np.mean(history)
    if avg == 0:
        return {}

    ratio = current / avg
    if ratio >= 1.5:
        regime = "Extremely High Volatility"
    elif ratio >= 1.25:
        regime = "High Volatility"
    elif ratio <= 0.75:
        regime = "Low Volatility"
    else:
        regime = "Normal Volatility"

    return {
        "Current": safe_float(current),
        "30-Day Average": safe_float(avg),
        "Ratio": safe_float(ratio),
        "Regime": regime
    }


def interpret_bollinger(width):
    if width is None:
        return {}

    if width >= 0.30:
        state = "Wide Bands"

    elif width <= 0.08:
        state = "Volatility Squeeze"

    else:
        state = "Normal"

    return {
        "Width": safe_float(width),
        "State": state
    }


def interpret_volume(volume_ratio):
    if volume_ratio is None:
        return {}

    if volume_ratio > 2:
        state = "Extremely High"

    elif volume_ratio > 1.3:
        state = "Above Average"

    elif volume_ratio < 0.7:
        state = "Below Average"

    else:
        state = "Average"

    return {
        "Ratio": safe_float(volume_ratio),
        "State": state
    }


def interpret_momentum(momentum):
    if momentum is None:
        return {}

    if momentum > 0:
        trend = "Positive"

    elif momentum < 0:
        trend = "Negative"

    else:
        trend = "Neutral"

    return {
        "Value": safe_float(momentum),
        "Direction": trend
    }


def interpret_ema(close, ema20, ema50, ema200):
    bullish = 0

    if close > ema20:
        bullish += 1

    if close > ema50:
        bullish += 1

    if close > ema200:
        bullish += 1

    if bullish == 3:
        regime = "Strong Bullish"

    elif bullish == 2:
        regime = "Bullish"

    elif bullish == 1:
        regime = "Weak"

    else:
        regime = "Bearish"

    return {
        "Bullish Signals": bullish,
        "Trend": regime
    }


def build_market_regime(latest,history_df):
    regime = {}

    regime["RSI"] = interpret_rsi(latest.get("RSI"))
    regime["MACD"] = interpret_macd(
        latest.get("MACD"),
        latest.get("MACD_signal")
    )
    regime["ADX"] = interpret_adx(latest.get("ADX"))
    regime["ATR"] = interpret_atr(
        latest.get("ATR"),
        history_df["ATR"].tail(30).tolist()
    )
    regime["Bollinger"] = interpret_bollinger(
        latest.get("BB_Width")
    )
    regime["Volume"] = interpret_volume(
        latest.get("Volume_Ratio")
    )
    regime["Momentum"] = interpret_momentum(
        latest.get("Momentum")
    )
    regime["EMA"] = interpret_ema(
        latest.get("Close"),
        latest.get("EMA_20"),
        latest.get("EMA_50"),
        latest.get("EMA_200")
    )

    return regime


from collections import Counter

def summarize_risk_history(probabilities):
    probabilities = np.array(probabilities)

    current = float(probabilities[-1])
    avg7 = float(np.mean(probabilities[-7:]))
    avg30 = float(np.mean(probabilities[-30:]))
    maximum = float(np.max(probabilities))
    minimum = float(np.min(probabilities))
    std = float(np.std(probabilities))

    change7 = percentage_change(current, avg7)
    change30 = percentage_change(current, avg30)

    slope = np.polyfit(np.arange(len(probabilities)), probabilities, 1)[0]

    if slope > 0.005:
        trend = "Rapidly Increasing"
    elif slope > 0.001:
        trend = "Increasing"
    elif slope < -0.005:
        trend = "Rapidly Decreasing"
    elif slope < -0.001:
        trend = "Decreasing"
    else:
        trend = "Stable"

    return {
        "Current Probability": current,
        "7 Day Average": avg7,
        "30 Day Average": avg30,
        "Maximum": maximum,
        "Minimum": minimum,
        "Standard Deviation": std,
        "7 Day Change (%)": change7,
        "30 Day Change (%)": change30,
        "Trend": trend
    }


def summarize_volatility_history(predictions):
    predictions = np.array(predictions)

    current = float(predictions[-1])
    avg7 = float(np.mean(predictions[-7:]))
    avg30 = float(np.mean(predictions[-30:]))
    maximum = float(np.max(predictions))
    minimum = float(np.min(predictions))
    std = float(np.std(predictions))

    change7 = percentage_change(current, avg7)
    change30 = percentage_change(current, avg30)

    slope = np.polyfit(np.arange(len(predictions)), predictions, 1)[0]

    if slope > 0.005:
        trend = "Increasing"

    elif slope < -0.005:
        trend = "Decreasing"

    else:
        trend = "Stable"

    return {
        "Current Prediction": current,
        "7 Day Average": avg7,
        "30 Day Average": avg30,
        "Maximum": maximum,
        "Minimum": minimum,
        "Standard Deviation": std,
        "7 Day Change (%)": change7,
        "30 Day Change (%)": change30,
        "Trend": trend
    }


def summarize_shap(shap_df):
    positive = shap_df[shap_df["shap_value"] > 0]
    negative = shap_df[shap_df["shap_value"] < 0]

    top_positive = (
        positive
        .sort_values("shap_value", ascending=False)
        .head(10)
        .to_dict("records")
    )

    top_negative = (
        negative
        .sort_values("shap_value")
        .head(10)
        .to_dict("records")
    )

    return {
        "Top Positive Drivers": top_positive,
        "Top Negative Drivers": top_negative,
        "Total Positive Contribution": float(positive["shap_value"].sum()),
        "Total Negative Contribution": float(negative["shap_value"].sum()),
        "Strongest Positive Driver":
            top_positive[0]["feature"] if len(top_positive) else None,
        "Strongest Negative Driver":
            top_negative[0]["feature"] if len(top_negative) else None
    }


def summarize_news(news_df):

    summary = {}

    if "sentiment_label" in news_df.columns:

        counts = news_df["sentiment_label"].value_counts().to_dict()

        summary["Positive"] = counts.get("positive", 0)
        summary["Neutral"] = counts.get("neutral", 0)
        summary["Negative"] = counts.get("negative", 0)

        summary["Dominant Sentiment"] = max(
            counts,
            key=counts.get
        )

    if "sentiment_score" in news_df.columns:

        summary["Average Sentiment Score"] = float(
            news_df["sentiment_score"].mean()
        )

        summary["Maximum Sentiment Score"] = float(
            news_df["sentiment_score"].max()
        )

        summary["Minimum Sentiment Score"] = float(
            news_df["sentiment_score"].min()
        )

    if "title" in news_df.columns:

        summary["Recent Headlines"] = (
            news_df["title"]
            .tail(10)
            .tolist()
        )

    return summary


def summarize_macro(latest_features):

    macro = {}

    columns = [
        "Inflation",
        "FedFundsRate",
        "GDP",
        "GDP_Growth",
        "Treasury10Y",
        "Treasury2Y",
        "ConsumerSentiment",
        "Unemployment",
        "IndustrialProduction",
        "CPI",
        "OilPrice",
        "GoldPrice",
        "DollarIndex",
        "VIX"
    ]

    for col in columns:

        if col in latest_features.index:

            macro[col] = safe_float(
                latest_features[col]
            )

    return macro


def detect_model_consistency(classifier_result, regression_result):

    probability = classifier_result["current_probability"]

    risk = classifier_result["current_prediction"]

    volatility = regression_result["current_prediction"]

    if risk == 1 and probability > 0.80 and volatility > np.mean(regression_result["history_predictions"]):
        level = "Very High"

    elif risk == 1 and volatility < np.mean(regression_result["history_predictions"]):
        level = "Moderate"

    elif risk == 0 and volatility > np.mean(regression_result["history_predictions"]):
        level = "Mixed"

    else:
        level = "High"

    return {
        "Consistency": level,
        "Classification": risk,
        "Probability": probability,
        "Predicted Volatility": float(volatility)
    }


def executive_snapshot(company, ticker, classifier_result, regression_result):

    return {
        "Company": company,
        "Ticker": ticker,
        "Risk": "High Risk" if classifier_result["current_prediction"] == 1 else "Low Risk",
        "Probability": float(classifier_result["current_probability"]),
        "Confidence": confidence_level(classifier_result["current_probability"]),
        "Predicted Volatility": float(regression_result["current_prediction"])
    }


def build_intelligence_summary(
    technical_summary,
    risk_summary,
    volatility_summary,
    news_summary,
    shap_summary,
    consistency
):

    return {

        "Risk Trend":
            risk_summary["Trend"],

        "Volatility Trend":
            volatility_summary["Trend"],

        "Dominant News Sentiment":
            news_summary.get("Dominant Sentiment"),

        "Strongest Positive Driver":
            shap_summary["Strongest Positive Driver"],

        "Strongest Negative Driver":
            shap_summary["Strongest Negative Driver"],

        "Model Consistency":
            consistency["Consistency"],

        "Technical Regime":
            technical_summary["EMA"]["Trend"]
            if "EMA" in technical_summary
            else None,

        "Momentum":
            technical_summary["Momentum"]["Direction"]
            if "Momentum" in technical_summary
            else None,

        "Trend Strength":
            technical_summary["ADX"]["Trend Strength"]
            if "ADX" in technical_summary
            else None,

        "Volatility Regime":
            technical_summary["ATR"]["Regime"]
            if "ATR" in technical_summary
            else None
    }


def build_llm_context(company,ticker,classifier_result,regression_result,shap_df,latest_features,history_df,news_df):
    technical_summary = build_market_regime(latest_features,history_df)
    risk_summary = summarize_risk_history(classifier_result["history_probabilities"])
    volatility_summary = summarize_volatility_history(regression_result["history_predictions"])
    shap_summary = summarize_shap(shap_df)
    news_summary = summarize_news(news_df)
    macro_summary = summarize_macro(latest_features)
    consistency = detect_model_consistency(classifier_result,regression_result)
    executive = executive_snapshot(company,ticker,classifier_result,regression_result)
    intelligence = build_intelligence_summary(technical_summary,risk_summary,volatility_summary,news_summary,shap_summary,consistency)

    context = {
        "executive_snapshot": executive,
        "company": company,
        "ticker": ticker,
        "risk_prediction": {
            "classification": (
                "High Risk"
                if classifier_result["current_prediction"] == 1
                else "Low Risk"
            ),
            "probability": float(
                classifier_result["current_probability"]
            ),
            "confidence": confidence_level(
                classifier_result["current_probability"]
            ),
            "signal_strength": classify_probability(
                classifier_result["current_probability"]
            )
        },
        "volatility_prediction": {
            "predicted_volatility": float(
                regression_result["current_prediction"]
            )
        },
        "historical_risk_analysis": risk_summary,
        "historical_volatility_analysis": volatility_summary,
        "technical_analysis": technical_summary,
        "macroeconomic_analysis": macro_summary,
        "news_analysis": news_summary,
        "shap_analysis": shap_summary,
        "model_consistency": consistency,
        "overall_intelligence": intelligence
    }

    return context