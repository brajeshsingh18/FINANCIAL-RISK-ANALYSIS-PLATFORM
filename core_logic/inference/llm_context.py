import numpy as np
import pandas as pd
import joblib
from core_logic.utils.feature_descriptions import describe_feature

#macro_features_to_be_used=pd.read_parquet("data/feature_engineered/macro/macro_features.parquet")
# feature_list = macro_features_to_be_used.iloc[:, 0].tolist()

feature_list=joblib.load("models/macro_feature_columns.pkl")
def safe_float(x):
    try:
        if isinstance(x, pd.DataFrame):
            x = x.iloc[0, 0]
        elif isinstance(x, pd.Series):
            x = x.iloc[0]

        if pd.isna(x):
            return None

        return float(x)
    except:
        return None


def get_value(df, col):
    if isinstance(df, pd.Series):
        return safe_float(df.get(col))

    if col not in df.columns or df.empty:
        return None

    return safe_float(df[col].iloc[0])


def percentage_change(current, previous):
    if previous is None or previous == 0:
        return None
    return ((current - previous) / abs(previous)) * 100

def prediction_confidence(classifier_result):
    prediction = int(classifier_result["current_prediction"][0])
    probability = float(classifier_result["current_probability"][0])

    return (1 - probability) if prediction == 0 else probability


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
        "Value": round(safe_float(rsi),2),
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
        "MACD": round(safe_float(macd),2),
        "Signal": round(safe_float(signal),2),
        "Difference": round(safe_float(diff),2),
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
        "Value": round(safe_float(adx),2),
        "Trend Strength": strength
    }


def interpret_atr(current, history):
    if current is None:
        return {}

    history = [x for x in history if pd.notna(x)]

    if not history:
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
        "Current": round(safe_float(current),2),
        "30-Day Average": round(safe_float(avg),2),
        "Ratio": round(safe_float(ratio),2),
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
        "Width": round(safe_float(width),2),
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
        "Ratio": round(safe_float(volume_ratio),2),
        "State": state
    }


def interpret_momentum(momentum):
    if momentum is None:
        return {}

    if momentum > 0:
        direction = "Positive"
    elif momentum < 0:
        direction = "Negative"
    else:
        direction = "Neutral"

    return {
        "Value": round(safe_float(momentum),2),
        "Direction": direction
    }


def interpret_ema(close, ema12, ema26):
    if None in (close, ema12, ema26):
        return {}

    bullish = int(close > ema12) + int(close > ema26)

    if bullish == 2:
        trend = "Bullish"
    elif bullish == 1:
        trend = "Mixed"
    else:
        trend = "Bearish"

    return {
        "Bullish Signals": bullish,
        "Trend": trend
    }


def build_market_regime(latest, history_df):
    return {
        "RSI": interpret_rsi(get_value(latest, "rsi_14")),
        "MACD": interpret_macd(
            get_value(latest, "macd"),
            get_value(latest, "macd_signal")
        ),
        "ADX": interpret_adx(get_value(latest, "adx_14")),
        "ATR": interpret_atr(
            get_value(latest, "atr_14"),
            history_df["atr_14"].tail(30).tolist()
        ),
        "Bollinger": interpret_bollinger(get_value(latest, "bb_width")),
        "Volume": interpret_volume(get_value(latest, "relative_volume")),
        "Momentum": interpret_momentum(get_value(latest, "roc_10")),
        "EMA": interpret_ema(
            get_value(latest, "close"),
            get_value(latest, "ema_12"),
            get_value(latest, "ema_26")
        )
    }


from collections import Counter

def summarize_risk_history(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    current = probabilities[-1]
    avg7 = probabilities[-7:].mean()
    avg30 = probabilities[-30:].mean()

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
        "Current Probability": round(current * 100, 2),
        "7 Day Average": round(avg7 * 100, 2),
        "30 Day Average": round(avg30 * 100, 2),
        "Maximum": round(probabilities.max() * 100, 2),
        "Minimum": round(probabilities.min() * 100, 2),
        "Standard Deviation": round(probabilities.std() * 100, 2),
        "7 Day Change (%)": round(percentage_change(current, avg7), 2) if percentage_change(current, avg7) is not None else None,
        "30 Day Change (%)": round(percentage_change(current, avg30), 2) if percentage_change(current, avg30) is not None else None,
        "Trend": trend
    }


def summarize_volatility_history(predictions):
    predictions = np.asarray(predictions, dtype=float)

    current = predictions[-1]
    avg7 = predictions[-7:].mean()
    avg30 = predictions[-30:].mean()

    slope = np.polyfit(np.arange(len(predictions)), predictions, 1)[0]

    if slope > 0.005:
        trend = "Increasing"
    elif slope < -0.005:
        trend = "Decreasing"
    else:
        trend = "Stable"

    return {
        "Current Prediction": round(current * 100, 2),
        "7 Day Average": round(float(avg7)*100,2),
        "30 Day Average": round(float(avg30)*100,2),
        "Maximum": round(float(predictions.max())*100,2),
        "Minimum": round(float(predictions.min())*100,2),
        "Standard Deviation": round(float(predictions.std())*100,2),
        "7 Day Change (%)": round(percentage_change(current, avg7), 2) if percentage_change(current, avg7) is not None else None,
        "30 Day Change (%)": round(percentage_change(current, avg30), 2) if percentage_change(current, avg30) is not None else None,
        "Trend": trend
    }


def summarize_shap(shap_df):
    positive = shap_df[shap_df["shap_value"] > 0]
    negative = shap_df[shap_df["shap_value"] < 0]

    top_positive = (
        positive.sort_values("shap_value", ascending=False)
        .head(10)
        .assign(
            shap_value=lambda x: x["shap_value"].round(3),
            description=lambda x: x["feature"].apply(describe_feature)
        )
        .to_dict("records")
    )

    top_negative = (
        negative.sort_values("shap_value")
        .head(10)
        .assign(
            shap_value=lambda x: x["shap_value"].round(3),
            description=lambda x: x["feature"].apply(describe_feature)
        )
        .to_dict("records")
    )

    return {
        "SHAP Target": "High Risk (Class 1)",
        "Top Drivers Increasing Risk": top_positive,
        "Top Drivers Reducing Risk": top_negative,
        "Total Risk Increasing Contribution": round(float(positive["shap_value"].sum()), 3),
        "Total Risk Reducing Contribution": round(float(negative["shap_value"].sum()), 3),
        "Strongest Risk Increasing Driver":
            top_positive[0]["description"] if top_positive else None,
        "Strongest Risk Reducing Driver":
            top_negative[0]["description"] if top_negative else None,
    }


def summarize_news(news_df):
    summary = {}

    if "sentiment_label" in news_df.columns:
        counts = news_df["sentiment_label"].value_counts().to_dict()

        summary.update({
            "Positive": counts.get("positive", 0),
            "Neutral": counts.get("neutral", 0),
            "Negative": counts.get("negative", 0),
            "Dominant Sentiment": max(counts, key=counts.get) if counts else None
        })

    if "sentiment_score" in news_df.columns:
        summary.update({
            "Average Sentiment Score": round(float(news_df["sentiment_score"].mean()),3),
            "Maximum Sentiment Score": round(float(news_df["sentiment_score"].max()),3),
            "Minimum Sentiment Score": round(float(news_df["sentiment_score"].min()),3)
        })

    if "title" in news_df.columns:
        summary["Recent Headlines"] = news_df["title"].tail(10).tolist()

    return summary


def summarize_macro(latest):
    missing = [col for col in feature_list if col not in latest.columns]

    with open("macro_debug.txt", "w") as f:
        f.write(f"Feature list count: {len(feature_list)}\n")
        f.write(f"Latest column count: {len(latest.columns)}\n\n")

        f.write("Missing columns:\n")
        for col in missing:
            f.write(f"{col}\n")

        f.write("\nMacro values:\n")
        for col in feature_list:
            f.write(f"{col}: {get_value(latest, col)}\n")

    summary = {}
    for col in feature_list:
        value = get_value(latest, col)
        if value is None:
            continue
        if value not in (0, 1):
            value = round(value, 2)
        if col == "inflation_cpi":
            summary["consumer_price_index"] = value
        else:
            summary[col] = value
    return summary


def detect_model_consistency(classifier_result, regression_result):
    probability = float(classifier_result["current_probability"][0])
    prediction = int(classifier_result["current_prediction"][0])
    confidence = prediction_confidence(classifier_result)
    volatility = float(regression_result["current_prediction"])
    avg_volatility = np.mean(regression_result["history_predictions"])
    vol_ratio = volatility / avg_volatility
    if prediction == 1:     
        if confidence >= 0.90 and vol_ratio >= 1.10:
            level = "Very High"
        elif vol_ratio < 0.70:
            level = "Mixed"
        else:
            level = "High"

    else:                   
        if confidence >= 0.90 and vol_ratio <= 1.20:
            level = "Very High"
        elif vol_ratio > 2.0:
            level = "Mixed"
        else:
            level = "High"

    return {
        "Consistency": level,
        "Classification": prediction,
        "Probability": round(probability * 100, 2),
        "Confidence": round(confidence * 100, 2),
        "Predicted Volatility": round(volatility * 100, 2),
        "Volatility Ratio": round(vol_ratio, 2)
    }

def executive_snapshot(company, ticker, classifier_result, regression_result):
    return {
        "Company": company,
        "Ticker": ticker,
        "Risk": "High Risk" if int(classifier_result["current_prediction"][0]) else "Low Risk",
        "Probability": round(float(classifier_result["current_probability"][0])*100,2),
        "Confidence": confidence_level(prediction_confidence(classifier_result)),
        "Predicted Volatility": round(float(regression_result["current_prediction"])*100,2)
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
        "Risk Trend": risk_summary["Trend"],
        "Volatility Trend": volatility_summary["Trend"],
        "Dominant News Sentiment": news_summary.get("Dominant Sentiment"),
        "Strongest Risk Increasing Driver": shap_summary["Strongest Risk Increasing Driver"],
        "Strongest Risk Reducing Driver": shap_summary["Strongest Risk Reducing Driver"],
        "Model Consistency": consistency["Consistency"],
        "Technical Regime": technical_summary.get("EMA", {}).get("Trend"),
        "Momentum": technical_summary.get("Momentum", {}).get("Direction"),
        "Trend Strength": technical_summary.get("ADX", {}).get("Trend Strength"),
        "Volatility Regime": technical_summary.get("ATR", {}).get("Regime")
    }


def build_llm_context(
    company,
    ticker,
    classifier_result,
    regression_result,
    shap_df,
    latest_features,
    history_df,
    news_df
):
    technical_summary = build_market_regime(
        latest_features,
        history_df
    )

    risk_summary = summarize_risk_history(
        classifier_result["history_probability"]
    )

    volatility_summary = summarize_volatility_history(
        regression_result["history_predictions"]
    )

    shap_summary = summarize_shap(shap_df)

    news_summary = summarize_news(news_df)

    macro_summary = summarize_macro(latest_features)

    consistency = detect_model_consistency(
        classifier_result,
        regression_result
    )

    executive = executive_snapshot(
        company,
        ticker,
        classifier_result,
        regression_result
    )

    intelligence = build_intelligence_summary(
        technical_summary,
        risk_summary,
        volatility_summary,
        news_summary,
        shap_summary,
        consistency
    )
    prediction = int(classifier_result["current_prediction"][0])
    probability = float(classifier_result["current_probability"][0])
    if prediction == 1:
        signal_strength = classify_probability(probability)
    else:
        signal_strength = "Very Strong Low-Risk Signal"


    return {
        "executive_snapshot": executive,
        "company": company,
        "ticker": ticker,
        "risk_prediction": {
            "classification": (
                "High Risk"
                if prediction==1
                else "Low Risk"
            ),
            "probability_percent": round(
                float(classifier_result["current_probability"]) * 100,2),
            "confidence": confidence_level(
                prediction_confidence(classifier_result)
            ),
            "signal_strength":signal_strength
        },

        "volatility_prediction": {
            "predicted_volatility": round(
                float(regression_result["current_prediction"]) * 100,2)
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