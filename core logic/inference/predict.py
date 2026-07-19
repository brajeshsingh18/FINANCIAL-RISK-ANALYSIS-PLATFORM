import pandas as pd
from classifier_predict import classifier_prediction
from regression_predict import regression_prediction

def predict(data: pd.DataFrame):
    cls = classifier_prediction(data)
    reg = regression_prediction(data)
    probability = float(cls["High_Risk_Probability"].iloc[0])
    prediction = int(cls["Predicted_Class"].iloc[0])
    risk_score = float(reg["Predicted_Risk_Score"].iloc[0])
    return {
        "risk_class": "High Risk" if prediction == 1 else "Low Risk",
        "risk_probability": round(probability * 100, 2),
        "risk_score": round(risk_score, 2)
    }