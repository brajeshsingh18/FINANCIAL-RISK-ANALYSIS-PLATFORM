import joblib
import pandas as pd

clf=joblib.load("models/best_classifer_final_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/classification_feature_columns.pkl")

def classifier_prediction(data:pd.DataFrame):
    X = data[feature_columns].copy()
    X = scaler.transform(X)
    prediction = clf.predict(X)
    probability = clf.predict_proba(X)[:, 1]
    result = pd.DataFrame({
        "Predicted_Class": prediction,
        "High_Risk_Probability": probability
    })
    return result