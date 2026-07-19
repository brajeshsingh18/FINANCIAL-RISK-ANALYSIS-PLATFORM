import joblib
import pandas as pd

reg = joblib.load("models/best_final_regression_model.pkl")
scaler = joblib.load("models/regression_scaler.pkl")
feature_columns = joblib.load("models/regression_feature_columns.pkl")


def regression_prediction(data: pd.DataFrame):
    X = data[feature_columns].copy()
    X = scaler.transform(X)
    prediction = reg.predict(X)
    result = pd.DataFrame({
        "Predicted_Risk_Score": prediction
    })
    return result