import joblib
import pandas as pd
from core_logic.exceptions import ModelLoadingError,PredictionError

try:
    reg = joblib.load("models/best_final_regression_model.pkl")
except Exception as e:
    raise ModelLoadingError(
        "Classifier model could not be loaded."
    ) from e
# scaler = joblib.load("models/regression_scaler.pkl") not needed
try:
    feature_columns = joblib.load("models/regression_feature_columns.pkl")
except Exception as e:
    raise ModelLoadingError(
        "Classifier model could not be loaded."
    ) from e

def regression_prediction(dictionary):
    if "latest_row" not in dictionary:
        raise ValueError("latest_row not found in inference dictionary.")

    if dictionary["latest_row"].empty:
        raise ValueError("No inference data available.")
    
    latest_row=dictionary['latest_row']
    df=dictionary['final_df']
    X = df[feature_columns].copy()
    X=X.reindex(columns=feature_columns)
    x=latest_row[feature_columns].copy()
    x=x.reindex(columns=feature_columns)
    # X = scaler.transform(X)  its best model is random forest for which scaling is not needed
    try:
        history_predictions = reg.predict(X)
        current_prediction = float(reg.predict(x)[0])
    except Exception as e:
        raise PredictionError(
        "Regressor prediction failed."
        )

    return {
        "history_predictions": history_predictions,
        "current_prediction": current_prediction
    }