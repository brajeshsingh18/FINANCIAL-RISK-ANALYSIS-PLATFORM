import joblib
import pandas as pd

reg = joblib.load("models/best_final_regression_model.pkl")
# scaler = joblib.load("models/regression_scaler.pkl") not needed
feature_columns = joblib.load("models/regression_feature_columns.pkl")

def regression_prediction(dictionary):
    latest_row=dictionary['latest_row']
    df=dictionary['final_df']
    X = df[feature_columns].copy()
    X=X.reindex(columns=feature_columns)
    x=latest_row[feature_columns].copy()
    x=x.reindex(columns=feature_columns)
    # X = scaler.transform(X)  its best model is random forest for which scaling is not needed
    history_predictions = reg.predict(X)
    current_prediction = float(reg.predict(x)[0])

    return {
        "history_predictions": history_predictions,
        "current_prediction": current_prediction
    }