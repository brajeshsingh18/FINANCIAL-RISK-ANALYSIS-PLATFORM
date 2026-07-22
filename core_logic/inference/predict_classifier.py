import joblib
import pandas as pd
from core_logic.inference.build_features_df import build_inference_dataframe
from core_logic.exceptions import ModelLoadingError,PredictionError

try:
    clf=joblib.load("models/best_classifier_final_model.pkl")
except Exception as e:
    raise ModelLoadingError(
        "Classifier model could not be loaded."
    ) from e
# scaler = joblib.load("models/scaler.pkl")  not needed
try:
    feature_columns = joblib.load("models/classificaion_feature_columns.pkl")
except Exception as e:
    raise ModelLoadingError(
        "Classifier feature columns could not be loaded."
    ) from e

def classifier_prediction(dictionary):
    if "latest_row" not in dictionary:
        raise ValueError("latest_row not found in inference dictionary.")
    if dictionary["latest_row"].empty:
        raise ValueError("No inference data available.")
    
    latest_row=dictionary['latest_row']
    df=dictionary['final_df']
    
    X = df[feature_columns].copy()
    x=latest_row[feature_columns].copy()
    # X = scaler.transform(X) no scaler needed in  lgbm which is our best classifier model 
    try:
        predict_all = clf.predict(X)
        probability_all = clf.predict_proba(X)[:,1]
        prediction=clf.predict(x)
        prob=clf.predict_proba(x)[:,1]
    except Exception as e:
        raise PredictionError(
        "Classifier prediction failed."
        )
   
    return {
        "current_prediction": prediction,
        "current_probability": prob,
        "history_predictions": predict_all,
        "history_probability": probability_all
    }