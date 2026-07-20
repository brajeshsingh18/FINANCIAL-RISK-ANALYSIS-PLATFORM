import joblib
import pandas as pd
from core_logic.inference.build_features_df import build_inference_dataframe

clf=joblib.load("models/best_classifer_final_model.pkl")
# scaler = joblib.load("models/scaler.pkl")  not needed
feature_columns = joblib.load("models/classification_feature_columns.pkl")

def classifier_prediction(dictionary):
    latest_row=dictionary['latest_row']
    df=dictionary['final_df']
    
    X = df[feature_columns].copy()
    X=X.reindex(columns=feature_columns)
    x=latest_row[feature_columns].copy()
    x=x.reindex(columns=feature_columns)
    # X = scaler.transform(X) no scaler needed in  lgbm which is our best classifier model 
    predict_all = clf.predict(X)
    probability_all = clf.predict_proba(X)[:,1]
    prediction=clf.predict(x)
    prob=clf.predict_proba(x)[:,1]
   
    return {
        "current_prediction": prediction,
        "current_probability": prob,
        "history_predictions": predict_all,
        "history_probability": probability_all
    }