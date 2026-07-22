import joblib
import shap
import pandas as pd
from core_logic.exceptions import SHAPExplanationError
classifier = joblib.load("models/best_classifier_final_model.pkl")
feature_columns=joblib.load("models/classificaion_feature_columns.pkl")
explainer = shap.TreeExplainer(classifier)

def explain_shap(latest_row:pd.DataFrame):
    if latest_row.empty:
        raise ValueError("Cannot generate SHAP explanation for empty dataframe.")
    
    latest_row = latest_row[feature_columns]
    try:
        shap_values = explainer(latest_row)
        df = pd.DataFrame({
            "feature": latest_row.columns,
            "shap_value": shap_values.values[0]
        })
    except Exception as e:
        raise SHAPExplanationError(
        "Failed to generate SHAP explanation."
        )
    
    df["abs"] = round(df["shap_value"].abs(),3)
    df = df.sort_values("abs",ascending=False)
    return df.head(10)