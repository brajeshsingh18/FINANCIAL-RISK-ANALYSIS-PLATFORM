import joblib
import shap
import pandas as pd
classifier = joblib.load("models/best_classifier_final_model.pkl")
explainer = shap.TreeExplainer(classifier)

def explain_shap(latest_row:pd.DataFrame):
    shap_values = explainer(latest_row)
    df = pd.DataFrame({
        "feature": latest_row.columns,
        "shap_value": shap_values.values[0]
    })
    df["abs"] = df["shap_value"].abs()
    df = df.sort_values("abs",ascending=False)
    return df.head(10)