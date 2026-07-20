import pandas as pd
from core_logic.inference.classifier_predict import classifier_prediction
from core_logic.inference.regression_predict import regression_prediction
from core_logic.inference.build_features_df import build_inference_dataframe
from core_logic.inference.shap_explaination import explain_shap
from core_logic.inference.llm_report import generate_report
def predict(company:str):
    complete_dict=build_inference_dataframe(company)
    classifier_result=classifier_prediction(complete_dict)
    regressor_result=regression_prediction(complete_dict)
    shap_result=explain_shap(complete_dict['latest_row'])
    
    llm_report = generate_report(
        company,
        classifier_result,
        regressor_result,
        shap_result,
        complete_dict["news"]
    )

    return {
        "classifier": classifier_result,
        "regression": regressor_result,
        "shap": shap_result,
        "llm_report": llm_report,
        "history_df": complete_dict["final_df"],
        "ticker": complete_dict["ticker"]
    }