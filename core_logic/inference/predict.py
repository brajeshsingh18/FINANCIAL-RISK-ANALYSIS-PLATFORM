import pandas as pd
from .predict_classifier import classifier_prediction
from .predict_regressor import regression_prediction
from .build_features_df import build_inference_dataframe
from .shap_explaination import explain_shap
from .llm_report import generate_report
from .llm_context import build_llm_context
from time import perf_counter
from ..exceptions import LLMContextError,LLMReportError,InferencePipelineError,FinancialRiskPlatformError

def analyse_stock(company:str):
    try:
        start_time=perf_counter()
        complete_dict=build_inference_dataframe(company)
        after_building=perf_counter()
        classifier_result=classifier_prediction(complete_dict)
        after_classifier=perf_counter()
        regressor_result=regression_prediction(complete_dict)
        after_regressor=perf_counter()
        shap_result=explain_shap(complete_dict['latest_row'])
        after_shap=perf_counter()
        
        try:    
            context=build_llm_context(company=company,ticker=complete_dict["ticker"],classifier_result=classifier_result,regression_result=regressor_result,shap_df=shap_result,latest_features=complete_dict["latest_row"],history_df=complete_dict["final_df"],news_df=complete_dict["news"])
        except Exception as e:
            raise LLMContextError(
            "Unable to construct LLM context."
            )
        try:
            final_llm_report=generate_report(context)
        except Exception as e:
             raise LLMReportError(
            "LLM report generation failed."
            )
        after_llm=perf_counter()
        return {
            "classifier": classifier_result,
            "regression": regressor_result,
            "shap": shap_result,
            "llm_report": final_llm_report,
            "history_df": complete_dict["final_df"],
            "ticker": complete_dict["ticker"],
            "inference_time":{
                'feature building':round(after_building-start_time,4),
                'classification':round(after_classifier-after_building,4),
                'regression':round(after_regressor-after_classifier,4),
                'shap values':round(after_shap-after_regressor,4),
                'llm':round(after_llm-after_shap,4),
                'total time taken':round(after_llm-start_time,4)
                }
            }
    except FinancialRiskPlatformError:
        raise
    except Exception as e:
        raise InferencePipelineError(
        f"Inference pipeline failed for {company}.") from e
    