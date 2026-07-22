import warnings
import logging
from pandas.errors import PerformanceWarning
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

logging.basicConfig(level=logging.ERROR)

logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)

import pandas as pd
from .predict_classifier import classifier_prediction
from .predict_regressor import regression_prediction
from .build_features_df import build_inference_dataframe
from .shap_explaination import explain_shap
from .llm_report import generate_report
from .llm_context import build_llm_context,summarize_shap,detect_model_consistency
from .refining_llm_output import report_to_json
from tickers_map import company_to_ticker
from time import perf_counter
from datetime import datetime, timezone
from core_logic.utils.config import API_VERSION
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
            llm_report=generate_report(context)
            final_llm_report=report_to_json(llm_report)
        except Exception as e:
             raise LLMReportError(
            "LLM report generation failed."
            )
        after_llm=perf_counter()

        if int(classifier_result["current_prediction"][0])==0:
            label="Low Risk"
            confidence=1-round(float(classifier_result["current_probability"][0]),2)
        else:
            label="High Risk"
            confidence=round(float(classifier_result["current_probability"][0]),2)

        
        consistency=detect_model_consistency(classifier_result,regressor_result)

        keys = [k for k, v in company_to_ticker.items() if v == complete_dict["ticker"]]
       
        

        
        return {
            "ticker": complete_dict["ticker"],
            "company":keys,
            "classification": {
                "label":label,
                "prediction": int(classifier_result["current_prediction"][0]),
                "probability": round(float(classifier_result["current_probability"][0]),4),
                "confidence":confidence
            },
            "volatility_prediction": {
                "predicted_volatility": round(float(regressor_result["current_prediction"]),4)
            },
            "model_consistency":{
                'level': consistency["Consistency"],
                'confidence':consistency["Confidence"],
                'volatility_ratio':consistency["Volatility Ratio"]
            },
            "shap": summarize_shap(shap_result),
            "llm_report": final_llm_report,
            "inference_metrics": {
                'feature building':round(after_building-start_time,4),
                'classification':round(after_classifier-after_building,4),
                'regression':round(after_regressor-after_classifier,4),
                'shap values':round(after_shap-after_regressor,4),
                'llm':round(after_llm-after_shap,4),
                'total time taken':round(after_llm-start_time,4)
                },
            "metadata":{
                "generated_at":  datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                 "api_version": API_VERSION
            }
            }
    
    except FinancialRiskPlatformError:
        raise
    except Exception as e:
        raise InferencePipelineError(
        f"Inference pipeline failed for {company}.") from e
    