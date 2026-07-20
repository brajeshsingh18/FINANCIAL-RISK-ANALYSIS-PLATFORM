
from prompts import llm_explainer_prompt
from get_llm import llm
def generate_report(company,classifier_result,regression_result,shap_results,news):

        top_shap = (
        shap_results.head(10)
        .to_string(index=False)
        )
        news_text = (news[["title", "sentiment_label"]].tail(10).to_string(index=False))

        prompt = llm_explainer_prompt.format(
        company=company,
        prediction=(
            "High Risk"
            if classifier_result["prediction"] == 1
            else "Low Risk"
        ),
        probability=classifier_result["prob"],
        volatility=regression_result["current_prediction"],
        shap=top_shap,
        news=news_text,
        )

        report = llm.invoke(prompt).content

        return report