
from prompts import llm_explainer_prompt
from get_llm import llm
from core_logic.inference.llm_context import build_llm_context
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage

def format_llm_context(context):

    report = []
    report.append("=" * 80)
    report.append("EXECUTIVE SNAPSHOT")
    report.append("=" * 80)
    executive = context["executive_snapshot"]
    for key, value in executive.items():
        report.append(f"{key}: {value}")

    report.append("\n")
    report.append("=" * 80)
    report.append("RISK PREDICTION")
    report.append("=" * 80)
    for key, value in context["risk_prediction"].items():
        report.append(f"{key}: {value}")

    report.append("\n")
    report.append("=" * 80)
    report.append("VOLATILITY PREDICTION")
    report.append("=" * 80)

    for key, value in context["volatility_prediction"].items():
        report.append(f"{key}: {value}")

    report.append("\n")
    report.append("=" * 80)
    report.append("HISTORICAL RISK ANALYSIS")
    report.append("=" * 80)
    for key, value in context["historical_risk_analysis"].items():
        report.append(f"{key}: {value}")

    report.append("\n")
    report.append("=" * 80)
    report.append("HISTORICAL VOLATILITY ANALYSIS")
    report.append("=" * 80)
    for key, value in context["historical_volatility_analysis"].items():
        report.append(f"{key}: {value}")

    report.append("\n")
    report.append("=" * 80)
    report.append("TECHNICAL ANALYSIS")
    report.append("=" * 80)
    technical = context["technical_analysis"]
    for indicator, values in technical.items():
        report.append(f"\n{indicator}")
        if isinstance(values, dict):
            for k, v in values.items():
                report.append(f"   {k}: {v}")
        else:
            report.append(str(values))

    report.append("\n")
    report.append("=" * 80)
    report.append("MACROECONOMIC ANALYSIS")
    report.append("=" * 80)
    for key, value in context["macroeconomic_analysis"].items():
        report.append(f"{key}: {value}")

    report.append("\n")
    report.append("=" * 80)
    report.append("NEWS ANALYSIS")
    report.append("=" * 80)
    news = context["news_analysis"]
    for key, value in news.items():
        if isinstance(value, list):
            report.append(f"\n{key}")
            for item in value:
                report.append(f" - {item}")
        else:
            report.append(f"{key}: {value}")

    report.append("\n")
    report.append("=" * 80)
    report.append("SHAP ANALYSIS")
    report.append("=" * 80)
    shap = context["shap_analysis"]
    for key, value in shap.items():
        if isinstance(value, list):
            report.append(f"\n{key}")
            for item in value:
                report.append(str(item))
        else:
            report.append(f"{key}: {value}")

    report.append("\n")
    report.append("=" * 80)
    report.append("MODEL CONSISTENCY")
    report.append("=" * 80)
    for key, value in context["model_consistency"].items():
        report.append(f"{key}: {value}")

    report.append("\n")
    report.append("=" * 80)
    report.append("OVERALL INTELLIGENCE")
    report.append("=" * 80)
    for key, value in context["overall_intelligence"].items():
        report.append(f"{key}: {value}")

    return "\n".join(report)


def generate_report(context):
    formatted_context = format_llm_context(context)

    messages = [
        SystemMessage(content=llm_explainer_prompt),
        HumanMessage(content=f"""
                    Analyze the following financial risk context and generate the report.
                    {format_llm_context(context)} """
                ),
            ]
    response = llm.invoke(messages)
    return response.content