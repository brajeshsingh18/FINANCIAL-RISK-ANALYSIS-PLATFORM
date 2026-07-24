import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards
from datetime import datetime

API_URL="http://127.0.0.1:8000/predict/"

st.set_page_config(
    page_title="Financial Risk Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}
.block-container{padding-top:1rem;padding-bottom:1rem;}
div[data-testid="metric-container"]{
background:#171B22;
border:1px solid #2D3748;
padding:15px;
border-radius:14px;
}
.st-emotion-cache-13ln4jf{
background:#0E1117;
}
</style>
""",unsafe_allow_html=True)

def fetch_prediction(company):
    try:
        r=requests.post(API_URL,json={"company":company},timeout=300)
        if r.status_code==200:
            return r.json()
        st.error(f"Error {r.status_code}")
        return None
    except Exception as e:
        st.error(e)
        return None

def risk_color(label):
    if label=="Low Risk":
        return "#2ECC71"
    if label=="Medium Risk":
        return "#F39C12"
    return "#E74C3C"

def risk_icon(label):
    if label=="Low Risk":
        return "🟢"
    if label=="Medium Risk":
        return "🟡"
    return "🔴"

with st.sidebar:
    st.title("📈 Financial Risk")
    option_menu(
        "",
        ["Dashboard"],
        icons=["bar-chart-fill"],
        default_index=0
    )
    st.divider()
    company=st.text_input(
        "Company",
        placeholder="Apple, Google, Tesla..."
    )
    analyze=st.button(
        "Analyze",
        use_container_width=True,
        type="primary"
    )
    st.divider()
    st.markdown("### Examples")
    c1,c2=st.columns(2)
    with c1:
        if st.button("Apple",use_container_width=True):
            company="Apple"
        if st.button("Google",use_container_width=True):
            company="Google"
        if st.button("Tesla",use_container_width=True):
            company="Tesla"
    with c2:
        if st.button("Microsoft",use_container_width=True):
            company="Microsoft"
        if st.button("Amazon",use_container_width=True):
            company="Amazon"
        if st.button("NVIDIA",use_container_width=True):
            company="NVIDIA"
    st.divider()
    st.caption("CatBoost • SHAP • FastAPI • LLM")

st.title("Financial Risk Intelligence Platform")
st.caption("AI-Powered Multi-Model Financial Risk Assessment")

if "response" not in st.session_state:
    st.session_state.response=None

if analyze and company.strip():
    with st.spinner("Running Financial Risk Analysis..."):
        st.session_state.response=fetch_prediction(company)

response=st.session_state.response

if response is None:
    st.info("Enter a company name and click Analyze.")
    st.stop()

history=response["historical_data"]
stock_history=pd.DataFrame(history["stock_history"])
price_history=pd.DataFrame(history["price_history"])
probability_history=pd.DataFrame(history["probability_history"])
volatility_history=pd.DataFrame(history["volatility_history"])

classification=response["classification"]
volatility=response["volatility_prediction"]
consistency=response["model_consistency"]
metadata=response["metadata"]

risk=classification["label"]
risk_color_value=risk_color(risk)

st.markdown("---")

c1,c2,c3,c4,c5,c6=st.columns(6)

with c1:
    st.metric(
        "Company",
        response["company"]
    )

with c2:
    st.metric(
        "Ticker",
        response["ticker"]
    )

with c3:
    st.markdown(
        f"""
<div style="background:{risk_color_value};padding:18px;border-radius:14px;text-align:center">
<h4 style="margin:0;color:white;">Risk Level</h4>
<h2 style="margin:8px;color:white;">{risk_icon(risk)} {risk}</h2>
</div>
""",
        unsafe_allow_html=True
    )

with c4:
    st.metric(
        "Probability",
        f'{classification["probability"]*100:.2f}%'
    )

with c5:
    st.metric(
        "Confidence",
        f'{consistency["confidence"]:.2f}%'
    )

with c6:
    st.metric(
        "Predicted Volatility",
        f'{volatility["predicted_volatility"]*100:.2f}%'
    )

style_metric_cards(
    background_color="#171B22",
    border_color="#2D3748",
    border_left_color=risk_color_value
)

st.markdown("")

l,r=st.columns([2,1])

with l:
    st.subheader("Executive Summary")
    st.info(response["llm_report"]["executive_summary"])

with r:
    st.subheader("Model Consistency")

    level=consistency["level"]

    if level=="Very High":
        stars="⭐⭐⭐⭐⭐"
    elif level=="High":
        stars="⭐⭐⭐⭐"
    elif level=="Moderate":
        stars="⭐⭐⭐"
    else:
        stars="⭐⭐"

    st.metric(
        "Consistency",
        level
    )

    st.metric(
        "Volatility Ratio",
        f'{consistency["volatility_ratio"]:.2f}'
    )

    st.metric(
        "Generated",
        metadata["generated_at"][:10]
    )

st.markdown("---")

shap=response["shap"]

left,right=st.columns(2)

probability=classification["probability"]*100
volatility_value=volatility["predicted_volatility"]*100

with left:
    fig=go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        number={"suffix":"%","font":{"size":42}},
        title={"text":"Risk Probability","font":{"size":22}},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":risk_color_value},
            "steps":[
                {"range":[0,30],"color":"#2ECC71"},
                {"range":[30,70],"color":"#F39C12"},
                {"range":[70,100],"color":"#E74C3C"}
            ]
        }
    ))
    fig.update_layout(
        height=330,
        margin=dict(l=20,r=20,t=60,b=20),
        paper_bgcolor="#0E1117",
        font_color="white"
    )
    st.plotly_chart(fig,use_container_width=True)

with right:
    fig=go.Figure(go.Indicator(
        mode="gauge+number",
        value=volatility_value,
        number={"suffix":"%","font":{"size":42}},
        title={"text":"Predicted Volatility","font":{"size":22}},
        gauge={
            "axis":{"range":[0,10]},
            "bar":{"color":"#3498DB"},
            "steps":[
                {"range":[0,2],"color":"#2ECC71"},
                {"range":[2,5],"color":"#F39C12"},
                {"range":[5,10],"color":"#E74C3C"}
            ]
        }
    ))
    fig.update_layout(
        height=330,
        margin=dict(l=20,r=20,t=60,b=20),
        paper_bgcolor="#0E1117",
        font_color="white"
    )
    st.plotly_chart(fig,use_container_width=True)

st.markdown("---")

inc=pd.DataFrame(shap["Top Drivers Increasing Risk"])
dec=pd.DataFrame(shap["Top Drivers Reducing Risk"])

c1,c2=st.columns(2)

with c1:
    st.subheader("📈 Risk Increasing Drivers")
    fig=px.bar(
        inc.sort_values("abs"),
        x="abs",
        y="description",
        orientation="h",
        text="shap_value"
    )
    fig.update_traces(
        marker_color="#E74C3C",
        texttemplate="%{text}",
        textposition="outside"
    )
    fig.update_layout(
        height=420,
        xaxis_title="SHAP Contribution",
        yaxis_title="",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="white",
        margin=dict(l=20,r=20,t=30,b=20)
    )
    st.plotly_chart(fig,use_container_width=True)

with c2:
    st.subheader("📉 Risk Reducing Drivers")
    fig=px.bar(
        dec.sort_values("abs"),
        x="abs",
        y="description",
        orientation="h",
        text="shap_value"
    )
    fig.update_traces(
        marker_color="#2ECC71",
        texttemplate="%{text}",
        textposition="outside"
    )
    fig.update_layout(
        height=420,
        xaxis_title="SHAP Contribution",
        yaxis_title="",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="white",
        margin=dict(l=20,r=20,t=30,b=20)
    )
    st.plotly_chart(fig,use_container_width=True)

a,b,c,d=st.columns(4)

with a:
    st.metric(
        "Risk Increasing",
        shap["Total Risk Increasing Contribution"]
    )

with b:
    st.metric(
        "Risk Reducing",
        shap["Total Risk Reducing Contribution"]
    )

with c:
    st.metric(
        "Strongest Increase",
        shap["Strongest Risk Increasing Driver"]
    )

with d:
    st.metric(
        "Strongest Reduction",
        shap["Strongest Risk Reducing Driver"]
    )

st.markdown("---")

report=response["llm_report"]
metrics=response["inference_metrics"]

st.subheader("AI Generated Financial Analysis")

sections=[
("Executive Summary","executive_summary"),
("Risk Assessment","risk_assessment"),
("Technical Analysis","technical_analysis"),
("News & Sentiment","news_and_sentiment"),
("Macroeconomic Environment","macroeconomic_environment"),
("SHAP Explanation","shap_explanation"),
("Historical Analysis","historical_analysis"),
("Model Consistency","model_consistency"),
("Limitations","limitations"),
("Final Conclusion","final_conclusion")
]

for title,key in sections:
    with st.expander(title,expanded=(key=="executive_summary")):
        st.write(report[key])

st.markdown("---")

st.subheader("Inference Performance")

a,b,c=st.columns(3)

with a:
    st.metric(
        "Feature Engineering",
        f'{metrics["feature building"]:.2f}s'
    )
    st.metric(
        "Classification",
        f'{metrics["classification"]:.2f}s'
    )

with b:
    st.metric(
        "Regression",
        f'{metrics["regression"]:.2f}s'
    )
    st.metric(
        "SHAP Values",
        f'{metrics["shap values"]:.2f}s'
    )

with c:
    st.metric(
        "LLM Report",
        f'{metrics["llm"]:.2f}s'
    )
    st.metric(
        "Total Time",
        f'{metrics["total time taken"]:.2f}s'
    )

st.markdown("---")

st.subheader("Market Trends")

left,right=st.columns(2)

with left:
    fig=px.line(
        price_history,
        x="date",
        y="close",
        title="Price History",
        markers=True
    )

    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="white",
        height=400
    )

    st.plotly_chart(fig,use_container_width=True)

with right:
    fig=go.Figure(data=[
        go.Candlestick(
            x=stock_history["Date"],
            open=stock_history["Open"],
            high=stock_history["High"],
            low=stock_history["Low"],
            close=stock_history["Close"]
        )
    ])

    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="white",
        height=400
    )

    st.plotly_chart(fig,use_container_width=True)

left,right=st.columns(2)

with left:
    fig=px.line(
        probability_history,
        x="date",
        y="probability",
        title="Historical Risk Probability",
        markers=True
    )

    fig.update_traces(line_color="#E74C3C")

    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="white",
        height=400
    )

    st.plotly_chart(fig,use_container_width=True)

with right:
    fig=px.line(
        volatility_history,
        x="date",
        y="volatility",
        title="Historical Predicted Volatility",
        markers=True
    )

    fig.update_traces(line_color="#3498DB")

    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="white",
        height=400
    )

    st.plotly_chart(fig,use_container_width=True)

st.markdown("---")

st.caption(
    f'Financial Risk Intelligence Platform | FastAPI • CatBoost • SHAP • LLM | API {metadata["api_version"]}'
)

