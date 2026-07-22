# 📊 Financial Risk Platform

An end-to-end **financial risk intelligence platform** that combines **technical indicators**, **FinBERT news sentiment**, and **macroeconomic data** with **tree-based ML models** to predict the short-term financial risk and volatility of publicly traded equities. Predictions are explained using **SHAP risk attribution** and rendered into a structured, evidence-grounded **LLM-generated risk report**.

---

## 🎯 What It Does

Given a company name (e.g. `"Google"`, `"JPMorgan Chase"`, `"NVIDIA"`), the platform:

1. Resolves the company to its stock ticker.
2. Pulls historical price data, recent news, and macroeconomic context.
3. Engineers technical features, FinBERT sentiment scores, and macro features.
4. Runs a **classifier** (`high risk` vs `low risk`) and a **volatility regressor**.
5. Produces **SHAP-based risk attribution** for explainability.
6. Feeds a structured, grounded context to an **LLM analyst** that writes a professional risk report.

The output is a financial risk profile, not investment advice.

---

## 🏗️ Architecture

```
                         User query (e.g. "Google")
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │  Ticker Resolution     │  ← LLM (Groq) + ticker map
                        └───────────┬───────────┘
                                    ▼
       ┌────────────────┬─────────────────┬────────────────┐
       │                │                 │                │
       ▼                ▼                 ▼                ▼
   Stock prices      Financial        Macroeconomic     (cached)
   (yfinance)        news (RSS/API)   data (FRED)
       │                │                 │
       ▼                ▼                 ▼
   Technical        FinBERT          Macro
   indicators       sentiment        features
       │                │                 │
       └────────────────┴─────────────────┘
                        │
                        ▼
                 Feature engineering
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
      Classifier               Regressor
      (High/Low Risk)          (Volatility %)
            │                       │
            └───────────┬───────────┘
                        ▼
                 SHAP risk attribution
                        ▼
              Structured LLM context
                        ▼
            LLM-generated risk report
                  (Groq + LangChain)
```

---

## 📁 Project Structure

```
FINANCIAL RISK PLATFORM/
│
├── frontend/
│   └── app.py                     # Streamlit/UI entry point
│
├── api/
│   └── main.py                    # FastAPI service
│
├── core_logic/
│   ├── exceptions.py              # Custom exception hierarchy
│   ├── utils/                     # Config, loggers, feature descriptions
│   │
│   ├── preprocessing/             # Raw data cleaning
│   │   ├── clean_stock_data.py
│   │   ├── clean_news_data.py
│   │   └── clean_macro_data.py
│   │
│   ├── features/                  # Feature engineering
│   │   ├── stock_features.py          # Technical indicators
│   │   ├── finbert_sentiment.py       # FinBERT inference
│   │   ├── generate_sentiment.py
│   │   ├── news_preperation.py
│   │   ├── macro_features.py
│   │   ├── target_engineering.py      # Risk / volatility labels
│   │   └── final_dataset.py           # Master merge
│   │
│   ├── train_models/
│   │   ├── training_classifier.py
│   │   └── training_regressor.py
│   │
│   └── inference/                 # End-to-end inference pipeline
│       ├── build_features_df.py       # Live feature construction
│       ├── predict_classifier.py
│       ├── predict_regressor.py
│       ├── predict.py                 # Orchestrator
│       ├── shap_explaination.py       # SHAP attribution
│       ├── llm_context.py             # Structured report context
│       └── llm_report.py              # LLM report generator
│
├── models/                        # Trained artifacts (.pkl)
├── data/                          # Raw + processed datasets
├── reports/                       # Generated reports
├── notebooks/                     # Exploratory work
│
├── prompts.py                     # LLM prompts (ticker + analyst)
├── tickers_map.py                 # Company-name → ticker dictionary
├── get_llm.py                     # LLM client (Groq)
├── test_entire_inference_pipeline.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Tools |
|---|---|
| Data Collection | `yfinance`, `fredapi`, news RSS/API |
| Data Processing | `pandas`, `numpy` |
| Technical Indicators | `ta`, `pandas-ta` (RSI, MACD, Bollinger, ATR, EMA, ADX, momentum) |
| Sentiment | `transformers` (FinBERT) |
| Classical ML | `scikit-learn`, `xgboost`, `lightgbm`, `catboost` |
| Explainability | `shap` |
| Deep Learning | `torch` (LSTM-ready) |
| LLM / Orchestration | `groq`, `langchain`, `langgraph`, `langchain-groq` |
| Visualization | `matplotlib`, `seaborn` |
| Serving | `FastAPI` (api/), Streamlit-style UI (frontend/) |
| Utilities | `tqdm`, `python-dotenv`, `joblib`, `nltk` |

---

## 🧠 Modeling

The platform trains **two complementary heads** on the same engineered feature set:

| Head | Target | Type | Output |
|---|---|---|---|
| Classifier | `high_risk` (0/1) | Gradient-boosted trees | Risk probability + class |
| Regressor | `future_volatility_20` | Gradient-boosted trees | Predicted volatility (%) |

Models are selected and saved after hyperparameter tuning. The latest training run uses **XGBoost / LightGBM**; CatBoost is supported in `requirements.txt` but disabled in the active environment due to a known package-compat issue (see `classifier_results.csv` / `regression_results.csv` for current metrics).

---

## 🔍 Explainability

- **SHAP** attribution is computed for every prediction. Strongest positive contributors push the model toward *high risk*; strongest negative contributors push toward *low risk*. Each feature is paired with a human-readable description from `core_logic/utils/feature_descriptions.py`.
- **Structured LLM context** is built from classifier output, regressor output, SHAP, technical indicators, news, and macro data. The LLM (Groq) is constrained by an extensive prompt in `prompts.py` that **forbids fabrication** of news, numbers, or advice and **never modifies supplied predictions**.

---

## 🚀 Getting Started

### 1. Clone & enter

```bash
git clone <your-repo-url>
cd "FINANCIAL RISK PLATFORM"
```

### 2. Create a virtual environment

```bash
python -m venv myvenv
# Windows
myvenv\Scripts\activate
# macOS / Linux
source myvenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
FRED_API_KEY=your_fred_api_key
```

### 5. (Optional) Retrain the models

```bash
python core_logic/preprocessing/clean_stock_data.py
python core_logic/preprocessing/clean_news_data.py
python core_logic/preprocessing/clean_macro_data.py
python core_logic/features/stock_features.py
python core_logic/features/finbert_sentiment.py
python core_logic/features/macro_features.py
python core_logic/features/target_engineering.py
python core_logic/features/final_dataset.py
python core_logic/train_models/training_classifier.py
python core_logic/train_models/training_regressor.py
```

Pretrained artifacts already live in `models/` and are loaded by the inference pipeline.

### 6. Run inference

**Programmatic smoke test:**

```bash
python test_entire_inference_pipeline.py
```

**API:**

```bash
uvicorn api.main:app --reload
```

**UI:**

```bash
python frontend/app.py
```

Then query with a company name, e.g. `"Google"`, `"JPMorgan"`, `"NVIDIA"`.

---

## 📊 Features Used

| Category | Examples |
|---|---|
| Price-based | Open, High, Low, Close, Volume |
| Trend | SMA, EMA, MACD, ADX |
| Momentum | RSI, momentum oscillators |
| Volatility | Bollinger Bands width, ATR, rolling std |
| Sentiment | FinBERT positive / negative / neutral scores from news |
| Macro | CPI, federal funds rate, treasury yields, etc. (FRED) |

---

## 📤 Outputs

Each inference call returns:

- Risk class (`low` / `high`) and probability
- Predicted 20-day volatility (%)
- Model confidence
- SHAP feature contributions (with human-readable descriptions)
- A fully-grounded LLM risk report covering:
  Executive Summary → Risk Assessment → Technical Analysis → News & Sentiment → Macroeconomic Environment → SHAP Explanation → Historical Analysis → Model Consistency → Limitations → Final Conclusion.

---

## ⚠️ Limitations & Disclaimer

- The platform is for **research and educational purposes only**.
- It does **not** provide investment, trading, or buy/sell advice.
- News coverage, macro context, and historical windows are limited by what is currently available through upstream sources.
- Predictions are probabilistic and can be wrong.

---

## 🛣️ Roadmap

- [ ] Full Streamlit/Flask dashboard wiring
- [ ] LSTM / Temporal Fusion Transformer sequence models
- [ ] Real-time news ingestion pipeline
- [ ] Portfolio-level risk aggregation
- [ ] Backtesting framework with risk-adjusted metrics
- [ ] Live API authentication + rate limiting
- [ ] Expanded ticker map and multi-region coverage

---

## 📜 License

For educational and research use.

## ✍️ Author

**B.Tech Python Project — Financial Risk Intelligence Platform.**
