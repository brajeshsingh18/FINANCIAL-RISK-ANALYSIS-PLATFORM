# 📊 Financial Risk Platform

A Python-based platform for analyzing and predicting financial market risk by combining **technical indicators**, **news sentiment analysis (FinBERT)**, and **machine learning** (scikit-learn, XGBoost, with PyTorch/LSTM ready for future use).

---

## 🎯 Overview

The platform ingests historical stock data and financial news, engineers technical features, scores news with FinBERT, trains risk models, and produces reports predicting market risk for selected equities.

## 🏗️ Project Structure

```
FINANCIAL RISK PLATFORM/
│
├── app.py                       # Main entry point (dashboard / orchestrator)
├── requirements.txt             # Python dependencies
├── .gitignore
├── README.md
│
├── core logic/                  # Pipeline modules
│   ├── get_data.py              # Fetch stock prices (yfinance) and news
│   ├── make_features.py         # Technical indicators + feature engineering
│   ├── sentiment.py             # FinBERT-based news sentiment scoring
│   ├── train_model.py           # Train ML risk models
│   ├── predict.py               # Generate predictions
│   └── create_report.py         # Build reports and visualizations
│
├── data/                        # Data storage (gitignored)
│   ├── stocks/                  # Raw historical price data
│   ├── news/                    # Raw and processed news
│   └── processed/               # Engineered features
│
├── models/                      # Saved trained models (gitignored)
└── reports/                     # Generated outputs (gitignored)
```

"CatBoost evaluation was planned but could not be completed due to a package compatibility issue in the development environment. The project compares Logistic Regression, Random Forest, LightGBM, XGBoost, and multiple regression models."

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| Data Collection | `yfinance`, `requests` |
| Data Processing | `pandas`, `numpy` |
| Technical Indicators | `ta` |
| Machine Learning | `scikit-learn`, `xgboost` |
| Deep Learning | `torch` (LSTM / TFT ready) |
| NLP & Sentiment | `transformers` (FinBERT) |
| Visualization | `matplotlib`, `seaborn` |
| Model Persistence | `joblib` |
| Utilities | `scipy`, `tqdm`, `python-dotenv` |

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd "FINANCIAL RISK PLATFORM"
```

### 2. Create and activate a virtual environment

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

### 4. Run the pipeline

```bash
# 1. Fetch stock + news data
python "core logic/get_data.py"

# 2. Engineer features and technical indicators
python "core logic/make_features.py"

# 3. Score news with FinBERT
python "core logic/sentiment.py"

# 4. Train the risk model
python "core logic/train_model.py"

# 5. Generate predictions
python "core logic/predict.py"

# 6. Build the report
python "core logic/create_report.py"
```

Or launch the full app:

```bash
python app.py
```

## 📈 Pipeline Flow

```
[Stock Prices] ──┐
                  ├──> Feature Engineering ──> Model Training ──> Predictions ──> Report
[Financial News] ─┘            │
                               └──> FinBERT Sentiment ──┘
```

## 📊 Features Used

- **Price-based**: Open, High, Low, Close, Volume
- **Technical Indicators** (via `ta`):
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Moving Averages (SMA, EMA)
  - ATR (Average True Range)
- **Sentiment**: FinBERT-based scoring of financial news headlines/articles
- **Volatility**: Rolling standard deviation, returns distribution

## 🤖 Models

- **Baseline**: scikit-learn classifiers / regressors
- **Primary**: XGBoost
- **Future**: LSTM / Temporal Fusion Transformer (PyTorch)

## 📁 Outputs

- `models/` — saved trained model artifacts
- `reports/` — risk score plots, feature importances, prediction summaries

## 🛣️ Roadmap

- [ ] Streamlit/Flask dashboard in `app.py`
- [ ] LSTM/TFT sequence models for multi-horizon forecasting
- [ ] Real-time news ingestion pipeline
- [ ] Portfolio-level risk aggregation
- [ ] Backtesting framework
- [ ] API for live predictions

## 📜 License

This project is for educational and research purposes.

## ✍️ Author

B.Tech Python Project — Financial Risk Analysis Platform
