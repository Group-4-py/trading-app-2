# AutoTrader — AI-Powered Daily Trading System

An automated daily trading system that uses machine learning to predict next-day stock price movements for five major US companies. Built with Python, Streamlit, and scikit-learn.

**Live Application:** [https://autotrader4.streamlit.app/)

---

## Team Members

| Name | Role | Focus |
|------|------|-------|
| Bojana Belincevic | ML Engineer | ETL & Model Development |
| David Carrillo | Backend Developer | API Wrapper & Data Pipeline |
| Sebastião Clemente | Frontend Developer | App & Deployment |
| Bassem El Halawani | Data Analyst | Feature Engineering & Strategy |
| Theo Henry | Product Lead | Product Direction & Prioritization |
| Ocke Moulijn | Insights Analyst | Model Validation & Testing |

---

## Features

- **ML Predictions** — Binary classification model predicts whether each stock will go UP or DOWN the next trading day
- **5 Stocks Tracked** — AAPL, MSFT, GOOG, AMZN, NVDA
- **20+ Technical Features** — Returns, volatility, moving averages, RSI, MACD, Bollinger Bands, ATR, volume indicators
- **Live Trading Signals** — BUY / SELL / HOLD signals based on model confidence
- **Backtesting** — Compare Buy-and-Hold, Buy-and-Sell, and Benchmark strategies with customizable parameters
- **SimFin API Integration** — Real-time data via custom OOP API wrapper with rate limiting and error handling

---

## Installation

### Prerequisites
- Python 3.11
- A SimFin API key (free at [simfin.com](https://www.simfin.com/)) — optional, the app runs in demo mode without one

### Setup

```bash
# Clone the repository
git clone https://github.com/Group-4-py/trading-app-2.git
cd python-2-group-project

# Create and activate a virtual environment
conda create -n trading-app python=3.11
conda activate trading-app

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
cd app
streamlit run Home.py
```

The app will open in your browser. Enter your SimFin API key in the sidebar, or use demo mode (synthetic data).

### Retraining the Model (Optional)

To retrain the ML model from scratch:

1. Download `us-shareprices-daily.csv` from [SimFin Bulk Download](https://www.simfin.com/) (Share Prices → US)
2. Place it at `data/raw/us-shareprices-daily.csv`
3. Run:
   ```bash
   python ml/train_model.py
   ```

---

## Project Structure

```
.
├── app/                        # Streamlit web application
│   ├── Home.py                 # Landing page (entry point)
│   ├── pages/                  # Multi-page navigation
│   │   ├── 1_Go_Live.py       # Real-time predictions & signals
│   │   ├── 2_Model_Insights.py# ML model analysis & metrics
│   │   └── 3_Backtesting.py   # Strategy backtesting simulator
│   ├── utils/                  # Shared utility modules
│   │   ├── config.py           # Central configuration (tickers, features, theme)
│   │   ├── etl.py              # 10-step ETL pipeline
│   │   ├── model.py            # Model loading & inference wrapper
│   │   ├── pysimfin.py         # PySimFin API wrapper (OOP, Req 2.1)
│   │   ├── trading_strategy.py # Trading strategies & backtester
│   │   ├── charts.py           # Plotly chart builders
│   │   ├── data_helpers.py     # Data loading & synthetic fallback
│   │   └── style.py            # CSS theme injection
│   └── assets/                 # Logos, team photos, images
├── ml/                         # Machine learning
│   ├── train_model.py          # Model training script
│   └── model/                  # Saved model artifacts (.joblib)
├── etl/                        # CLI-based ETL scripts
│   ├── etl_share_prices.py     # Single-ticker ETL
│   ├── run_all_tickers.py      # Batch ETL for all tickers
│   └── etl_utils.py            # Shared ETL utilities
├── data/                       # Data directory
│   ├── raw/                    # Raw SimFin downloads (gitignored)
│   └── processed/              # Processed parquet feature files
├── notebooks/                  # Jupyter notebooks (ETL exploration)
├── docs/                       # Documentation
│   └── executive_summary.pdf   # Executive summary for stakeholders
├── requirements.txt            # Python dependencies
├── AI_USAGE_LOG.md             # AI tool usage documentation
└── README.md                   # This file
```

---

## System Architecture

| Phase | Component | Description |
|-------|-----------|-------------|
| **Offline (Part 1)** | SimFin Bulk → ETL → ML Model | Download historical data, process through 10-step ETL pipeline, train and export LogisticRegression model |
| **Online (Part 2)** | SimFin API → Wrapper → Streamlit | Fetch fresh data via PySimFin wrapper, apply ETL transformations, generate predictions, display in browser |

### Data Flow

```
SimFin (API or Bulk CSV)
        ↓
  PySimFin Wrapper / Raw CSV Loader
        ↓
  ETL Pipeline (10 steps: clean → returns → volatility → MAs → RSI → MACD → Bollinger → volume → ATR → target)
        ↓
  Feature Matrix (21 features + 4 ticker dummies)
        ↓
  LogisticRegression Model (StandardScaler → predict)
        ↓
  Streamlit App (Go Live · Model Insights · Backtesting)
```

---

## Web Application Pages

| Page | Description |
|------|-------------|
| **Home** | System overview, tracked companies, team profiles, technology stack |
| **Go Live** | Stock selector, real-time OHLCV data, model prediction (UP/DOWN), trading signal (BUY/SELL/HOLD), technical charts |
| **Model Insights** | Feature importance, confusion matrix, accuracy metrics, prediction distribution |
| **Backtesting** | Strategy simulator comparing Buy-and-Hold, Buy-and-Sell, and Benchmark with portfolio performance charts |

---

## Technologies

- **Python 3.11** — Core language
- **Streamlit** — Web application framework
- **scikit-learn** — Machine learning (LogisticRegression + StandardScaler pipeline)
- **Pandas** — Data manipulation and ETL
- **Plotly** — Interactive charts and visualizations
- **Requests** — HTTP client for SimFin API
- **joblib** — Model serialization
- **PyArrow** — Parquet file support
