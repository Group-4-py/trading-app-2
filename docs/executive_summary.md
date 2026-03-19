---
title: "AutoTrader — Executive Summary"
subtitle: "AI-Powered Daily Trading System"
author: "Bojana Belincevic, David Carrillo, Sebastião Clemente, Bassem El Halawani, Theo Henry, Ocke Moulijn"
date: "March 2026"
geometry: margin=2.5cm
fontsize: 11pt
---

# 1. Introduction

AutoTrader is an automated daily trading system that predicts next-day stock price movements for five major US companies using Machine Learning. The system combines a data analytics module (ETL pipeline and ML model) with an interactive web application built in Streamlit, allowing users to view real-time predictions, analyze model performance, and backtest trading strategies.

The five companies tracked are **Apple (AAPL)**, **Microsoft (MSFT)**, **Alphabet (GOOG)**, **Amazon (AMZN)**, and **NVIDIA (NVDA)**, representing a cross-section of the US technology and consumer sectors.

# 2. Data Sources

All financial data is sourced from **SimFin** (simfin.com), a platform providing free access to fundamental and market data for publicly traded companies.

- **Share Prices (Mandatory):** Daily OHLCV (Open, High, Low, Close, Volume) data covering approximately five years of trading history. This is the primary dataset used to define the ML model's features and target variable.
- **Company Information (Mandatory):** Metadata including ticker symbol, company name, industry sector, and other identifiers.
- **Financial Statements (Optional):** Quarterly income statements, balance sheets, and cash flow statements are accessible via the API wrapper but not used as model features in the current version.

For the offline training phase, data was obtained via SimFin's bulk download. For the online web application, data is fetched in real time through a custom Python API wrapper.

# 3. ETL Pipeline

The ETL (Extract, Transform, Load) pipeline converts raw share-price data into a feature-rich dataset suitable for machine learning. It is implemented in Python using Pandas and follows a 10-step process:

1. **Clean and validate** raw price data (standardize column names, handle missing values, remove duplicates)
2. **Daily returns** — 1-day, 5-day, and 10-day percentage changes
3. **Rolling volatility** — 10-day and 20-day standard deviation of returns
4. **Moving averages** — Simple (5, 10, 20, 50-day) and Exponential (12, 26-day), normalized by price
5. **RSI-14** — Relative Strength Index, a momentum oscillator
6. **MACD** — Moving Average Convergence Divergence with signal line
7. **Bollinger Bands** — Upper/lower bands and bandwidth (20-period, 2 standard deviations)
8. **Volume features** — 10-day volume moving average and volume ratio
9. **ATR-14** — Average True Range, a volatility measure
10. **Target generation** — Binary label: 1 if the next day's closing price is higher than today's, 0 otherwise

The same ETL pipeline is used in both the offline training script and the live web application, ensuring consistency between training and inference.

# 4. Machine Learning Model

The prediction task is **binary classification**: given today's market data and technical indicators, predict whether the stock price will go UP (1) or DOWN (0) the next trading day.

**Model architecture:**

- **Algorithm:** Logistic Regression with balanced class weights
- **Preprocessing:** StandardScaler (feature normalization)
- **Features:** 21 technical indicators plus 4 one-hot encoded ticker dummy variables (25 total)
- **Regularization:** Tested C values of 0.01, 0.1, 1.0, and 10.0; selected C=10.0

**Training methodology:**

- All five tickers are combined into a single dataset (5,950 samples after ETL)
- An 80/20 temporal train-test split preserves chronological order, preventing data leakage
- Balanced class weights address the slight imbalance between UP and DOWN days

**Performance:** The model achieves approximately 52% test accuracy, which is expected for daily stock price prediction — a notoriously difficult task. The system is designed to demonstrate a complete ML pipeline rather than achieve high predictive accuracy.

# 5. Web Application Architecture

The web application is built with **Streamlit** and structured as a multi-page app with four pages:

| Page | Purpose |
|------|---------|
| **Home** | System overview, tracked companies with logos, team member profiles, technology stack description |
| **Go Live** | Stock ticker selector, real-time OHLCV data via the PySimFin API wrapper, ETL applied to fresh data, model-generated trading signals (BUY/SELL/HOLD) displayed with confidence scores |
| **Model Insights** | Feature importance rankings, confusion matrix, accuracy/precision/recall/F1 metrics, prediction distribution analysis |
| **Backtesting** | Interactive strategy simulator comparing Buy-and-Hold, Buy-and-Sell, and Benchmark, with portfolio performance charts and metrics |

**PySimFin API Wrapper:** A custom object-oriented Python class handles all communication with the SimFin Data API. It includes rate limiting (max 2 requests/second), custom exception classes for authentication and rate-limit errors, and automatic response format conversion to Pandas DataFrames.

**Demo Mode:** When no API key is provided, the application generates realistic synthetic data using geometric Brownian motion, allowing full functionality without an active SimFin account.

# 6. Trading Strategies (Bonus)

Two ML-driven trading strategies are implemented alongside a benchmark:

- **Buy-and-Hold:** Scans the backtest for UP-prediction days, spreads capital across up to 12 bullish entry points, and never sells during the backtest.
- **Buy-and-Sell:** Active trading — buys when model confidence for UP exceeds 50%, sells on bearish signals, after 3 days maximum hold, or at a 2% stop-loss.
- **Benchmark:** Buys once on day one and holds the full period with no ML input, used as a baseline comparison.

Performance metrics include total return, annualized return, Sharpe ratio, maximum drawdown, win rate, and total trades.

# 7. Challenges and Conclusions

**Challenges encountered:**

- **Class imbalance:** The initial model exhibited a strong bias toward predicting a single class. Adding balanced class weights resolved this, producing meaningful predictions for both UP and DOWN days.
- **Feature stationarity:** Raw price levels are non-stationary, so all price-based features were normalized (divided by current price) to make them comparable across tickers and time periods.
- **API rate limits:** SimFin's free tier allows only 2 requests per second, requiring careful rate limiting in the API wrapper to avoid throttling during live data fetches.
- **Team coordination:** Managing concurrent development across ETL, ML, API, and frontend components required clear interfaces between modules and consistent use of Git branching.

**Conclusions:**

AutoTrader demonstrates a complete end-to-end machine learning trading system: from raw data ingestion through feature engineering, model training, real-time prediction, and interactive visualization. While daily stock prediction accuracy is inherently limited, the system showcases clean Python engineering, proper ML methodology (temporal splits, no data leakage), and a functional, user-friendly web interface. The modular architecture allows each component (ETL, model, API wrapper, strategies) to be improved independently.

**Deployment:** The application is hosted on Streamlit Cloud and publicly accessible without local installation.
