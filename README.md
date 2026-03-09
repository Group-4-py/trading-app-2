**Group 4 Trading App - READ.ME**

# Part 1 – ETL & ML Engine

This folder contains the data pipeline and machine learning model that power our trading app. It transforms raw SimFin daily prices into engineered features and trains a classifier that predicts whether tomorrow’s close will be **up** or **down** for each stock. [file:112][web:120]

---

## Project structure

- `etl/etl_utils.py` – Core ETL and feature engineering logic (loading CSV, computing technical indicators, saving parquet).
- `etl/run_all_tickers.py` – Orchestrates ETL for all tickers and writes one processed file per ticker.
- `data/raw/` – Local raw data (large CSVs), **not tracked in Git**.
- `data/processed/` – Clean feature tables in parquet format (one file per ticker).
- `ml/train_model.py` – Trains the logistic regression model on all processed data.
- `ml/models/` – Saved model artifacts (`all_tickers_model.joblib`).

---

## Data & features

For each ticker and date we build a feature set similar to what the Streamlit app consumes. [file:112][web:120]

**Key columns:**

- Price & OHLCV: `open`, `high`, `low`, `close`, `adj_close`, `volume`, `price`.
- Returns: `return_1d`, `return_5d`, `return_10d`.
- Volatility: `volatility_10d`, `volatility_20d` (rolling std of returns).
- Moving averages: `sma_5`, `sma_10`, `sma_20`, `sma_50`.
- Trend & momentum: `ema_12`, `ema_26`, `macd`, `macd_signal`, `rsi_14`. [web:118][web:120]
- Bollinger bands: `bb_upper`, `bb_lower`, `bb_width`.
- Volume features: `volume_sma_10`, `volume_ratio`.
- Risk: `atr_14` (Average True Range).
- Target: `target` = 1 if next day close > today’s close, else 0.

All feature engineering is done in `etl_utils.py` so both the model and the Streamlit app can reuse the same logic. [file:112]

---

## How to run ETL

1. Activate the project environment:

   ```bash
   conda activate trading-app   # or your virtualenv
