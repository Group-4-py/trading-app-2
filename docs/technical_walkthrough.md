# AutoTrader — Complete Technical Walkthrough

This document explains every component of the AutoTrader application, how each script works, and how they connect to each other. It is written so that someone who knows Python but has never seen this codebase can fully understand and explain the project.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Configuration — `app/utils/config.py`](#3-configuration--apputilsconfigpy)
4. [The SimFin API Wrapper — `app/utils/pysimfin.py`](#4-the-simfin-api-wrapper--apputilspysimfinpy)
5. [The ETL Pipeline — `app/utils/etl.py`](#5-the-etl-pipeline--apputilsetlpy)
6. [Data Helpers & Demo Mode — `app/utils/data_helpers.py`](#6-data-helpers--demo-mode--apputilsdata_helperspy)
7. [Offline Model Training — `ml/train_model.py`](#7-offline-model-training--mltrain_modelpy)
8. [Model Loading & Inference — `app/utils/model.py`](#8-model-loading--inference--apputilsmodelpy)
9. [Trading Strategies — `app/utils/trading_strategy.py`](#9-trading-strategies--apputilstrading_strategypy)
10. [Chart Builders — `app/utils/charts.py`](#10-chart-builders--apputilschartspy)
11. [Custom Theme — `app/utils/style.py`](#11-custom-theme--apputilsstylepy)
12. [Page 1: Home — `app/Home.py`](#12-page-1-home--apphomepy)
13. [Page 2: Go Live — `app/pages/1_Go_Live.py`](#13-page-2-go-live--apppages1_go_livepy)
14. [Page 3: Model Insights — `app/pages/2_Model_Insights.py`](#14-page-3-model-insights--apppages2_model_insightspy)
15. [Page 4: Backtesting — `app/pages/3_Backtesting.py`](#15-page-4-backtesting--apppages3_backtestingpy)
16. [Standalone ETL Scripts — `etl/`](#16-standalone-etl-scripts--etl)
17. [How Everything Connects — Full Data Flow](#17-how-everything-connects--full-data-flow)
18. [Key Python Concepts Used](#18-key-python-concepts-used)

---

## 1. Project Overview

AutoTrader is a machine-learning-powered stock trading system that:

1. Fetches historical stock price data from the SimFin financial data platform
2. Transforms that raw data into 21 technical indicator features via an ETL pipeline
3. Trains a Logistic Regression classifier to predict whether a stock price will go UP or DOWN the next trading day
4. Serves predictions through an interactive Streamlit web application with four pages
5. Lets users backtest two ML-driven trading strategies against a benchmark

The system tracks five companies: Apple (AAPL), Microsoft (MSFT), Alphabet (GOOG), Amazon (AMZN), and NVIDIA (NVDA).

There are two distinct runtime modes:

- **Offline** — A training script (`ml/train_model.py`) reads a bulk CSV download from SimFin, runs the ETL pipeline, trains the model, and saves it as a `.joblib` file. This only needs to run once.
- **Online** — The Streamlit web app fetches fresh data (or generates synthetic demo data), re-runs the same ETL pipeline, loads the trained model, and displays predictions. This runs every time a user visits the app.

---

## 2. Project Structure

```
python-2-group-project/
│
├── app/                          # Streamlit web application
│   ├── Home.py                   # Entry point & landing page
│   ├── pages/                    # Streamlit multi-page convention
│   │   ├── 1_Go_Live.py          # Real-time predictions page
│   │   ├── 2_Model_Insights.py   # ML analysis page
│   │   └── 3_Backtesting.py      # Strategy simulator page
│   ├── utils/                    # Shared Python modules
│   │   ├── config.py             # All constants and settings
│   │   ├── pysimfin.py           # SimFin API wrapper (OOP)
│   │   ├── etl.py                # Feature engineering pipeline
│   │   ├── data_helpers.py       # Data loading & synthetic data generation
│   │   ├── model.py              # Model loading, inference, fallback
│   │   ├── trading_strategy.py   # Trading strategies & backtester
│   │   ├── charts.py             # All Plotly chart functions
│   │   └── style.py              # Custom CSS theme injection
│   ├── assets/                   # Logos, team photos
│   └── .streamlit/config.toml    # Streamlit theme configuration
│
├── ml/                           # Machine learning
│   ├── train_model.py            # Training script
│   └── model/
│       └── all_tickers_model.joblib  # Saved trained model
│
├── etl/                          # Standalone CLI ETL scripts
│   ├── etl_share_prices.py       # Single-ticker ETL CLI
│   ├── run_all_tickers.py        # Batch ETL for all tickers
│   └── etl_utils.py              # Shared ETL helpers
│
├── data/
│   ├── raw/                      # SimFin bulk CSV downloads
│   └── processed/                # Processed parquet files (one per ticker)
│
├── requirements.txt              # Python dependencies
└── docs/                         # Documentation
```

Streamlit uses a naming convention for multi-page apps: the main entry point is `Home.py`, and any `.py` files inside `pages/` automatically become navigation items in the sidebar. The numeric prefixes (`1_`, `2_`, `3_`) control the display order.

---

## 3. Configuration — `app/utils/config.py`

This is the central configuration file. Every other module imports constants from here instead of hardcoding values. This means you only need to change something in one place.

**What it defines:**

- `TICKERS` — A dictionary of the five tracked companies with their names, sectors, logo paths, and website URLs. Every page uses this to populate dropdown selectors and display company info.

- `TICKER_LIST` — A simple list `["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"]` derived from `TICKERS.keys()`.

- `SIMFIN_BASE_URL` and `SIMFIN_RATE_LIMIT` — The SimFin API base URL and the minimum delay between requests (0.5 seconds, which equals max 2 requests/second).

- `MODEL_PATH` — The directory where the trained model `.joblib` file lives (`ml/model/`).

- `MODEL_FEATURES` — A list of 21 feature column names that the model expects. This is critical: the ETL pipeline produces these exact columns, and the model was trained on them. The list includes `close`, `return_1d`, `return_5d`, `return_10d`, `volatility_10d`, `volatility_20d`, `sma_5`, `sma_10`, `sma_20`, `sma_50`, `ema_12`, `ema_26`, `rsi_14`, `macd`, `macd_signal`, `bb_upper`, `bb_lower`, `bb_width`, `volume_sma_10`, `volume_ratio`, `atr_14`.

- `TICKER_DUMMIES` — The one-hot encoded ticker columns used by the trained model: `["ticker_AMZN", "ticker_GOOG", "ticker_MSFT", "ticker_NVDA"]`. AAPL is the baseline (dropped during one-hot encoding with `drop_first=True`), so it does not appear in this list.

- `INITIAL_CAPITAL` and `TRANSACTION_COST` — Defaults for the backtesting simulator ($100,000 starting capital, 0.1% per trade).

- `COLORS` — A dictionary of hex color codes for the dark neon theme.

- `TEAM_MEMBERS` — A list of dictionaries with each team member's name, role, focus area, photo path, and LinkedIn URL.

**Why this matters:** Every module in the project imports from `config.py`. If you change a ticker symbol, add a new feature to the model, or adjust a default value, you do it here and every part of the application picks up the change automatically.

---

## 4. The SimFin API Wrapper — `app/utils/pysimfin.py`

This module is an **object-oriented API wrapper** that handles all communication with the SimFin financial data API. It demonstrates OOP principles: encapsulation, custom exception hierarchies, context managers, and a persistent HTTP session.

### Class Hierarchy

```python
SimFinAPIError(Exception)           # Base exception for all API errors
├── SimFinRateLimitError            # Raised on HTTP 429 (too many requests)
└── SimFinAuthError                 # Raised on HTTP 401 (invalid API key)
```

Each custom exception stores the HTTP `status_code` and a human-readable `message`. This allows calling code to catch specific error types:

```python
try:
    df = client.get_share_prices("AAPL")
except SimFinAuthError:
    print("Bad API key")
except SimFinRateLimitError:
    print("Slow down")
except SimFinAPIError as e:
    print(f"Other error: {e.status_code}")
```

### The `PySimFin` Class

**`__init__(self, api_key)`** — Stores the API key, creates the authorization header (`"api-key {key}"`), initializes a `requests.Session()` (reuses TCP connections for performance), and sets `_last_request_time` to 0.

**`_rate_limit(self)`** — A private method called before every API request. It checks how long ago the last request was made. If less than `SIMFIN_RATE_LIMIT` (0.5 seconds) has passed, it sleeps for the remaining time. This prevents hitting SimFin's free-tier limit of 2 requests/second.

**`_request(self, endpoint, params)`** — The core private method that makes HTTP GET requests. It:
1. Calls `_rate_limit()` to enforce the rate limit
2. Builds the full URL from `base_url + endpoint`
3. Makes the GET request via `self._session.get()` with a 30-second timeout
4. Catches connection errors, timeouts, and other request exceptions
5. Checks the HTTP status code: 401 raises `SimFinAuthError`, 429 raises `SimFinRateLimitError`, anything else non-200 raises `SimFinAPIError`
6. Returns the parsed JSON response

**`_compact_to_dataframe(self, data)`** — SimFin returns data in a "compact" format: `{"columns": [...], "data": [[...], [...]]}`. This method converts that into a Pandas DataFrame by passing the column names and row data to `pd.DataFrame()`.

**`get_share_prices(self, ticker, start, end)`** — The main public method. Calls the `/companies/prices/compact` endpoint, then renames SimFin's verbose column names (e.g., `"Last Closing Price"` → `"Close"`, `"Trading Volume"` → `"Volume"`) into standard OHLCV format. Converts the Date column to datetime and sorts chronologically.

**`get_financial_statement(self, ticker, statement, start, end, period)`** — Retrieves income statements (`"pl"`), balance sheets (`"bs"`), or cash flow statements (`"cf"`). Validates the statement type against a set of valid options.

**`get_company_info(self, ticker)`** — Returns a dictionary of company metadata (name, industry, etc.).

**Context manager support** — The class implements `__enter__` and `__exit__`, so you can use it with `with`:
```python
with PySimFin("my-key") as client:
    df = client.get_share_prices("AAPL")
# Session is automatically closed
```

**Where it's used:** `data_helpers.py` creates a `PySimFin` instance when the user provides an API key, calls `get_share_prices()`, and passes the returned DataFrame into the ETL pipeline.

---

## 5. The ETL Pipeline — `app/utils/etl.py`

ETL stands for Extract, Transform, Load. This module takes raw OHLCV price data and produces a feature-rich dataset ready for the ML model. It is implemented as a sequence of pure functions — each function takes a DataFrame and returns a new DataFrame with additional columns.

**This is the single most important module in the project** because it is used in three places:
1. The training script (`ml/train_model.py`) — to prepare training data
2. The web app (`data_helpers.py`) — to prepare live data for prediction
3. The standalone CLI scripts (`etl/etl_utils.py`) — to generate processed parquet files

Using the same ETL code everywhere guarantees that the features the model was trained on are identical to the features it receives at prediction time.

### Step-by-step breakdown

**Step 1: `clean_raw_prices(df)`** — Data cleaning and validation.
- Renames inconsistent column names (e.g., `"Adj. Close"` → `"adj_close"`)
- Lowercases all column names and replaces spaces/dots with underscores
- Converts the `date` column to `datetime` type
- Sorts by date and removes duplicate dates (keeps the last occurrence)
- Forward-fills missing values in price columns (`open`, `high`, `low`, `close`, `volume`) — this handles gaps in the data where a day might have a missing value
- Creates a `price` column that uses `adj_close` if available, otherwise falls back to `close`
- Drops any remaining rows where `price` is NaN

**Step 2: `add_return_features(df)`** — Computes percentage price changes.
- `return_1d` = 1-day return (today's price vs yesterday's)
- `return_5d` = 5-day return (today vs 5 days ago)
- `return_10d` = 10-day return (today vs 10 days ago)

Uses `pct_change()` which calculates `(current - previous) / previous`.

**Step 3: `add_volatility_features(df)`** — Rolling standard deviation of daily returns.
- `volatility_10d` = standard deviation of `return_1d` over the last 10 days
- `volatility_20d` = standard deviation of `return_1d` over the last 20 days

Higher volatility means the stock price has been fluctuating more.

**Step 4: `add_moving_averages(df)`** — Simple Moving Averages (SMA) and Exponential Moving Averages (EMA).
- SMA for windows of 5, 10, 20, and 50 days
- EMA for spans of 12 and 26 days

**Critically, all moving averages are normalized by dividing by the current price and subtracting 1.** For example, `sma_20 = rolling_mean(price, 20) / price - 1`. This means a value of -0.02 tells you "the 20-day average is 2% below the current price." This normalization is important because:
- It makes the features comparable across different stocks (a $200 stock and a $800 stock both produce values around 0)
- It makes the features stationary (they don't trend upward with the stock price over time)

**Step 5: `add_rsi(df, period=14)`** — Relative Strength Index, a momentum oscillator.
- Computes the average gain and average loss over the last 14 days
- RSI = 100 - (100 / (1 + average_gain / average_loss))
- RSI ranges from 0 to 100. Above 70 is "overbought" (might go down), below 30 is "oversold" (might go up)

**Step 6: `add_macd(df)`** — Moving Average Convergence Divergence.
- MACD line = 12-day EMA minus 26-day EMA, normalized by price
- Signal line = 9-day EMA of the MACD line
- When MACD crosses above the signal line, it's a bullish signal; when it crosses below, it's bearish

**Step 7: `add_bollinger_bands(df, window=20, num_std=2.0)`** — Price bands based on volatility.
- Middle band = 20-day SMA
- Upper band = middle + 2 standard deviations
- Lower band = middle - 2 standard deviations
- `bb_width` = distance between upper and lower bands, normalized by price
- All values are normalized by current price

**Step 8: `add_volume_features(df)`** — Volume-based indicators.
- `volume_sma_10` = 10-day average volume, normalized to [0, 1] range
- `volume_ratio` = today's volume divided by the 10-day average volume. Values above 1 mean higher-than-normal trading activity.

**Step 9: `add_atr(df, period=14)`** — Average True Range, a volatility measure.
- True Range = max(high-low, |high-prev_close|, |low-prev_close|)
- ATR = 14-day rolling average of True Range, normalized by price
- Falls back to `volatility_10d` if high/low/close columns are missing

**Step 10: `add_target(df)`** — Generates the binary label the model is trained to predict.
- `target = 1` if tomorrow's closing price > today's closing price (price went UP)
- `target = 0` otherwise (price went DOWN or stayed flat)

Uses `shift(-1)` to look at the next day's price. The last row will always be NaN (no "next day" exists) and is dropped.

### The orchestrator: `run_etl(df, include_target=True)`

This function calls all 10 steps in sequence:
```python
df = clean_raw_prices(df)
df = add_return_features(df)
df = add_volatility_features(df)
df = add_moving_averages(df)
df = add_rsi(df)
df = add_macd(df)
df = add_bollinger_bands(df)
df = add_volume_features(df)
df = add_atr(df)
if include_target:
    df = add_target(df)
```

After all steps, it drops rows with NaN values in the essential feature columns. These NaN values come from the rolling window warmup period — for example, `sma_50` needs 50 days of data before it can produce its first value, so the first 49 rows will have NaN for that column.

The `include_target` parameter exists because:
- During training, you need the target column to train the model (`include_target=True`)
- During live prediction, you don't need (and can't compute) the target because you don't know tomorrow's price yet. However, in this app, the target is always included because it's used to compute historical accuracy.

---

## 6. Data Helpers & Demo Mode — `app/utils/data_helpers.py`

This module is the **bridge between the data sources and the ETL pipeline**. It decides where data comes from and provides cached, ready-to-use DataFrames to the Streamlit pages.

### Synthetic Data Generator: `generate_demo_prices(ticker, days=504)`

When no API key is provided, this function generates realistic fake stock price data using **Geometric Brownian Motion (GBM)** — the same mathematical model used in the Black-Scholes option pricing formula.

The process:
1. Look up base parameters for the ticker (starting price, daily drift, daily volatility). For example, NVDA starts at $790 with higher volatility (0.028) than MSFT at $410 (0.016).
2. Generate random daily log-returns from a normal distribution: `log_returns = rng.normal(drift, vol, days)`
3. Compute cumulative prices: `close_prices = base_price * exp(cumsum(log_returns))`
4. Generate Open, High, Low from the Close price by adding random intraday ranges
5. Generate realistic volume using a log-normal distribution

The random number generator uses `RandomState(hash(ticker))`, so the same ticker always produces the same synthetic data — this makes the demo reproducible.

### Data Loading Functions

**`load_price_data(ticker, days, api_key)`** — The primary data loading function. Decorated with `@st.cache_data(ttl=300)`, which means Streamlit caches the result for 5 minutes — repeated calls with the same arguments return the cached DataFrame instantly without re-fetching.

Priority:
1. If `api_key` is provided → create a `PySimFin` client and call `get_share_prices()`
2. If the API returns insufficient data (< 50 rows) or fails → fall back to demo data
3. If no `api_key` → use demo data directly

**`load_processed_data(ticker, days, api_key)`** — Calls `load_price_data()` then passes the result through `run_etl()`. Also cached for 5 minutes.

**`get_latest_features(ticker, api_key)`** — Returns only the most recent row of processed data, containing the latest feature values. This is used on the Go Live page to make today's prediction.

**`get_prediction_history(ticker, model, n_days, api_key)`** — Generates a table of past predictions:
1. Loads processed data (extra rows to allow for warmup)
2. Takes the last `n_days` rows
3. Runs `model.predict()` and `model.predict_proba()` on each row
4. Builds a DataFrame with columns: `date`, `price`, `prediction`, `prob_down`, `prob_up`, `confidence`, `actual`
5. The `actual` column comes from the `target` column in the ETL output, so you can compare what the model predicted vs what actually happened

---

## 7. Offline Model Training — `ml/train_model.py`

This script runs once, offline, to produce the trained model that the web app loads. It is a standard Python script (not a Streamlit page) that you run from the command line:

```bash
python ml/train_model.py
```

### Step-by-step

**Step 1: Load raw CSV** — `load_raw_csv(csv_path)`

The SimFin bulk download is a large semicolon-separated CSV file containing daily prices for thousands of US stocks. The function reads it in chunks of 250,000 rows (to manage memory) and keeps only rows for our five tickers. Uses `pd.read_csv(csv_path, sep=";", usecols=PRICE_COLUMNS, chunksize=250_000)`.

**Step 2: Apply ETL per ticker** — `build_dataset(raw_df)`

For each ticker, it filters the raw data and calls `run_etl(df, include_target=True)` — the exact same function used by the web app. Each ticker's processed DataFrame gets a `ticker` column added, then all five are concatenated into one combined dataset (~5,950 rows total).

**Step 3: Build feature matrix** — `prepare_features(df)`

The `ticker` column is one-hot encoded using `pd.get_dummies(df, columns=["ticker"], drop_first=True)`. With `drop_first=True`, AAPL (alphabetically first) becomes the implicit baseline — it has no dummy column. The remaining four tickers get columns: `ticker_AMZN`, `ticker_GOOG`, `ticker_MSFT`, `ticker_NVDA`.

The final feature matrix combines the 21 technical indicators from `MODEL_FEATURES` with the 4 ticker dummy columns = 25 features total.

`X` is the feature matrix, `y` is the target column (0 or 1).

**Step 4: Train** — `train_model(X, y)`

1. **Temporal train-test split** — `train_test_split(X, y, test_size=0.2, shuffle=False)`. The `shuffle=False` is critical: it means the first 80% of the data (earlier dates) is training, and the last 20% (later dates) is testing. Shuffling would be wrong for time series data because it would leak future information into training.

2. **Hyperparameter search over regularization strength** — Tests C values of 0.01, 0.1, 1.0, and 10.0. C controls how much the model is allowed to fit closely to the training data (higher C = less regularization = more freedom to fit). Each value is tested by:
   - Building a `Pipeline([StandardScaler(), LogisticRegression(C=C, class_weight="balanced")])` — StandardScaler normalizes features to zero mean and unit variance, which is important for Logistic Regression
   - Fitting on the training set
   - Computing accuracy on the test set
   - Keeping the best pipeline

3. **`class_weight="balanced"`** — This tells sklearn to automatically adjust the weight of each class inversely proportional to its frequency. If there are slightly more UP days than DOWN days, DOWN days get higher weight during training. This prevents the model from just predicting the majority class for everything.

4. Prints a classification report showing precision, recall, and F1 for both classes.

**Step 5: Save** — `save_model(pipeline, path)`

Uses `joblib.dump()` to serialize the entire sklearn Pipeline (StandardScaler + LogisticRegression) to `ml/model/all_tickers_model.joblib`.

After saving, the script prints a reminder to verify that `TICKER_DUMMIES` in `config.py` matches the dummy columns generated during training.

---

## 8. Model Loading & Inference — `app/utils/model.py`

This module handles loading the trained model and providing a consistent interface for predictions across the app.

### `load_model(ticker=None)` — The entry point

1. Computes the path to `ml/model/all_tickers_model.joblib` relative to the project root
2. If the file exists, loads it with `joblib.load()` and wraps it in a `ModelWrapper`
3. If the file doesn't exist (e.g., on a fresh deployment where training hasn't been run), returns a `DummyClassifier` instead

This fallback mechanism means the app always works — even without a trained model.

### `ModelWrapper` — Wrapping the trained sklearn model

The `ModelWrapper` class wraps the loaded sklearn Pipeline and adds functionality needed by the app:

**`predict(self, X)` and `predict_proba(self, X)`** — These don't just call the underlying model directly. They first call `_prepare_model_input()` to align the input features with what the model expects:

1. **Add ticker dummy columns** — When Go Live predicts for NVDA, it sets `ticker_NVDA = 1.0` and all other ticker dummies to 0.0
2. **Handle missing columns** — If any expected feature is missing from the input, it's added as 0.0
3. **Reorder columns** — The final DataFrame columns are reordered to match `expected_feature_names` exactly

**`get_feature_importance(self)`** — Extracts the absolute values of the Logistic Regression coefficients, normalizes them to sum to 1, and returns them as a DataFrame sorted by importance. For a Pipeline, it drills into `named_steps["clf"]` to find the classifier.

**`get_model_metrics(self)`** — Returns metadata about the model (type, file path, whether it's a dummy).

### `DummyClassifier` — The heuristic fallback

When no trained model is available, this class provides reasonable predictions using hand-coded rules based on technical indicators:

- Starts with a base score of 0.5 (50% UP probability)
- If `return_1d` is positive, increases the score; if negative, decreases it
- If `rsi_14 < 30` (oversold), adds 0.15 to the score; if `rsi_14 > 70` (overbought), subtracts 0.15
- If MACD is above the signal line, adds 0.08 (bullish); otherwise subtracts 0.08
- Adds Gaussian noise for realism
- Clips the final score to [0.05, 0.95]

This isn't a real ML model, but it produces sensible-looking predictions that demonstrate the app's functionality.

### `calculate_model_metrics(model, X, y, train_ratio=0.8)`

Evaluates a model against data using an 80/20 temporal split:
1. Takes the last 20% of rows as the test set
2. Runs `model.predict()` and `model.predict_proba()` on the test set
3. Computes accuracy, precision, recall, F1, and AUC-ROC using sklearn metrics
4. Returns everything as a dictionary

Used by the Model Insights page to display performance metrics.

---

## 9. Trading Strategies — `app/utils/trading_strategy.py`

This module implements three trading strategies that simulate a portfolio over historical data. They all receive arrays of predictions/probabilities and prices, and return a DataFrame recording every day's portfolio state.

### Strategy 1: `strategy_buy_and_hold(predictions, prices, initial_capital, ...)`

<<<<<<< HEAD
A dollar-cost averaging approach:
=======
The Buy-and-Hold strategy uses ML-guided staged accumulation:
>>>>>>> 8390a99 (Updated trading strategy with new Buy & Hold definition)
1. Scans all days where the model predicts UP
2. Selects `n_buy_ins` (default 12) evenly spaced entry points from those bullish days using `np.linspace`
3. Splits the initial capital equally across those entry points
4. On each selected day, buys as many shares as possible with the allocated amount (minus transaction costs)
5. Never sells — accumulates shares over time
6. The last buy-in deploys any remaining cash to avoid leaving a small uninvested balance

<<<<<<< HEAD
This strategy is conservative — it relies on the model only for timing its purchases.
=======
This strategy is passive and long-only — it relies on the model only for timing its staged entries, not for exits.
>>>>>>> 8390a99 (Updated trading strategy with new Buy & Hold definition)

### Strategy 2: `strategy_buy_and_sell(probabilities, prices, initial_capital, ...)`

An active trading strategy:
1. **Buy** when P(UP) >= 0.50 and not currently holding shares
2. **Sell** when any of these conditions is true:
   - Model turns bearish: P(UP) < 0.50
   - Maximum hold period reached: held for 3 days
   - Stop-loss triggered: unrealized loss >= 2%
3. After selling, waits one bar before re-entering (the `elif` structure prevents same-day flip)

This generates many more trades because it responds to every model signal change.

### Benchmark: `benchmark_buy_and_hold(prices, initial_capital)`

<<<<<<< HEAD
The simplest possible strategy: buy as many shares as possible on day 1, hold forever. No ML involved. Used as a baseline to evaluate whether the model adds value.
=======
The classic benchmark: buy as many shares as possible on day 1 and hold for the full period. No ML involved. Used as a baseline to evaluate whether the model adds value.
>>>>>>> 8390a99 (Updated trading strategy with new Buy & Hold definition)

### `compute_strategy_metrics(backtest_df, initial_capital)`

Computes standard portfolio performance metrics from a backtest result:
- **Total return** — `(final_value - initial) / initial`
- **Annualized return** — Extrapolates the total return to a 252-trading-day year
- **Sharpe ratio** — `mean(daily_returns) / std(daily_returns) * sqrt(252)`. Measures risk-adjusted return. Higher is better.
- **Maximum drawdown** — The largest peak-to-trough decline. Computed as `(portfolio_value - cummax) / cummax`
- **Win rate** — Of completed sell trades, what fraction sold at a higher price than the most recent buy

---

## 10. Chart Builders — `app/utils/charts.py`

This module contains 14 functions that each build a specific Plotly chart. All charts use a consistent dark theme defined at the top of the file with neon accent colors (cyan, green, red, purple, yellow).

A helper function `_apply_defaults(fig, height, **kwargs)` applies the standard layout (font family, transparent background, grid colors, hover styling) to every chart.

**Price charts:**
- `candlestick_chart()` — OHLCV candlestick with volume bars in a subplot below
- `price_with_ma_chart()` — Price line with SMA overlays and Bollinger Band shading
- `rsi_chart()` — RSI-14 with colored overbought (70) and oversold (30) zones
- `macd_chart()` — MACD line, signal line, and histogram bars

**Prediction charts:**
- `prediction_timeline()` — Price line with green/red triangle markers for UP/DOWN predictions
- `confidence_gauge()` — Semicircular gauge showing prediction probability (0-100%)
- `prediction_distribution_chart()` — Histogram of P(UP) values split at the 0.50 decision boundary
- `accuracy_over_time_chart()` — Rolling accuracy line chart with a 50% baseline

**Model analysis:**
- `feature_importance_chart()` — Horizontal bar chart of top 15 features by importance
- `confusion_matrix_chart()` — Heatmap with correct cells in green, incorrect in red

**Backtesting charts:**
- `portfolio_chart()` — Multi-line chart comparing portfolio value across strategies
- `drawdown_chart()` — Area chart showing portfolio dips from peak value
- `trade_actions_chart()` — Price line with BUY/SELL triangle markers

**Other:**
- `returns_distribution_chart()` — Histogram of daily returns with a normal distribution overlay (uses `scipy.stats.norm`)

---

## 11. Custom Theme — `app/utils/style.py`

This module injects a full custom CSS stylesheet into every Streamlit page via `st.markdown()` with `unsafe_allow_html=True`.

The `inject_custom_css()` function is called at the top of every page. It imports three Google Fonts:
- **Orbitron** — A futuristic geometric font used for headings
- **Rajdhani** — A clean technical font used for body text
- **JetBrains Mono** — A monospace font for code-like elements

It defines CSS custom properties (variables) for the color palette and overrides Streamlit's default styling for:
- The app background, sidebar, and header
- Metric cards (glass-effect cards with hover glow animations)
- Tabs (custom styling with gradient active indicator)
- Buttons (neon cyan gradient with glow effect)
- DataFrames and tables (dark theme integration)
- Custom classes like `.glass-card`, `.feature-box`, `.team-card`, `.hero-bg`
- Trading signal boxes (`.signal-up`, `.signal-down`, `.signal-hold`) with colored borders and glow effects
- Prediction badges (`.badge-up`, `.badge-down`) with colored backgrounds

---

## 12. Page 1: Home — `app/Home.py`

The entry point and landing page. Run with `streamlit run app/Home.py`.

**Sidebar** — Displays navigation info and an API key input. The key is stored in `st.session_state["api_key_stored"]` so other pages can access it. If empty, shows "Demo Mode" message.

**Hero section** — Loads the app logo from `assets/logo.png`, encodes it as base64, and embeds it in HTML. This is done because Streamlit's file serving doesn't always work for custom HTML.

**Key stats banner** — Four `st.metric()` cards showing stocks tracked, model type, feature count, and signal frequency.

**System architecture** — Two glass cards explaining Part 1 (offline analytics) and Part 2 (online web app).

**Data flow diagram** — An ASCII art diagram inside an expander showing how data flows from SimFin through the ETL, model, and into the web app pages.

**Companies we track** — Displays the five ticker cards with logos, using a loop over `TICKERS.items()`. Each logo is loaded, base64-encoded, and embedded in HTML.

**How it works** — Four step cards (Data Ingestion → ETL → ML Prediction → Signal).

**Development team** — Team member cards with photos, roles, and LinkedIn links. Photos are base64-encoded like the logos.

**Technology stack** — Five cards for Python, Streamlit, Scikit-learn, Plotly, and Streamlit Cloud.

---

## 13. Page 2: Go Live — `app/pages/1_Go_Live.py`

The real-time prediction page. This is where the complete online data flow executes.

**Sidebar** — Ticker selector dropdown, historical data slider (60-504 trading days), prediction history slider (10-120 days).

**Data loading** (lines 105-108):
```python
raw_df = load_price_data(selected_ticker, days=lookback_days, api_key=api_key)
processed_df = load_processed_data(selected_ticker, days=lookback_days, api_key=api_key)
model = load_model(selected_ticker)
```
This is the core pipeline execution. `load_price_data` fetches raw OHLCV. `load_processed_data` calls `load_price_data` internally and then runs the ETL. `load_model` loads the trained model (or DummyClassifier).

**Latest prediction** (lines 113-167):
1. `get_latest_features()` returns the most recent row of processed features
2. `model.predict()` returns 0 (DOWN) or 1 (UP)
3. `model.predict_proba()` returns probabilities for both classes
4. Determines the trading signal: BUY if UP with >30% confidence, SELL if DOWN with >30% confidence, HOLD otherwise
5. Displays the signal in a styled card with two confidence gauges

**Key price metrics** — Shows the latest close price with change percentage, RSI, volatility, MACD, and ATR.

**Market data analysis** — Three tabs:
- Candlestick chart with volume
- Technical analysis (price with moving averages and Bollinger Bands)
- Indicators (RSI and MACD charts side by side)

**Prediction history** — Calls `get_prediction_history()` to generate predictions for the last N days. Shows summary metrics (total predictions, UP/DOWN split, average confidence, accuracy vs actual outcomes) and three charts: prediction timeline, probability distribution, and rolling accuracy.

**Feature snapshot** — Displays the raw feature values for the most recent day in a two-column table.

**Returns analysis** — Daily returns distribution histogram with normal curve overlay.

**Raw data explorer** — An expander with two tabs showing the last 50 rows of raw and processed data.

---

## 14. Page 3: Model Insights — `app/pages/2_Model_Insights.py`

This page explains the ML pipeline and shows model performance metrics.

**Data loading** (lines 54-67):
```python
model = load_model(selected_ticker)
processed_df = load_processed_data(selected_ticker, days=504, api_key=api_key)
pred_history = get_prediction_history(selected_ticker, model, n_days=120, api_key=api_key)
```
Then calls `calculate_model_metrics()` to compute accuracy, precision, recall, F1, and AUC-ROC using an 80/20 temporal split on the processed data.

**Model overview** — Three glass cards explaining the problem definition (binary classification), feature engineering categories, and dataset info.

**Performance metrics** — Five metric cards showing accuracy, precision, recall, F1, and AUC-ROC. An expander explains what each metric means.

**Feature importance** — Horizontal bar chart (from `model.get_feature_importance()`) alongside a ranked table of top 10 features.

**Confusion matrix** — Generated from the prediction history by comparing `prediction` vs `actual`. Displayed as a heatmap with a classification report table.

**Prediction analysis** — Three tabs:
- Rolling accuracy over time (with adjustable window size)
- Probability distribution histogram
- Feature correlation heatmap (top 12 features)

**ML Pipeline Details** — Three tabs with text explanations of the ETL process, model architecture, and evaluation methodology.

---

## 15. Page 4: Backtesting — `app/pages/3_Backtesting.py`

The strategy simulation page. Users configure parameters and compare three trading strategies.

**Sidebar controls:**
- Ticker selector
- Initial capital (default $100,000)
- Transaction cost in basis points (default 10 bps = 0.1%)
- Backtest period (60-504 trading days)
- Checkboxes to enable/disable each of the three strategies
- "Run Backtest" button

<<<<<<< HEAD
**Strategy definitions** — Three glass cards explaining Buy & Hold (ML), Buy & Sell (ML), and Benchmark (No ML).
=======
**Strategy definitions** — Three glass cards explaining Buy & Hold, Buy & Sell, and Benchmark.
>>>>>>> 8390a99 (Updated trading strategy with new Buy & Hold definition)

**Backtest execution** (lines 174-223):

The page tracks a "fingerprint" of the current parameters. If any parameter changes (or the user clicks Run Backtest), it re-runs the simulation:

1. Load processed data and model for the selected ticker
2. Extract features and run `model.predict()` and `model.predict_proba()` on all rows
3. Call each enabled strategy function with the predictions, prices, and capital
4. Call `compute_strategy_metrics()` for each result
5. Store everything in `st.session_state` so it persists across Streamlit reruns

**Results display:**
- **Portfolio performance chart** — Multi-line chart comparing all strategies
- **Strategy comparison table** — DataFrame with total return, annualized return, Sharpe ratio, max drawdown, win rate, total trades, and final portfolio value
- **Best strategy highlight** — Four metric cards for the top performer
- **Detailed strategy analysis** — Dropdown to select one strategy for deep analysis:
  - Equity curve (portfolio value over time)
  - Drawdown analysis
  - Trade actions (BUY/SELL markers on the price chart)
  - Trade log table (every BUY and SELL with date, price, shares, cash, and portfolio value)
- **Monthly returns analysis** — Bar chart of monthly returns, color-coded green (positive) or red (negative)

---

## 16. Standalone ETL Scripts — `etl/`

These are command-line scripts for running the ETL pipeline outside the web app, producing processed parquet files in `data/processed/`.

**`etl/etl_utils.py`** — Shared utilities:
- Adds `app/` to `sys.path` so it can import `utils.config` and `utils.etl`
- `load_share_prices()` — Reads the bulk CSV in chunks, filtering for requested tickers
- `filter_ticker()` — Filters one ticker from the loaded data
- `run_etl_for_ticker()` — Loads data and runs ETL for a single ticker
- `run_etl_for_ticker_from_df()` — Runs ETL from an already-loaded DataFrame and saves to parquet

**`etl/etl_share_prices.py`** — CLI for processing one ticker:
```bash
python etl/etl_share_prices.py --ticker AAPL
```

**`etl/run_all_tickers.py`** — Batch processing for all five tickers:
```bash
python etl/run_all_tickers.py
```
Loads the CSV once, then loops through all tickers and saves each to `data/processed/{TICKER}.parquet`.

The parquet files in `data/processed/` (AAPL.parquet, AMZN.parquet, etc.) are the output of these scripts. They're not directly used by the web app (which runs ETL on the fly) but serve as pre-processed datasets for analysis in notebooks.

---

## 17. How Everything Connects — Full Data Flow

### Connection Map

```
config.py ◄──── imported by every module for constants
    │
    ├── pysimfin.py (uses SIMFIN_BASE_URL, SIMFIN_RATE_LIMIT)
    ├── etl.py (standalone — no config imports, but produces MODEL_FEATURES columns)
    ├── data_helpers.py (uses TICKERS, TICKER_LIST)
    ├── model.py (uses MODEL_PATH, MODEL_FEATURES, TICKER_DUMMIES)
    ├── trading_strategy.py (uses INITIAL_CAPITAL, TRANSACTION_COST)
    └── All pages (use TICKERS, TICKER_LIST, MODEL_FEATURES, etc.)
```

### Offline Training Flow

```
SimFin Bulk CSV (data/raw/us-shareprices-daily.csv)
        │
        ▼
ml/train_model.py
        │
        ├── load_raw_csv() ──── reads CSV in chunks, filters 5 tickers
        │
        ├── build_dataset() ──── for each ticker:
        │       │                    calls app/utils/etl.py → run_etl()
        │       │                    (same code the web app uses)
        │       └── concatenates all tickers into one DataFrame
        │
        ├── prepare_features() ── one-hot encodes tickers, selects features
        │
        ├── train_model() ──── StandardScaler + LogisticRegression Pipeline
        │                      Tests C=0.01, 0.1, 1.0, 10.0
        │                      Temporal 80/20 split (no shuffling)
        │
        └── save_model() ──── writes ml/model/all_tickers_model.joblib
```

### Online Web App Flow (what happens when a user visits a page)

```
User visits a Streamlit page
        │
        ▼
Page calls data_helpers.load_processed_data(ticker, days, api_key)
        │
        ├── Calls load_price_data(ticker, days, api_key)
        │       │
        │       ├── If api_key provided:
        │       │       Creates PySimFin(api_key)
        │       │       Calls client.get_share_prices(ticker, start, end)
        │       │       Returns raw OHLCV DataFrame
        │       │
        │       └── Else (or on API failure):
        │               Calls generate_demo_prices(ticker, days)
        │               Returns synthetic OHLCV DataFrame via GBM
        │
        └── Calls etl.run_etl(raw_df, include_target=True)
                Returns processed DataFrame with 21 features + target
        │
        ▼
Page calls model.load_model(ticker)
        │
        ├── If ml/model/all_tickers_model.joblib exists:
        │       Loads with joblib.load()
        │       Returns ModelWrapper(pipeline, ticker)
        │
        └── Else:
                Returns DummyClassifier(seed=hash(ticker))
        │
        ▼
Page uses model.predict(features) and model.predict_proba(features)
        │
        ├── Go Live: displays signal (BUY/SELL/HOLD) + confidence
        ├── Model Insights: computes metrics, feature importance
        └── Backtesting: runs strategies on predictions
                │
                ├── strategy_buy_and_hold(predictions, prices, ...)
                ├── strategy_buy_and_sell(probabilities, prices, ...)
                └── benchmark_buy_and_hold(prices, ...)
                        │
                        └── compute_strategy_metrics() for each
```

### How `st.session_state` connects pages

The Home page stores `st.session_state["api_key_stored"]` when the user enters their SimFin API key. Every other page reads this value to decide whether to use the API or demo mode. This is how Streamlit's multi-page apps share state — through the session state dictionary.

---

## 18. Key Python Concepts Used

This section highlights the Python techniques and patterns used throughout the project, for reference when explaining to a professor.

**Object-Oriented Programming (OOP):**
- `PySimFin` class with encapsulation (private methods prefixed with `_`), a persistent session, and context manager protocol (`__enter__`/`__exit__`)
- Custom exception hierarchy (`SimFinAPIError` → `SimFinRateLimitError`, `SimFinAuthError`)
- `ModelWrapper` and `DummyClassifier` providing the same interface (`predict`, `predict_proba`, `get_feature_importance`) — a form of duck typing / polymorphism

**Functional Pipeline Pattern:**
- The ETL pipeline chains pure functions: each takes a DataFrame, adds columns, returns a new DataFrame
- `run_etl()` orchestrates the chain in sequence

**Decorators:**
- `@st.cache_data(ttl=300)` on data loading functions — caches results for 5 minutes to avoid redundant API calls or ETL runs

**Data Processing with Pandas:**
- Method chaining, rolling windows, exponential weighted means
- `pct_change()`, `shift()`, `clip()`, `ffill()`, `get_dummies()`
- GroupBy operations for monthly returns in backtesting

**NumPy:**
- Vectorized operations for strategy simulation
- `np.linspace` for evenly spaced buy-in points
- `np.flatnonzero` to find indices matching conditions
- Random number generation with `RandomState` for reproducibility

**Scikit-learn:**
- `Pipeline` combining `StandardScaler` + `LogisticRegression`
- `train_test_split` with `shuffle=False` for temporal ordering
- Classification metrics: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`
- `classification_report` and `confusion_matrix`

**File I/O:**
- `joblib.dump()`/`joblib.load()` for model serialization
- `pd.read_csv()` with chunked reading for large files
- `.to_parquet()` for efficient columnar storage
- Base64 encoding of images for inline HTML embedding

**Web Application (Streamlit):**
- Multi-page app convention (`Home.py` + `pages/` directory)
- `st.session_state` for cross-page state sharing
- Layout with `st.columns()`, `st.tabs()`, `st.expander()`
- Custom HTML/CSS injection via `st.markdown(unsafe_allow_html=True)`
- `@st.cache_data` for performance optimization

**External APIs:**
- `requests.Session` for connection reuse
- Rate limiting with `time.sleep()`
- Error handling for HTTP status codes

**Command-Line Interfaces:**
- `argparse` in `train_model.py` and `etl_share_prices.py`
- `sys.path` manipulation to share code between `ml/`, `etl/`, and `app/`
