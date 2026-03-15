# Part 1 – ETL & Machine Learning Engine

This part of the project contains the data pipeline and machine learning engine that power our trading web app. It reads bulk US share-price data from SimFin, applies a shared ETL pipeline (the same one used by the Streamlit app), and trains a multi-ticker classification model that predicts whether tomorrow's close price will go **up** or **down** for each stock.

---

## Team

- Bojana Belincevic – ML Engineer (ETL & Model Development)
- David Carrillo – Backend Developer (API Wrapper & Data Pipeline)
- Sebastião Clemente – Frontend Developer (App & Deployment)
- Bassem El Halawani – Data Analyst (Feature Engineering & Strategy)
- Theo Henry – Product Lead (Product Direction & Prioritization)
- Ocke Moulijn – Insights Analyst (Model Validation & Testing)

---

## Project structure (Part 1)

Top-level folders relevant for ETL + ML:

- `data/`
  - `raw/` – Raw SimFin bulk CSV (e.g. `us-shareprices-daily.csv`, not tracked in git).
  - `processed/` – Engineered feature tables in parquet format, one file per ticker (e.g. `AAPL.parquet`).
- `etl/`
  - `etl_share_prices.py` – CLI entrypoint to run ETL for a **single** ticker (`--ticker`).
  - `etl_utils.py` – Shared helpers used by CLI and model; loads raw CSV in chunks, filters tickers, and delegates to the app's ETL.
  - `run_all_tickers.py` – Orchestrator that runs ETL for all tickers in `TICKER_LIST` and writes one parquet per ticker.
- `ml/`
  - `model/` – Saved model artifacts (e.g. `all_tickers_model.joblib`).
  - `train_model.py` – Reads raw CSV, runs the shared ETL for all tickers, prepares the feature matrix, trains a logistic-regression classifier, and saves the model.
- `notebooks/`
  - `etl_exploration.ipynb` – Initial exploration of the raw SimFin data and prototype feature engineering.
- `app/`
  - `utils/config.py` – Central config for tickers, feature list `MODEL_FEATURES`, and ticker dummy names used by both the app and the model.
  - `utils/etl.py` – Canonical ETL/feature-engineering logic used by the app and re-used by Part 1 via imports.

This structure ensures there is **one source of truth** for tickers and feature definitions shared between ETL, the ML model, and the Streamlit app.

---

## ETL pipeline

The ETL layer in `etl/etl_utils.py` and `app/utils/etl.py` does the following:

1. **Load raw prices (streaming)**  
   `load_share_prices` reads `us-shareprices-daily.csv` in chunks (semicolon-separated), selecting only:

   ```text
   ["Ticker", "Date", "Open", "High", "Low", "Close", "Adj. Close", "Volume"]
   ```

   It can optionally filter rows to a subset of tickers (`TICKER_LIST`) to avoid loading the entire file into memory.

2. **Per-ticker filtering and sorting**  
   `filter_ticker` keeps one ticker, converts `Date` to a proper datetime, and sorts chronologically.

3. **Shared feature engineering (via app/utils/etl.py)**  
   `run_etl(prices_ticker, include_target=True)` produces the full feature table used by both the app and the model, including:

   - Price & OHLCV: `open`, `high`, `low`, `close`, `adj_close`, `volume`, `price`.
   - Returns: `return_1d`, `return_5d`, `return_10d`.
   - Volatility: `volatility_10d`, `volatility_20d`.
   - Moving averages: `sma_5`, `sma_10`, `sma_20`, `sma_50`.
   - Trend & momentum: `ema_12`, `ema_26`, `macd`, `macd_signal`, `rsi_14`.
   - Bollinger bands: `bb_upper`, `bb_lower`, `bb_width`.
   - Volume features: `volume_sma_10`, `volume_ratio`.
   - Risk: `atr_14` (Average True Range).
   - Target: `target` = 1 if next-day close > today's close, else 0.

4. **Saving processed data**  
   `run_etl_for_ticker_from_df` writes a parquet file per ticker to:

   ```text
   data/processed/{TICKER}.parquet
   ```

   and returns the features DataFrame.

The **same** ETL function (`run_etl`) is called by:

- the CLI ETL (`etl_share_prices.py` / `run_all_tickers.py`), and
- the model training script (`ml/train_model.py`),

so the app and model always stay aligned.

---

## Model training

The model in `ml/train_model.py` is a logistic-regression classifier that uses the engineered features to predict the binary `target` (up vs. down).

### 1. Load raw CSV

```python
raw_df = load_raw_csv(csv_path)
```

- Streams `us-shareprices-daily.csv` in 250k-row chunks.
- Filters rows to tickers in `TICKER_LIST`.
- Uses only the `PRICE_COLUMNS` listed above.

### 2. Run ETL per ticker and combine

```python
combined = build_dataset(raw_df)
```

For each ticker:

- Filter rows for that ticker.
- Run `run_etl(df, include_target=True)` from `app/utils/etl.py`.
- Add a `ticker` column.

Concatenate all tickers into one combined dataset.

### 3. Prepare features

```python
X, y, ticker_dummies = prepare_features(combined)
```

- One-hot encode `ticker` with `drop_first=True` (first ticker is the baseline).
- Select `MODEL_FEATURES` (from `app/utils/config.py`) plus all `ticker_*` dummy columns.
- Drop rows with missing data.
- Target `y` is the `target` column.

### 4. Train/test split and model

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # preserve temporal ordering
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
])
pipeline.fit(X_train, y_train)
```

- 80/20 **time-ordered** split (`shuffle=False`) to respect the temporal structure.
- The pipeline standardises features and fits a logistic-regression classifier.

### 5. Evaluation

```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy  : {acc:.3f}")
print(classification_report(y_test, y_pred, target_names=["DOWN (0)", "UP (1)"]))
```

The script prints accuracy, a full classification report, and the number of train/test samples.

### 6. Saving the model

```python
import joblib
from pathlib import Path

MODEL_OUTPUT = PROJECT_ROOT / "ml" / "model" / "all_tickers_model.joblib"
MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_OUTPUT)
```

After training, the script also prints the list of `ticker_*` dummy columns so the `TICKER_DUMMIES` setting in `app/utils/config.py` can be kept in sync.

---

## Installation & setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-org-or-user>/python-2-group-project.git
cd python-2-group-project
```

### 2. Create and activate the environment

Using conda:

```bash
conda create -n trading-app python=3.11
conda activate trading-app
pip install -r requirements.txt
```

(You can also use `python -m venv` instead of conda if you prefer.)

### 3. Download the raw SimFin data

1. Log in at https://www.simfin.com  
2. Go to **Bulk Download → Share Prices → US**  
3. Download `us-shareprices-daily.csv`  
4. Place the file at:

```text
data/raw/us-shareprices-daily.csv
```

---

## Running the ETL

### Option A – All tickers

From the project root:

```bash
conda activate trading-app  # if not already active

python etl/run_all_tickers.py
```

This will:

- Read `data/raw/us-shareprices-daily.csv` in chunks.
- Filter to tickers defined in `TICKER_LIST` (from `app/utils/config.py`).
- Run the shared ETL for each ticker.
- Save one parquet per ticker in `data/processed/` (e.g. `AAPL.parquet`).

### Option B – Single ticker

To generate features for a single ticker:

```bash
python etl/etl_share_prices.py --ticker AAPL
```

Optional arguments:

```bash
python etl/etl_share_prices.py --ticker MSFT --input data/raw/us-shareprices-daily.csv --output-dir data/processed
```

---

## Training the model

Once ETL has produced the processed parquet files (or if you prefer to re-run ETL from within the script), you can train the model:

```bash
python ml/train_model.py
```

Optional: specify a custom CSV path:

```bash
python ml/train_model.py --data data/raw/us-shareprices-daily.csv
```

The script will:

1. Load and filter the raw CSV.
2. Run the shared ETL for all tickers.
3. Build the feature matrix (`MODEL_FEATURES + ticker dummies`).
4. Perform an 80/20 time-based train/test split.
5. Fit a `StandardScaler + LogisticRegression` pipeline.
6. Print metrics and save `ml/model/all_tickers_model.joblib`.

After training, commit the model file so the app can load it:

```bash
git add ml/model/all_tickers_model.joblib
git commit -m "Update all_tickers model"
git push
```

---

## Using the model in the Streamlit app

Part 2 (the app) runs Streamlit, typically at:

```text
http://localhost:8501
```

The app can load the trained model like this:

```python
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "ml" / "model" / "all_tickers_model.joblib"

model = joblib.load(MODEL_PATH)
```

The app then:

- Reads the processed parquet for a selected ticker from `data/processed/`.
- Builds the same feature matrix (`MODEL_FEATURES + ticker dummies`).
- Calls `model.predict` or `model.predict_proba` to show the probability that tomorrow's close will be up or down for that ticker.

This tight integration between Part 1 (ETL + ML) and Part 2 (app) ensures that the model, features, and UI always stay in sync.