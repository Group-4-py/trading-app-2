import json

with open("etl/etl_exploration.ipynb", "r") as f:
    nb = json.load(f)

new_cells = []

explanations = {
    "import os": "**Setup and Environment Configuration:**\\nWe start by importing basic libraries (`os`, `pathlib`) and ensuring our Current Working Directory (CWD) is set to the project root. This ensures all relative paths pointing to the `data/` folder work correctly regardless of where the notebook is executed.",
    "import pandas as pd": "**Data Loading:**\\nHere we load the raw SimFin datasets (`us-shareprices-daily.csv` and `us-companies.csv`) into Pandas DataFrames. Notice the `sep=';'` argument, as SimFin bulk downloads use semicolons instead of commas.",
    "share_prices.head()": "**Initial Data Inspection (Share Prices):**\\nLet's peek at the first 5 rows of the share prices to understand the schema (e.g., Ticker, Date, Open, High, Low, Close, Volume).",
    "companies.head()": "**Initial Data Inspection (Companies):**\\nSimilarly, we check the companies dataset, which contains metadata for each ticker such as the company name and industry.",
    "share_prices['Ticker'].unique()[:20]": "**Exploring Available Tickers:**\\nWe print out the first 20 unique tickers in the dataset to verify the data loaded properly and pick one for our prototyping.",
    "ticker = 'AAPL'": "**Filtering Data for a Single Stock:**\\nTo build and test our feature engineering logic rapidly, we subset the massive dataset to only include historical prices for a single ticker (Apple - AAPL).",
    "sp_ticker['Date'] = pd.to_datetime": "**Basic Feature Engineering:**\\nWe first ensure the data is chronologically sorted. Then, we calculate basic technical indicators: the 1-day percentage return and the 5-day / 10-day simple moving averages.",
    "import numpy as np": "**Creating the Predictive Target:**\\nMachine learning models need a target to predict! Here, we shift the closing prices backwards by 1 to align tomorrow's closing price with today's row. We then create a binary target `target_up`: `1` if tomorrow's return is positive, and `0` otherwise.",
    "sp_ticker['ma_ratio_5_10']": "**Adding Momentum Features:**\\nWe can combine existing features to create new signals. Here, the ratio between the short-term (5-day) and long-term (10-day) moving averages captures price momentum.",
    "feature_cols =": "**Preparing Features (X) and Target (y):**\\nWe select only the columns we want our model to learn from (`X`) and the column we want to predict (`y`). Since rolling averages and shifts create `NaN` (missing) values at the start/end of the dataset, we drop those rows to keep the data clean.",
    "y.head()": "**Target Verification:**\\nA quick check of the first few target values to ensure the binary classification labels (1s and 0s) were constructed correctly.",
    "from sklearn.model_selection import train_test_split": "**Time-Aware Train/Test Split:**\\nIn finance, predicting the past using the future is a cardinal sin! We use `train_test_split` with `shuffle=False` to ensure our training set occurs strictly *before* our testing set chronologically.",
    "from sklearn.linear_model import LogisticRegression": "**Model Training and Prediction:**\\nWe instantiate a standard Logistic Regression model and train it using `fit()` on our training features and targets. Then, we generate predictions (`y_pred`) and probabilities (`y_proba`) on the unseen test set.",
    "pip install matplotlib": "**Installing Visualization Libraries:**\\nTo evaluate our model's performance visually, we need `matplotlib`.",
    "pip install seaborn": "**Installing Seaborn:**\\nSeaborn helps create beautiful, highly readable statistical graphs like confusion matrices.",
    "import matplotlib.pyplot as plt": "**Evaluating Performance (Confusion Matrix):**\\nWe plot a confusion matrix to visualize how often our model was right versus wrong for both classes (Up vs Not Up).",
    "y_test.value_counts(normalize=True)": "**Baseline Accuracy Check:**\\nFinally, we check the baseline distribution of the testing set. If the market naturally went up 58% of the time, a \"dumb\" model that always guesses \"Up\" would achieve 58% accuracy. Our trained model needs to beat this baseline to be useful!"
}

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # Find matching explanation
        matched_text = None
        for key in explanations:
            if key in source:
                matched_text = explanations[key]
                break
                
        if matched_text:
            # Check if the previous cell is already a markdown cell
            # If so, we can just edit or append, but let's just insert yours, or if the user had one, maybe replace it if it's too short.
            # To be safe, we always prepend the new markdown block for clarity, 
            # OR we check if the previous cell was markdown and just modify it to preserve the user's intent.
            
            new_md_cell = {
                "cell_type": "markdown",
                "id": "auto_generated_" + str(hash(key))[:8],
                "metadata": {},
                "source": [matched_text.replace("\\n", "\n")]
            }
            new_cells.append(new_md_cell)
            
    new_cells.append(cell)
    
nb["cells"] = new_cells

with open("etl/etl_exploration.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook successfully updated with markdown explanations!")
