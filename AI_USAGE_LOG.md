# AI Usage Log – Part 1
## Automated Daily Trading System

This document reflects how I used AI tools, mainly Perplexity and Antigravity, while developing Part 1 of the project, which covers the ETL pipeline and the machine learning engine.

---

## 1. Designing the Overall ETL and Model Architecture

**What I asked for:**
I used Perplexity to help me think through a clean architecture for Part 1. I asked for a design where the ETL logic, the training script, and the Streamlit app could all share one source of truth for data transformations and feature definitions.

**What worked:**
The assistant helped me converge on a structure with `etl/` scripts, a central `app/utils/etl.py` for feature engineering, `app/utils/config.py` for ticker and feature lists, and `ml/train_model.py` for the multi-ticker model. This early design discussion saved time by avoiding one-off scripts that would later diverge. It also pushed me to think in terms of reusable functions and clear inputs and outputs for each module.

**What was harder:**
Some of the first suggestions were more complex than we needed for this assignment — for example, using a full orchestration framework. I had to push back and ask for a simpler version that we could realistically implement within the course constraints. That back and forth was useful because it forced me to justify each architectural choice instead of just copying what the AI proposed.

---

## 2. Building the Exploratory Notebook for One Ticker

**What I asked for:**
At the start I asked Perplexity to help me turn a raw SimFin extract for one ticker into something I could model: sorting by date, computing daily returns, rolling averages, and a binary up-or-not-up target based on the next-day close.

**What worked:**
The assistant helped me structure the notebook into clear sections: loading data, basic cleaning, feature creation, target creation, and a first logistic regression on one ticker. It also helped me choose reasonable first features, such as `return_1d`, `ma_5`, `ma_10`, and a simple ratio, and set up a time-aware train/test split without shuffling. This gave me a working baseline that I understood well and later generalised into the shared ETL code.

**What was harder:**
Because the notebook went through many iterations, it was easy to let the AI keep adding code without stopping to review. I had to be disciplined about reading each generated cell, printing intermediate results, and deleting or refactoring chunks that were redundant. The main lesson here was that AI is great at producing code quickly, but the responsibility to curate and simplify still sits with me.

---

## 3. Generalising the ETL into Reusable Functions

**What I asked for:**
Once the single-ticker notebook worked, I asked Perplexity to help refactor the logic into a proper `run_etl` function that could be imported by both the ETL scripts and the model training code. I also asked for helpers to stream the large CSV in chunks and to save one parquet file per ticker.

**What worked:**
The refactoring suggestions were helpful in turning ad-hoc notebook cells into clean functions with clear signatures, such as `run_etl(prices_ticker, include_target=True)` and `run_etl_for_ticker_from_df`. The AI helped me think about memory issues with the bulk CSV and suggested chunked loading, filtering early by ticker, and writing out per-ticker parquet files. This made the pipeline more robust and closer to something used in a real data project.

**What was harder:**
I had to correct several details where the AI assumed slightly different column names or file paths from the ones we actually had. This reminded me that generated refactors are not drop-in replacements and that unit tests — or at least small sanity checks — are important after each structural change. I also learned that it helps to paste actual function signatures and example DataFrame heads into the prompt to keep the assistant grounded.

---

## 4. Training the Multi-Ticker Model

**What I asked for:**
For the modelling step I used Perplexity to design a training script that would loop over tickers, call the shared ETL, combine everything into a single dataset, one-hot encode the ticker, and train a logistic regression with proper scaling. I also asked for guidance on how to respect time order when splitting into train and test.

**What worked:**
The assistant produced a clear training flow that I could adapt: building a combined feature table, adding ticker dummy variables, selecting features based on `MODEL_FEATURES` from the config file, and using a `StandardScaler + LogisticRegression` pipeline. It also emphasised using `shuffle=False` in `train_test_split` to avoid leakage from future data. This made the final script both readable and consistent with the Streamlit part of the project.

**What was harder:**
Some of the first versions tried to be too clever with automatic feature selection or additional models that were beyond the scope of the assignment. I had to scale things back to a simpler logistic model and focus on correctness and alignment with Part 2 rather than chasing higher accuracy. This was a good reminder that AI can suggest many advanced options, but in a course context the priority is clarity and maintainability.

---

## 5. Debugging and Code Review Using Antigravity

**What I asked for:**
I used Antigravity mainly as a second opinion to debug errors, tidy up code style, and check for edge cases in the ETL and model training scripts. Typical prompts were along the lines of "this function fails on missing values, what is the cleanest fix?" or "can you spot any obvious issues with this pipeline before I run it on the full dataset?"

**What worked:**
Having two different AI tools gave me useful redundancy. When Perplexity suggested one approach and Antigravity suggested a slightly different refactor, comparing the two helped me understand the trade-offs and choose the simpler or safer version. Antigravity was particularly helpful at spotting small issues like off-by-one problems in shifting the target, inconsistent column naming, and missing imports.

**What was harder:**
Switching between tools can make it easier to accumulate too many partial versions of the same function. I had to keep my own mental source of truth and resist the temptation to accept every suggestion. In practice this meant running the full pipeline regularly, checking that the data shapes and feature names matched what Part 2 expected, and reverting changes that made the code harder to follow.

---

## Reflection on AI-Assisted Development

Overall, using Perplexity and Antigravity for Part 1 made it possible to reach a more mature architecture than I would probably have designed alone within the same time frame. The tools were especially valuable for scaffolding code, suggesting patterns for reusable ETL, and helping with debugging and documentation. At the same time, every successful step required active oversight: checking that the generated code matched the real data, simplifying overly complex suggestions, and keeping the whole pipeline consistent with Part 2 of the project.

The main lesson for me is that these tools are powerful collaborators rather than autopilots. When I provided concrete context, pasted real code, and asked very specific questions, the quality of the help was high. When I was vague or asked for huge changes in one go, the results were less reliable and required more cleanup.
