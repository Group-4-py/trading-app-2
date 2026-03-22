# AI Usage Log – Automated Daily Trading System

This document reflects how AI (mainly Claude Code, Perplexity and CODEX) was used throughout the development of this project, what worked well, what was harder to get right, and what lessons were learned about effective AI-assisted development. 

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

## 6. Reflection on AI-Assisted Development

Overall, using Perplexity and Antigravity for Part 1 made it possible to reach a more mature architecture than I would probably have designed alone within the same time frame. The tools were especially valuable for scaffolding code, suggesting patterns for reusable ETL, and helping with debugging and documentation. At the same time, every successful step required active oversight: checking that the generated code matched the real data, simplifying overly complex suggestions, and keeping the whole pipeline consistent with Part 2 of the project.

The main lesson for me is that these tools are powerful collaborators rather than autopilots. When I provided concrete context, pasted real code, and asked very specific questions, the quality of the help was high. When I was vague or asked for huge changes in one go, the results were less reliable and required more cleanup.

## 7. Generating the App Skeleton

**What I asked for:** A fully functioning multi-page Streamlit app skeleton from scratch: Home page, Go Live page, Model Insights page, and Backtesting page, with navigation, sidebar controls, charts, metric cards, and a placeholder ML model.

**What worked:** This was one of the most effective and time-saving uses of AI in the project. The result (commit `c243abe`) included a complete working app with consistent structure across all pages, a custom CSS theme, chart wrappers, a `config.py` for centralised settings, and a dummy model that could be swapped out later. It would have taken many hours to write this manually.

**What was harder:** Because the skeleton was generated all at once, there were some inconsistencies in style and naming that required cleanup later. The AI made reasonable default design choices, but they weren't always aligned with the team's actual vision for the look and feel.

---

## 8. Changing the Overall Theme

**What I asked for:** Initially I tried to overhaul the entire visual theme of the app in a single prompt: changing colours, fonts, card styles, and overall aesthetic at once.

**What was harder:** This approach was frustrating. When you ask AI to change "the whole design," it tends to either change too much (breaking things that were working) or make conservative changes that don't fully match the desired result. It was difficult to communicate a visual "feel" in text, and iterating on vague feedback like "make it look more futuristic" didn't produce consistent results. The AI and I went back and forth several times without converging on the right outcome.

---

## 9. Precise Formatting Modifications

**What I asked for:** After struggling with broad changes, I switched to making very specific, targeted requests, one element at a time. For example:
- "Make the signal box the same height as the gauge boxes"
- "Top-align the three boxes in the Go Live signal row"
- "For the price_change_pct in the "go live" tab, make it so if the change is positive, a green rectangle pointing up appears next to it and if the price change is negative, a red rectangle pointing down appears next to it".

**What worked:** This approach worked much better. AI is very effective at making precise, well-scoped edits when you tell it exactly what to change and where. The results were predictable and easy to verify. Giving the AI specific CSS property names, values, and element descriptions removed ambiguity and led to correct changes on the first or second attempt.

---

## 10. Adding Company Logo Support

**What I asked for:** Replace the emoji ticker icons with real PNG company logos: both in the "Companies We Track" section on the Home page and in the Go Live page header. Then extend this to the sidebar across all three pages, with a separate set of header-specific logos independent from the card logos.

**What worked:** AI handled the technical implementation well, base64 encoding images and embedding them in HTML `<img>` tags is a repetitive pattern that AI executes reliably. It correctly identified that pages in the `app/pages/` subdirectory needed `../` in the path to reach assets, and it set up the config in `config.py` so that logo paths are centralised and easy to change.

**What was harder:** Getting the right logo to show up in the right place required a few iterations. The first version used the same logo for both the card and the header, but I wanted them to be separate (so the header/sidebar logos could be a different resolution or style). This required adding a second key (`header_image`) to the config, which the AI did cleanly once I clarified the requirement.

---

## 11. Removing Emojis from Page Navigation

**What I asked for:** Remove the emoji prefixes from the page tabs in the Streamlit sidebar (e.g., "📈 Go Live" → "Go Live").

**What worked:** The AI correctly identified that Streamlit generates sidebar navigation labels directly from filenames, so the only way to remove the emojis was to rename the files using `git mv` to preserve git history. This was a non-obvious technical detail that I wouldn't have known immediately.

---

## 12. Fine-Tuning the Signal & Metric Layout

**What I asked for:** Several precise layout adjustments on the Go Live page:
- Top-align the signal box with the gauge boxes beside it
- Make the signal box the same height as the gauges
- Move the "Today's Signal" heading inside the card (it was pushing the card down)
- Restyle the "Last Close" metric: remove the delta below the value and replace it with a top-right corner badge showing a green ▲ or red ▼

**What worked:** Once I identified the root cause of each alignment issue (e.g., the heading above the signal box being the cause of the misalignment), the AI implemented clean fixes. The Last Close badge using `position: absolute` was a good solution that kept the box height fixed.

**What was harder:** Diagnosing layout bugs through text alone is difficult. Several of these required 2–3 iterations before the fix was correct. For example, the first attempt to top-align the columns used `vertical_alignment="top"` on `st.columns()`, but the real problem was the `st.markdown` heading above the card — the AI only found this after I pushed back and described more carefully what I was seeing.

The font size on the Last Close price also needed iteration: using `1.5rem` in a raw HTML block renders larger than Streamlit's `1.5rem` in its own CSS (different scaling contexts), so the value had to be tuned manually to `1.1rem` to match visually.

---

## 13. Git & Version Control Issues

**What happened:** A teammate force-pushed to `main`, overwriting the commit history that contained all our previous changes. This caused a rebase conflict that wiped many project files. Additionally, a `git clean -fd` command (run to reset the working directory) deleted untracked PNG logo files that had never been committed.

**What worked:** The AI correctly diagnosed the problem, identified that the files still existed on `origin/Part-2`, and restored them using `git checkout origin/Part-2 -- <paths>`. The logo files were also recovered from git history once it was confirmed they had been committed to that branch.

**What was harder:** Git conflict resolution in complex multi-contributor scenarios is error-prone even with AI assistance. The AI made a mistake by running `git clean -fd` which deleted untracked files (the logo PNGs). Recovering from cascading git errors required careful diagnosis at each step. This is an area where AI can move quickly but also cause hard-to-reverse damage if not careful.

---

## 14. Full Project Review Against Assignment Requirements

**What I asked for:** A comprehensive review of the entire repository against the assignment rubric to identify missing deliverables, code gaps, and areas for improvement before submission.

**What worked:** The AI compared every requirement (ETL, ML model, API wrapper, web app, deployment, deliverables) against our codebase and produced a prioritised action plan. It identified the gaps I hadn't noticed.

---

## 15. ML Model Improvement

**What I asked for:** Investigated whether the model's accuracy could be improved, and whether the Model Insights page was displaying accurate metrics.

**What worked:** The AI identified that the Model Insights page was computing metrics on live/synthetic data rather than the actual training test set, meaning the displayed values could be misleading. It also explained why Logistic Regression plateaus around 51% for daily stock prediction and suggested GradientBoosting as a practical upgrade.

---

## 16. AI Usage Log Update

**What I asked for:** Update this log with brief summaries of all recent AI-assisted conversations.

**What worked:** Straightforward documentation task, AI helped me summarize our chats.
---
