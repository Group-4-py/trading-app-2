# AI Usage Log – Automated Daily Trading System

This document reflects how AI (maintl Claude Code) was used throughout the development of this project, what worked well, what was harder to get right, and what lessons were learned about effective AI-assisted development.

---

## 1. Generating the App Skeleton

**What I asked for:** A fully functioning multi-page Streamlit app skeleton from scratch: Home page, Go Live page, Model Insights page, and Backtesting page, with navigation, sidebar controls, charts, metric cards, and a placeholder ML model.

**What worked:** This was one of the most effective and time-saving uses of AI in the project. The result (commit `c243abe`) included a complete working app with consistent structure across all pages, a custom CSS theme, chart wrappers, a `config.py` for centralised settings, and a dummy model that could be swapped out later. It would have taken many hours to write this manually.

**What was harder:** Because the skeleton was generated all at once, there were some inconsistencies in style and naming that required cleanup later. The AI made reasonable default design choices, but they weren't always aligned with the team's actual vision for the look and feel.

---

## 2. Changing the Overall Theme

**What I asked for:** Initially I tried to overhaul the entire visual theme of the app in a single prompt: changing colours, fonts, card styles, and overall aesthetic at once.

**What was harder:** This approach was frustrating. When you ask AI to change "the whole design," it tends to either change too much (breaking things that were working) or make conservative changes that don't fully match the desired result. It was difficult to communicate a visual "feel" in text, and iterating on vague feedback like "make it look more futuristic" didn't produce consistent results. The AI and I went back and forth several times without converging on the right outcome.

**Lesson learned:** Broad aesthetic changes are hard to communicate to AI. It works better as a precise code editor than as a designer.

---

## 3. Precise Formatting Modifications

**What I asked for:** After struggling with broad changes, I switched to making very specific, targeted requests, one element at a time. For example:
- "Make the signal box the same height as the gauge boxes"
- "Top-align the three boxes in the Go Live signal row"
- "For the price_change_pct in the "go live" tab, make it so if the change is positive, a green rectangle pointing up appears next to it and if the price change is negative, a red rectangle pointing down appears next to it".

**What worked:** This approach worked much better. AI is very effective at making precise, well-scoped edits when you tell it exactly what to change and where. The results were predictable and easy to verify. Giving the AI specific CSS property names, values, and element descriptions removed ambiguity and led to correct changes on the first or second attempt.

**Lesson learned:** The more specific and technical the prompt, the better the AI performs.

---

## 4. Adding Company Logo Support

**What I asked for:** Replace the emoji ticker icons with real PNG company logos: both in the "Companies We Track" section on the Home page and in the Go Live page header. Then extend this to the sidebar across all three pages, with a separate set of header-specific logos independent from the card logos.

**What worked:** AI handled the technical implementation well, base64 encoding images and embedding them in HTML `<img>` tags is a repetitive pattern that AI executes reliably. It correctly identified that pages in the `app/pages/` subdirectory needed `../` in the path to reach assets, and it set up the config in `config.py` so that logo paths are centralised and easy to change.

**What was harder:** Getting the right logo to show up in the right place required a few iterations. The first version used the same logo for both the card and the header, but I wanted them to be separate (so the header/sidebar logos could be a different resolution or style). This required adding a second key (`header_image`) to the config, which the AI did cleanly once I clarified the requirement.

---

## 5. Removing Emojis from Page Navigation

**What I asked for:** Remove the emoji prefixes from the page tabs in the Streamlit sidebar (e.g., "📈 Go Live" → "Go Live").

**What worked:** The AI correctly identified that Streamlit generates sidebar navigation labels directly from filenames, so the only way to remove the emojis was to rename the files using `git mv` to preserve git history. This was a non-obvious technical detail that I wouldn't have known immediately.

**What was harder:** Nothing particularly difficult here, once the root cause (filename-based nav) was understood, the fix was straightforward.

---

## 6. Fine-Tuning the Signal & Metric Layout

**What I asked for:** Several precise layout adjustments on the Go Live page:
- Top-align the signal box with the gauge boxes beside it
- Make the signal box the same height as the gauges
- Move the "Today's Signal" heading inside the card (it was pushing the card down)
- Restyle the "Last Close" metric: remove the delta below the value and replace it with a top-right corner badge showing a green ▲ or red ▼

**What worked:** Once I identified the root cause of each alignment issue (e.g., the heading above the signal box being the cause of the misalignment), the AI implemented clean fixes. The Last Close badge using `position: absolute` was a good solution that kept the box height fixed.

**What was harder:** Diagnosing layout bugs through text alone is difficult. Several of these required 2–3 iterations before the fix was correct. For example, the first attempt to top-align the columns used `vertical_alignment="top"` on `st.columns()`, but the real problem was the `st.markdown` heading above the card — the AI only found this after I pushed back and described more carefully what I was seeing.

The font size on the Last Close price also needed iteration: using `1.5rem` in a raw HTML block renders larger than Streamlit's `1.5rem` in its own CSS (different scaling contexts), so the value had to be tuned manually to `1.1rem` to match visually.

---

## 7. Git & Version Control Issues

**What happened:** A teammate force-pushed to `main`, overwriting the commit history that contained all our previous changes. This caused a rebase conflict that wiped many project files. Additionally, a `git clean -fd` command (run to reset the working directory) deleted untracked PNG logo files that had never been committed.

**What worked:** The AI correctly diagnosed the problem, identified that the files still existed on `origin/Part-2`, and restored them using `git checkout origin/Part-2 -- <paths>`. The logo files were also recovered from git history once it was confirmed they had been committed to that branch.

**What was harder:** Git conflict resolution in complex multi-contributor scenarios is error-prone even with AI assistance. The AI made a mistake by running `git clean -fd` which deleted untracked files (the logo PNGs). Recovering from cascading git errors required careful diagnosis at each step. This is an area where AI can move quickly but also cause hard-to-reverse damage if not careful.

---
