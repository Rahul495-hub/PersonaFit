# Fashion Identity Classifier

A Flask web application that classifies users into a shopping-style
"fashion identity" — **Trendsetter**, **Budget Conscious**, or **Luxury**
— based on a 5-question quiz, using a trained Random Forest model. Results
are enriched with personalized, retrieval-grounded recommendations from a
small RAG (Retrieval-Augmented Generation) layer, and the project includes
a full data-science pipeline (SMOTE-based augmentation, EDA, model
evaluation) plus a business-facing analytics dashboard.

## Features

- Interactive 5-question quiz
- Random Forest classifier trained on quiz-response data (`train_model.py`, `fashion_classifier.py`)
- **RAG-based personalized insights** (`rag.py`): retrieves relevant advice
  snippets from a small knowledge base (`data/knowledge_base.json`) based on
  the user's segment and actual answers, then composes a grounded
  recommendation with visible source snippets — rather than one fixed block
  of text per segment
- **SMOTE-based data augmentation** (`data_augmentation.py`): grows and
  class-balances the training set using SMOTEN (imbalanced-learn's SMOTE
  variant for all-categorical data)
- **Exploratory data analysis** (`eda.py`): class distribution, feature
  breakdowns, a Cramer's V association heatmap, and chi-square
  significance tests, exported as PNG charts + a Markdown report
- **Model evaluation**: `train_model.py` reports accuracy, a full
  per-class precision/recall/F1 breakdown, and a confusion matrix,
  persisted to `models/metrics.json`
- **Analytics dashboard** (`/dashboard`): a business/data-science facing
  view combining training-data composition, model performance, feature
  importance, EDA findings, and live quiz-completion stats from real
  visitors to this deployment
- Visual radar chart of segment probabilities
- Responsive Bootstrap UI

## Robustness

This project is set up to run with zero manual steps beyond `pip install`:

- **Works from any working directory** — all file paths (model, dataset,
  knowledge base) are resolved relative to the project folder itself, not
  wherever you happen to run `python` from.
- **Self-healing model loading** — if `models/*.joblib` is missing, or was
  saved with a different scikit-learn version than what's installed and
  can't be unpickled, `app.py` automatically retrains a fresh model from
  `data/fashion_data.csv` on startup instead of crashing.
- **Defensive session handling** — hitting `/answer` or `/results` with a
  missing/expired session (stale cookie, browser back button, direct URL
  access) redirects back into the quiz instead of raising a 500 error.
- **Clear errors instead of silent/cryptic ones** — a mismatched CSV
  schema, a quiz answer the model has never seen, or an empty knowledge
  base all raise a specific, actionable message rather than an opaque
  stack trace.

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. (Optional but recommended) Generate a larger, SMOTE-balanced training
   set from the raw survey data:
   ```bash
   python data_augmentation.py --target-per-class 100
   ```
   This writes `data/fashion_data_augmented.csv`, leaving the original
   `data/fashion_data.csv` untouched. `train_model.py` automatically
   prefers the augmented file if it exists.
2. (Optional) Run the exploratory data analysis:
   ```bash
   python eda.py
   ```
   Writes charts to `reports/figures/` and a summary to
   `reports/eda_summary.md` (also feeds the analytics dashboard).
3. Train the model:
   ```bash
   python train_model.py
   ```
4. Run the Flask application:
   ```bash
   python app.py
   ```
5. Open your web browser and navigate to http://127.0.0.1:5000/
6. Complete the 5-question quiz to see your fashion identity and
   personalized, source-cited recommendations
7. Visit http://127.0.0.1:5000/dashboard for the analytics dashboard —
   training data composition, model performance, feature importance, EDA
   findings, and live usage stats from anyone who's completed the quiz on
   this deployment

Steps 1–3 aren't required to run the app — if you skip straight to step
4, `app.py` auto-trains a model on the raw dataset on first run.

## Project Structure

- `app.py` — the main Flask application (quiz flow, results, and `/dashboard`)
- `fashion_classifier.py` — trains/loads the Random Forest model, computes evaluation metrics
- `rag.py` — the retrieval-augmented generation layer for personalized recommendations
- `data_augmentation.py` — SMOTE-based dataset augmentation (SMOTEN via imbalanced-learn)
- `eda.py` — exploratory data analysis: charts + Markdown/JSON summary report
- `train_model.py` — trains the model (prefers the augmented dataset if present)
- `data/fashion_data.csv` — raw training data (quiz answers → fashion identity label)
- `data/fashion_data_augmented.csv` — generated by `data_augmentation.py` (not committed by default)
- `data/knowledge_base.json` — the RAG knowledge base of style/shopping advice
- `data/submissions.csv` — auto-generated log of real quiz completions (for the dashboard's live usage stats)
- `models/` — saved model + label encoders + `metrics.json` (`.joblib`/`.json`)
- `reports/` — EDA output: `eda_summary.md`, `eda_summary.json`, `figures/*.png`
- `templates/` — HTML templates (`index.html`, `quiz.html`, `results.html`, `question.html`, `dashboard.html`)

## How It Works

1. The quiz collects 5 categorical answers.
2. `FashionClassifier.predict()` encodes the answers and runs them through
   the trained Random Forest to get a primary segment and per-segment
   probabilities (used for the secondary segment and the radar chart).
3. `FashionRAG.get_recommendation()` builds a query from the segment and the
   raw answers, retrieves the most relevant chunks from
   `data/knowledge_base.json` via TF-IDF + cosine similarity (scoped to that
   segment's own documents plus general wardrobe advice), and composes a
   short recommendation from them. Every recommendation is traceable back to
   the specific reference notes it drew from (shown under "Based on N
   reference notes" on the results page).
4. If an `ANTHROPIC_API_KEY` environment variable is set, step 3 instead asks
   Claude to synthesize the same retrieved notes into more natural prose,
   still strictly grounded in that retrieved context. Without a key, it
   falls back to a template-based composition — the app works fully offline
   either way.

## For data analysts, data scientists & business stakeholders

- **Data scientists**: `data_augmentation.py` and `eda.py` are standalone
  scripts, runnable independent of the Flask app, for iterating on the
  dataset and model. `fashion_classifier.py`'s `train_model()` returns a
  full metrics dict (accuracy, per-class precision/recall/F1, confusion
  matrix) you can script against directly.
- **Data analysts**: `reports/eda_summary.md` and its figures are meant to
  be read standalone — class distribution, which quiz questions actually
  carry signal (chi-square test), and a Cramer's V association matrix
  across all the categorical variables.
- **Business stakeholders**: `/dashboard` is the page to bookmark — it
  surfaces model accuracy, segment distribution, and (once real people
  start taking the quiz) a live breakdown of who's actually using it,
  without needing to read any code.

**Important caveat:** the bundled dataset is synthetic (rule-generated
with light noise), which is why accuracy currently looks close to perfect
— real survey responses will be noisier and messier. Replace
`data/fashion_data.csv` with real data, then re-run
`data_augmentation.py` → `train_model.py` → `eda.py` for numbers that
reflect an actual production model.

## Customization

- Add more advice by appending entries to `data/knowledge_base.json` (each
  needs `id`, `segment`, `topic`, `text`) — no code changes required.
- Add more shopping segments/questions in `app.py`'s `SHOPPING_SEGMENTS` /
  `QUESTIONS` lists, retrain on a matching dataset, and add matching
  `segment` entries to the knowledge base.
- Grow `data/fashion_data.csv` with more/real survey responses and rerun
  `train_model.py` for a stronger classifier — the bundled dataset is still
  synthetic and modest in size.
