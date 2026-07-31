from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import pandas as pd
import os
import csv
import json
import logging
from datetime import datetime, timezone
from fashion_classifier import FashionClassifier
from rag import FashionRAG

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Resolve every project path relative to this file's own location, not the
# current working directory. Without this, "python app.py" only works when
# launched from inside the project folder; running it from anywhere else
# (a different cwd, an IDE run button, a process manager) would raise
# FileNotFoundError looking for data/models in the wrong place.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, 'data', 'fashion_data.csv')
AUGMENTED_CSV = os.path.join(BASE_DIR, 'data', 'fashion_data_augmented.csv')
KB_PATH = os.path.join(BASE_DIR, 'data', 'knowledge_base.json')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
METRICS_PATH = os.path.join(MODELS_DIR, 'metrics.json')
EDA_SUMMARY_PATH = os.path.join(BASE_DIR, 'reports', 'eda_summary.json')
SUBMISSIONS_CSV = os.path.join(BASE_DIR, 'data', 'submissions.csv')

# Prefer the SMOTE-augmented, class-balanced dataset (see
# data_augmentation.py) if one has been generated; fall back to the raw
# survey-style dataset otherwise.
TRAINING_CSV = AUGMENTED_CSV if os.path.exists(AUGMENTED_CSV) else DATA_CSV

app = Flask(__name__)
app.jinja_env.globals.update(zip_lists=zip)
# A fixed key (overridable via env var) keeps sessions valid across
# restarts. os.urandom(24) on every startup invalidates every in-progress
# quiz session the moment the server restarts/reloads, which looks like a
# random, hard-to-reproduce bug ("my answers disappeared") rather than an
# obvious one. Set FLASK_SECRET_KEY in production to override this default.
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-fashion-identity-classifier-key')

# Initialize the classifier and load the trained model. If no model has
# been trained yet, or the saved model can't be loaded (e.g. it was
# pickled with a different scikit-learn version than what's installed
# here), train a fresh one automatically from the bundled dataset instead
# of crashing on startup or requiring a manual `python train_model.py` step.
classifier = FashionClassifier(models_dir=MODELS_DIR)
if not classifier.load_model():
    logger.warning("No usable trained model found — training a fresh one from %s", TRAINING_CSV)
    try:
        classifier.train_model(TRAINING_CSV)
        logger.info("Model trained and saved successfully.")
    except Exception:
        logger.exception(
            "Automatic training failed. The app will start, but /results will "
            "show an error until a model is available."
        )

# Initialize the RAG engine for personalized, retrieval-grounded recommendations
rag_engine = FashionRAG(kb_path=KB_PATH)

# Shopping Segments
# NOTE: "label" must match the exact class string the model was trained on
# (see data/fashion_data.csv / the "Fashion Identity" column), since that's
# what classifier.predict() returns. "name" is only for display and can be
# friendlier/longer without breaking the lookup.
SHOPPING_SEGMENTS = [
    {
        "label": "Trendsetter",
        "name": "Trendsetter",
        "description": "You're always at the forefront of fashion, quickly adopting new styles and influencing others. You follow fashion blogs, social media influencers, and stay updated on runway trends.",
        "recommendations": "Focus on bold statement pieces that showcase your fashion-forward perspective. Mix high-street brands with unique finds to create signature looks.",
        "shopping_habits": "Regular shopping at cutting-edge boutiques, online trend-focused retailers, and designer sample sales",
        "image": "trendsetter.jpg"
    },
    {
        "label": "Budget Conscious",
        "name": "Budget Conscious Shopper",
        "description": "You have a keen eye for value and prioritize cost-effectiveness without sacrificing style. You're strategic about your purchases and know how to create stylish outfits affordably.",
        "recommendations": "Invest in versatile basics that can be mixed and matched, and add affordable trend pieces seasonally. Look for quality at discount stores and during sales events.",
        "shopping_habits": "Sales shopping, thrift stores, outlet malls, and budget-friendly retailers",
        "image": "budget.jpg"
    },
    {
        "label": "Luxury",
        "name": "Luxury Enthusiast",
        "description": "You appreciate fine craftsmanship and are willing to invest in high-quality pieces. You value exclusivity, premium materials, and the heritage of luxury brands.",
        "recommendations": "Focus on timeless investment pieces from established luxury houses. Choose quality over quantity and build a curated collection of signature items.",
        "shopping_habits": "Designer boutiques, high-end department stores, and exclusive shopping experiences",
        "image": "luxury.jpg"
    },
    {
        "label": "Sustainable Conscious",
        "name": "Sustainable Conscious",
        "description": "You prioritize ethical and environmental considerations in your shopping choices. You research brands' practices and prefer companies with transparent supply chains.",
        "recommendations": "Look for certified sustainable brands, secondhand luxury, and timeless designs with long lifespans. Invest in quality pieces made with eco-friendly materials.",
        "shopping_habits": "Ethical brands, secondhand marketplaces, vintage stores, and local artisan shops",
        "image": "sustainable.jpg"
    },
    {
        "label": "Convenience Seeker",
        "name": "Convenience Seeker",
        "description": "You value efficiency and ease in your shopping experience. Time is precious to you, and you prefer straightforward shopping with minimal decision-making.",
        "recommendations": "Use subscription services and personal shoppers to streamline your wardrobe updates. Build a capsule wardrobe of reliable basics that work well together.",
        "shopping_habits": "Online shopping, subscription services, and one-stop retailers with wide selections",
        "image": "convenience.jpg"
    }
]

# Quick lookup from the model's raw class label to its full segment info
SEGMENTS_BY_LABEL = {segment["label"]: segment for segment in SHOPPING_SEGMENTS}

# Questions for the quiz
QUESTIONS = [
    {
        'id': 0,
        'text': 'How often do you shop for new clothes?',
        'options': ['Weekly', 'Monthly', 'Quarterly']
    },
    {
        'id': 1,
        'text': 'Where do you typically shop for clothes?',
        'options': ['Fast Fashion', 'Department Stores', 'Luxury Boutiques']
    },
    {
        'id': 2,
        'text': 'What influences your clothing purchases the most?',
        'options': ['Trends', 'Price', 'Quality']
    },
    {
        'id': 3,
        'text': 'How would you describe your go-to daily outfit?',
        'options': ['Trendy', 'Business Casual', 'Formal']
    },
    {
        'id': 4,
        'text': 'If you had to choose, would you prefer timeless pieces or trendy items?',
        'options': ['Trendy', 'Mix', 'Timeless']
    }
]

@app.route('/')
def index():
    # Reset session on home page visit
    session.clear()
    return render_template('index.html')

@app.route('/quiz')
def quiz():
    if 'current_question' not in session or 'answers' not in session:
        session['current_question'] = 0
        session['answers'] = {}

    # Defensive bounds check: if a session somehow points past the last
    # question (e.g. an old/stale cookie from a previous version of the
    # quiz with a different question count), reset rather than raising an
    # IndexError.
    if session['current_question'] >= len(QUESTIONS):
        session['current_question'] = 0
        session['answers'] = {}

    return render_template('quiz.html', 
                         question=QUESTIONS[session['current_question']]['text'],
                         options=QUESTIONS[session['current_question']]['options'],
                         current_question=session['current_question'],
                         total_questions=len(QUESTIONS),
                         progress=(session['current_question'] / len(QUESTIONS)) * 100)

@app.route('/answer', methods=['POST'])
def answer():
    # Guard against a missing/expired session (e.g. cookies cleared,
    # browser back button, or hitting this endpoint directly) instead of
    # raising a KeyError and returning a 500 error.
    if 'current_question' not in session or 'answers' not in session:
        flash('Your session expired. Please start the quiz again.')
        return redirect(url_for('quiz'))

    answer = request.form.get('answer')
    if answer is None:
        flash('Please select an answer')
        return redirect(url_for('quiz'))

    current_question = session['current_question']
    if current_question >= len(QUESTIONS):
        return redirect(url_for('results'))

    # Store answer with question ID
    question_id = QUESTIONS[current_question]['id']
    question_text = QUESTIONS[current_question]['text']
    session['answers'][f'{question_id + 1}.{question_text}'] = answer
    session['current_question'] = current_question + 1
    session.modified = True

    if session['current_question'] >= len(QUESTIONS):
        return redirect(url_for('results'))
    
    return redirect(url_for('quiz'))

@app.route('/results')
def results():
    if 'answers' not in session or len(session['answers']) != len(QUESTIONS):
        return redirect(url_for('quiz'))
    
    try:
        # Make prediction using the trained model
        prediction = classifier.predict(session['answers'])

        # Get the predicted segment and probabilities
        primary_segment = prediction['prediction']
        probabilities = prediction['probabilities']

        # IMPORTANT: probabilities are ordered according to the model's
        # classes_ attribute (alphabetical for string labels), NOT the order
        # segments happen to be listed in this file. The previous version
        # hardcoded ['Trendsetter', 'Budget Conscious', 'Luxury'], which
        # silently mismatched the real order and mislabeled the secondary
        # segment and radar chart for most users. Pull the true order from
        # the model itself instead.
        segments = list(classifier.model.classes_)
        sorted_probs = sorted(zip(segments, probabilities), key=lambda x: x[1], reverse=True)
        secondary_segment = sorted_probs[1][0]

        # Prepare data for the radar chart
        radar_data = {
            'labels': segments,
            'scores': probabilities
        }

        # Look up the rich, pre-written segment descriptions
        primary_info = SEGMENTS_BY_LABEL.get(primary_segment)
        secondary_info = SEGMENTS_BY_LABEL.get(secondary_segment)

        # RAG step: retrieve knowledge-base snippets relevant to this
        # specific user's segment AND their actual quiz answers, then
        # generate a grounded, personalized recommendation from them.
        recommendation = rag_engine.get_recommendation(
            segment=primary_segment,
            quiz_answers=session['answers'],
        )

        _log_submission(primary_segment, secondary_segment)

        return render_template('results.html',
                             primary_segment=primary_segment,
                             secondary_segment=secondary_segment,
                             primary_info=primary_info,
                             secondary_info=secondary_info,
                             recommendation=recommendation,
                             radar_data=radar_data)

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        flash('An error occurred while processing your results. Please try again.')
        return redirect(url_for('quiz'))

def _log_submission(primary_segment, secondary_segment):
    """Append a completed quiz result to data/submissions.csv for the
    business dashboard's live usage stats. Best-effort only — a logging
    failure (e.g. read-only filesystem) should never break the results
    page for the person taking the quiz."""
    try:
        is_new = not os.path.exists(SUBMISSIONS_CSV)
        os.makedirs(os.path.dirname(SUBMISSIONS_CSV), exist_ok=True)
        with open(SUBMISSIONS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(['timestamp', 'primary_segment', 'secondary_segment'])
            writer.writerow([datetime.now(timezone.utc).isoformat(), primary_segment, secondary_segment])
    except Exception:
        logger.exception("Could not log submission (non-fatal, continuing).")


@app.route('/dashboard')
def dashboard():
    """A business/data-science facing view of the model and dataset behind
    the quiz: training-data composition, model performance, feature
    importance, and (if any visitors have completed the quiz) real usage
    stats — aimed at analysts and stakeholders evaluating the tool, not
    at quiz-takers."""

    # --- Training data composition -----------------------------------
    training_data_info = {'available': False}
    try:
        df = pd.read_csv(TRAINING_CSV)
        counts = df['Fashion Identity'].value_counts()
        training_data_info = {
            'available': True,
            'source_file': os.path.basename(TRAINING_CSV),
            'is_augmented': TRAINING_CSV == AUGMENTED_CSV,
            'total_rows': int(len(df)),
            'labels': counts.index.tolist(),
            'counts': counts.values.tolist(),
        }
    except Exception:
        logger.exception("Could not load training data for dashboard.")

    # --- Model performance metrics (from train_model.py / auto-train) --
    model_metrics = None
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH) as f:
                model_metrics = json.load(f)
        except Exception:
            logger.exception("Could not load model metrics for dashboard.")

    # --- Feature importance --------------------------------------------
    feature_importance = None
    try:
        raw_importance = classifier.get_feature_importance()
        feature_importance = {
            'labels': [k.split('.', 1)[-1].strip() for k in raw_importance.keys()],
            'scores': [round(v, 4) for v in raw_importance.values()],
        }
    except Exception:
        logger.exception("Could not compute feature importance for dashboard.")

    # --- EDA summary (chi-square tests, class balance) ------------------
    eda_summary = None
    if os.path.exists(EDA_SUMMARY_PATH):
        try:
            with open(EDA_SUMMARY_PATH) as f:
                eda_summary = json.load(f)
        except Exception:
            logger.exception("Could not load EDA summary for dashboard.")

    # --- Live usage stats (real quiz completions, if any) ---------------
    usage_stats = {'available': False}
    if os.path.exists(SUBMISSIONS_CSV):
        try:
            subs = pd.read_csv(SUBMISSIONS_CSV)
            if not subs.empty:
                counts = subs['primary_segment'].value_counts()
                usage_stats = {
                    'available': True,
                    'total_submissions': int(len(subs)),
                    'labels': counts.index.tolist(),
                    'counts': counts.values.tolist(),
                    'last_submission': subs['timestamp'].iloc[-1],
                }
        except Exception:
            logger.exception("Could not load live submissions for dashboard.")

    return render_template(
        'dashboard.html',
        training_data_info=training_data_info,
        model_metrics=model_metrics,
        feature_importance=feature_importance,
        eda_summary=eda_summary,
        usage_stats=usage_stats,
    )


if __name__ == '__main__':
    app.run(debug=True) 