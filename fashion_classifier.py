import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import json
import logging

logger = logging.getLogger(__name__)

# Default models directory, resolved relative to this file rather than the
# current working directory, so the classifier works the same whether it's
# run as `python app.py`, `python train_model.py`, or imported from
# somewhere else entirely.
_DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


class FashionClassifier:
    def __init__(self, models_dir=None):
        self.model = None
        self.label_encoders = {}
        self.models_dir = models_dir or _DEFAULT_MODELS_DIR
        self.feature_columns = [
            '1.How often do you shop for new clothes?',
            '2.Where do you typically shop for clothes?',
            '3.What influences your clothing purchases the most?',
            '4.How would you describe your go-to daily outfit?',
            '5.If you had to choose, would you prefer timeless pieces or trendy items?'
        ]

    def load_data(self, csv_path):
        """Load and preprocess the data from CSV file"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Training data not found at '{csv_path}'. Make sure the file "
                "exists, or pass the correct path to train_model()."
            )

        df = pd.read_csv(csv_path)

        missing_cols = [c for c in self.feature_columns + ['Fashion Identity'] if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Training CSV is missing expected column(s): {missing_cols}. "
                "Check that the header row matches the quiz questions exactly."
            )

        # Clean and preprocess the data
        df = df.dropna(subset=self.feature_columns + ['Fashion Identity'])
        if df.empty:
            raise ValueError(
                "No usable rows left after dropping missing values — check "
                "that the CSV has data in every required column."
            )

        # Encode categorical variables. These columns are always categorical
        # by design (quiz answer options), so encode them unconditionally
        # rather than relying on a dtype check: pandas represents string
        # columns as 'object' on 2.x but as a distinct 'str' dtype on 3.x,
        # so a dtype == 'object' check silently skips encoding on newer
        # pandas and breaks training with no obvious error.
        for column in self.feature_columns:
            self.label_encoders[column] = LabelEncoder()
            df[column] = self.label_encoders[column].fit_transform(df[column])

        return df

    def train_model(self, csv_path):
        """Train the Random Forest model.

        Returns a dict with 'accuracy', 'classification_report' (per-class
        precision/recall/F1), and 'confusion_matrix' (with matching
        'labels') — used both for the console summary in train_model.py
        and for the business/data-science dashboard in the Flask app.
        """
        # Load and preprocess the data
        df = self.load_data(csv_path)

        # Prepare features and target
        X = df[self.feature_columns]
        y = df['Fashion Identity']

        # Guard against a dataset too small or too imbalanced to stratify/
        # split sensibly (this would otherwise raise a confusing sklearn
        # error deep inside train_test_split).
        class_counts = y.value_counts()
        if len(df) < 10 or class_counts.min() < 2:
            logger.warning(
                "Training data is very small (%d rows). Skipping the "
                "train/test split and training on the full dataset instead; "
                "the reported accuracy will be less meaningful with so "
                "little data.", len(df)
            )
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            metrics = self._evaluate(X, y)
        else:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train the model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            metrics = self._evaluate(X_test, y_test)

        # Save the model, label encoders, and evaluation metrics
        self.save_model()
        self._save_metrics(metrics)

        return metrics

    def _evaluate(self, X_eval, y_eval):
        """Compute accuracy, a full per-class classification report, and a
        confusion matrix on the given (held-out, ideally) data."""
        y_pred = self.model.predict(X_eval)
        labels = list(self.model.classes_)

        return {
            'accuracy': float(self.model.score(X_eval, y_eval)),
            'classification_report': classification_report(
                y_eval, y_pred, labels=labels, output_dict=True, zero_division=0
            ),
            'confusion_matrix': confusion_matrix(y_eval, y_pred, labels=labels).tolist(),
            'labels': labels,
            'n_eval_samples': int(len(y_eval)),
        }

    def _save_metrics(self, metrics):
        """Persist evaluation metrics alongside the model so the dashboard
        can display them without re-running training."""
        os.makedirs(self.models_dir, exist_ok=True)
        with open(os.path.join(self.models_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

    def save_model(self):
        """Save the trained model and label encoders"""
        os.makedirs(self.models_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(self.models_dir, 'fashion_model.joblib'))
        joblib.dump(self.label_encoders, os.path.join(self.models_dir, 'label_encoders.joblib'))

    def load_model(self):
        """Load the trained model and label encoders.

        Returns False (rather than raising) if no model file exists yet, or
        if the saved model can't be unpickled — for example because it was
        saved with a different, incompatible scikit-learn version than the
        one currently installed. Callers can treat False as "no usable
        model" and train a fresh one instead of crashing.
        """
        model_path = os.path.join(self.models_dir, 'fashion_model.joblib')
        encoders_path = os.path.join(self.models_dir, 'label_encoders.joblib')

        if not (os.path.exists(model_path) and os.path.exists(encoders_path)):
            return False

        try:
            self.model = joblib.load(model_path)
            self.label_encoders = joblib.load(encoders_path)
            return True
        except Exception:
            logger.exception(
                "Found saved model files but couldn't load them (likely a "
                "version mismatch). Treating as no model available."
            )
            self.model = None
            self.label_encoders = {}
            return False

    def predict(self, user_data):
        """Predict fashion identity for new user data"""
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not trained. Please train the model first.")

        # Convert user data to DataFrame
        df = pd.DataFrame([user_data])

        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected answer(s) for: {missing}")

        # Encode categorical variables
        for column in self.feature_columns:
            if column in self.label_encoders:
                raw_value = df[column].iloc[0]
                try:
                    df[column] = self.label_encoders[column].transform([raw_value])
                except ValueError:
                    # The value wasn't seen during training (e.g. quiz
                    # options were edited without retraining the model).
                    # Fail with a clear, actionable message instead of a
                    # cryptic sklearn ValueError.
                    known = list(self.label_encoders[column].classes_)
                    raise ValueError(
                        f"Unrecognized answer '{raw_value}' for question "
                        f"'{column}'. The model was trained on: {known}. "
                        "Retrain the model (python train_model.py) if the "
                        "quiz questions/options have changed."
                    )

        # Make prediction
        X = df[self.feature_columns]
        prediction = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        return {
            'prediction': prediction[0],
            'probabilities': probabilities[0].tolist()
        }

    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not trained. Please train the model first.")

        importance = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_columns, importance))
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
