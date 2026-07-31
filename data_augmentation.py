"""
data_augmentation.py — Increase the training set size using SMOTE.

Why SMOTENC/SMOTEN instead of plain SMOTE
------------------------------------------
Classic SMOTE interpolates *numeric* feature values between a sample and
its nearest neighbors, which doesn't make sense for our data: every
feature here is categorical (e.g. "Weekly"/"Monthly"/"Quarterly"), so
interpolating between encoded integers would invent categories that don't
exist. imbalanced-learn ships two variants built for this:

  - SMOTEN     — all features are categorical (our case)
  - SMOTENC    — a mix of categorical and continuous features

Both generate synthetic samples by taking the *majority vote* of a
sample's nearest neighbors for each categorical column, instead of a
numeric interpolation — so every synthetic row is still built from real,
valid category values, just recombined in new ways.

This script:
  1. Loads the raw survey-style dataset (data/fashion_data.csv).
  2. Encodes it, then oversamples every class up to a common target count
     using SMOTEN — both balancing the classes AND growing the dataset.
  3. Decodes back to readable category labels and writes the result to
     data/fashion_data_augmented.csv, leaving the original raw file
     untouched (a data scientist can always regenerate this file, or
     diff it against the source).

Falls back to a manual bootstrap-style oversampler if imbalanced-learn
isn't installed, so the pipeline still runs (with a clear log message)
rather than hard failing on a missing optional dependency.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_CSV = os.path.join(BASE_DIR, 'data', 'fashion_data.csv')
AUGMENTED_CSV = os.path.join(BASE_DIR, 'data', 'fashion_data_augmented.csv')

FEATURE_COLUMNS = [
    '1.How often do you shop for new clothes?',
    '2.Where do you typically shop for clothes?',
    '3.What influences your clothing purchases the most?',
    '4.How would you describe your go-to daily outfit?',
    '5.If you had to choose, would you prefer timeless pieces or trendy items?'
]
TARGET_COLUMN = 'Fashion Identity'


def _smoten_augment(X_enc, y, target_per_class, random_state=42):
    """Use imbalanced-learn's SMOTEN (categorical-only SMOTE)."""
    from imblearn.over_sampling import SMOTEN

    class_counts = y.value_counts()
    # SMOTE's k_neighbors must be smaller than the smallest class size;
    # cap it defensively so this doesn't blow up on a tiny/imbalanced
    # input dataset.
    k_neighbors = max(1, min(5, class_counts.min() - 1))

    sampling_strategy = {cls: max(count, target_per_class) for cls, count in class_counts.items()}

    smoten = SMOTEN(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
    X_res, y_res = smoten.fit_resample(X_enc, y)
    return X_res, y_res, "SMOTEN (imbalanced-learn)"


def _manual_bootstrap_augment(X_enc, y, target_per_class, random_state=42):
    """Fallback used only if imbalanced-learn isn't installed.

    Not true SMOTE — it's a simple stratified bootstrap resample (with
    replacement) up to the same target count per class, so the rest of
    the pipeline (train_model.py, the Flask app) still has a larger
    dataset to work with. Clearly labeled as a fallback in the output.
    """
    rng = np.random.RandomState(random_state)
    frames = []
    for cls in y.unique():
        idx = np.where(y == cls)[0]
        n_needed = max(target_per_class, len(idx))
        chosen = rng.choice(idx, size=n_needed, replace=True)
        frames.append((X_enc.iloc[chosen], y.iloc[chosen]))
    X_res = pd.concat([f[0] for f in frames], ignore_index=True)
    y_res = pd.concat([f[1] for f in frames], ignore_index=True)
    return X_res, y_res, "stratified bootstrap (fallback — install imbalanced-learn for true SMOTE)"


def augment(input_csv=RAW_CSV, output_csv=AUGMENTED_CSV, target_per_class=100, random_state=42):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Could not find source dataset at '{input_csv}'.")

    df = pd.read_csv(input_csv).dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    if df.empty:
        raise ValueError("Source dataset has no usable rows after dropping missing values.")

    print(f"Loaded {len(df)} raw rows from {input_csv}")
    print("Class distribution before augmentation:")
    print(df[TARGET_COLUMN].value_counts().to_string())

    # Encode every column (features + target) to integers, since SMOTE
    # variants operate on numeric arrays.
    encoders = {}
    X_enc = pd.DataFrame(index=df.index)
    for col in FEATURE_COLUMNS:
        enc = LabelEncoder()
        X_enc[col] = enc.fit_transform(df[col])
        encoders[col] = enc

    y_enc_encoder = LabelEncoder()
    y = pd.Series(y_enc_encoder.fit_transform(df[TARGET_COLUMN]), name=TARGET_COLUMN)

    try:
        X_res, y_res, method = _smoten_augment(X_enc, y, target_per_class, random_state)
    except ImportError:
        print(
            "\n'imbalanced-learn' isn't installed — falling back to a "
            "simple stratified bootstrap resample instead of true SMOTE.\n"
            "Run: pip install imbalanced-learn   (then re-run this script) "
            "for genuine synthetic samples.\n"
        )
        X_res, y_res, method = _manual_bootstrap_augment(X_enc, y, target_per_class, random_state)

    # Decode back to human-readable category labels.
    out = pd.DataFrame()
    for col in FEATURE_COLUMNS:
        # Values from SMOTEN are guaranteed to be valid known category
        # codes (majority-vote reconstruction, not interpolation), so a
        # plain inverse_transform is safe.
        out[col] = encoders[col].inverse_transform(X_res[col].astype(int))
    out[TARGET_COLUMN] = y_enc_encoder.inverse_transform(y_res.astype(int))

    # Shuffle so classes aren't grouped in contiguous blocks.
    out = out.sample(frac=1, random_state=random_state).reset_index(drop=True)

    out.to_csv(output_csv, index=False)

    print(f"\nAugmentation method: {method}")
    print(f"Wrote {len(out)} rows to {output_csv}")
    print("Class distribution after augmentation:")
    print(out[TARGET_COLUMN].value_counts().to_string())

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment the fashion identity dataset using SMOTE.")
    parser.add_argument("--target-per-class", type=int, default=100,
                         help="Minimum number of rows per class after augmentation (default: 100)")
    parser.add_argument("--input", default=RAW_CSV, help="Path to the source CSV")
    parser.add_argument("--output", default=AUGMENTED_CSV, help="Path to write the augmented CSV")
    args = parser.parse_args()

    augment(input_csv=args.input, output_csv=args.output, target_per_class=args.target_per_class)
