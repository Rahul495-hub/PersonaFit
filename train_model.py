import pandas as pd
import numpy as np
from fashion_classifier import FashionClassifier
import os

# Resolve paths relative to this file, not the current working directory,
# so `python train_model.py` works the same whether run from inside the
# project folder or from anywhere else.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_CSV = os.path.join(DATA_DIR, 'fashion_data.csv')
AUGMENTED_CSV = os.path.join(DATA_DIR, 'fashion_data_augmented.csv')

def create_sample_data():
    """Create a sample dataset for training"""
    # Sample data with different fashion identities
    data = {
        '1.How often do you shop for new clothes?': ['Weekly', 'Monthly', 'Quarterly', 'Weekly', 'Monthly', 'Quarterly', 'Weekly', 'Monthly'],
        '2.Where do you typically shop for clothes?': ['Fast Fashion', 'Department Stores', 'Luxury Boutiques', 'Fast Fashion', 'Department Stores', 'Luxury Boutiques', 'Fast Fashion', 'Department Stores'],
        '3.What influences your clothing purchases the most?': ['Trends', 'Price', 'Quality', 'Trends', 'Price', 'Quality', 'Trends', 'Price'],
        '4.How would you describe your go-to daily outfit?': ['Trendy', 'Business Casual', 'Formal', 'Trendy', 'Business Casual', 'Formal', 'Trendy', 'Business Casual'],
        '5.If you had to choose, would you prefer timeless pieces or trendy items?': ['Trendy', 'Mix', 'Timeless', 'Trendy', 'Mix', 'Timeless', 'Trendy', 'Mix'],
        'Fashion Identity': ['Trendsetter', 'Budget Conscious', 'Luxury', 'Trendsetter', 'Budget Conscious', 'Luxury', 'Trendsetter', 'Budget Conscious']
    }
    
    return pd.DataFrame(data)

def main():
    # Only generate the tiny bundled sample dataset if no real dataset
    # exists yet. Previously this ran unconditionally and silently
    # overwrote data/fashion_data.csv (including any larger, real dataset)
    # with an 8-row placeholder every time the script was run.
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_CSV):
        df = create_sample_data()
        df.to_csv(DATA_CSV, index=False)
        print(f"No existing dataset found — wrote bundled sample data to {DATA_CSV}")

    # Prefer the SMOTE-augmented dataset if one has been generated (see
    # data_augmentation.py), since it's larger and class-balanced. Falls
    # back to the raw dataset if augmentation hasn't been run yet.
    if os.path.exists(AUGMENTED_CSV):
        training_csv = AUGMENTED_CSV
        print(f"Using SMOTE-augmented dataset: {AUGMENTED_CSV}")
    else:
        training_csv = DATA_CSV
        print(f"No augmented dataset found — training on the raw dataset: {DATA_CSV}")
        print("(Run `python data_augmentation.py` first for a larger, class-balanced training set.)")

    # Initialize classifier
    classifier = FashionClassifier()

    # Train the model
    metrics = classifier.train_model(training_csv)
    print(f"\nModel trained on {metrics['n_eval_samples']} held-out evaluation rows")
    print(f"Accuracy: {metrics['accuracy']:.3f}")

    print("\nPer-class performance:")
    report = metrics['classification_report']
    for label in metrics['labels']:
        stats = report[label]
        print(f"  {label:20s} precision={stats['precision']:.2f}  recall={stats['recall']:.2f}  f1={stats['f1-score']:.2f}  support={int(stats['support'])}")

    print("\nConfusion matrix (rows=actual, cols=predicted):")
    labels = metrics['labels']
    header = "".join(f"{l[:12]:>14s}" for l in labels)
    print(f"{'':20s}{header}")
    for label, row in zip(labels, metrics['confusion_matrix']):
        row_str = "".join(f"{v:>14d}" for v in row)
        print(f"{label:20s}{row_str}")

    print(f"\nFull metrics saved to {os.path.join(classifier.models_dir, 'metrics.json')}")

    # Test prediction
    test_data = {
        '1.How often do you shop for new clothes?': 'Weekly',
        '2.Where do you typically shop for clothes?': 'Fast Fashion',
        '3.What influences your clothing purchases the most?': 'Trends',
        '4.How would you describe your go-to daily outfit?': 'Trendy',
        '5.If you had to choose, would you prefer timeless pieces or trendy items?': 'Trendy'
    }

    prediction = classifier.predict(test_data)
    print(f"\nTest prediction: {prediction['prediction']}")
    print(f"Probabilities: {prediction['probabilities']}")

    # Get feature importance
    importance = classifier.get_feature_importance()
    print("\nFeature Importance:")
    for feature, imp in importance.items():
        print(f"{feature}: {imp:.4f}")

if __name__ == "__main__":
    main()