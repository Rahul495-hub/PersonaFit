"""
eda.py — Exploratory Data Analysis for the Fashion Identity Classifier.

Generates a set of PNG charts and a markdown summary report aimed at data
analysts/scientists reviewing the dataset behind the classifier:

  - Class (target) distribution
  - Each feature's distribution, broken down by segment
  - A Cramer's V association matrix across all categorical variables
    (Cramer's V is the appropriate analog of a correlation matrix for
    nominal/categorical data — Pearson correlation doesn't apply here
    since nothing is numeric/ordinal)
  - A chi-square test of independence between each feature and the
    target, to flag which quiz questions actually carry predictive signal

Run with:  python eda.py
Outputs go to reports/figures/*.png and reports/eda_summary.md
"""

import os
import json
import itertools
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe backend, no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_CSV = os.path.join(DATA_DIR, 'fashion_data.csv')
AUGMENTED_CSV = os.path.join(DATA_DIR, 'fashion_data_augmented.csv')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

FEATURE_COLUMNS = [
    '1.How often do you shop for new clothes?',
    '2.Where do you typically shop for clothes?',
    '3.What influences your clothing purchases the most?',
    '4.How would you describe your go-to daily outfit?',
    '5.If you had to choose, would you prefer timeless pieces or trendy items?'
]
TARGET_COLUMN = 'Fashion Identity'

sns.set_theme(style="whitegrid", palette="deep")


def cramers_v(x, y):
    """Cramer's V: a 0-1 measure of association between two categorical
    variables, derived from the chi-square statistic. 0 = no association,
    1 = perfect association. This is the standard categorical analog of a
    Pearson correlation coefficient."""
    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    r, k = confusion.shape
    phi2 = chi2 / n
    # Bias correction (Bergsma 2013), standard practice for small samples
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return np.sqrt(phi2corr / denom) if denom > 0 else 0.0


def plot_class_distribution(df, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = df[TARGET_COLUMN].value_counts().index
    sns.countplot(data=df, y=TARGET_COLUMN, order=order, ax=ax)
    ax.set_title('Fashion Identity — class distribution')
    ax.set_xlabel('Number of respondents')
    ax.set_ylabel('')
    for container in ax.containers:
        ax.bar_label(container)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_breakdowns(df, path):
    fig, axes = plt.subplots(len(FEATURE_COLUMNS), 1, figsize=(8, 4 * len(FEATURE_COLUMNS)))
    for ax, col in zip(axes, FEATURE_COLUMNS):
        crosstab = pd.crosstab(df[col], df[TARGET_COLUMN], normalize='index')
        crosstab.plot(kind='barh', stacked=True, ax=ax, colormap='viridis')
        ax.set_title(col)
        ax.set_xlabel('Share of respondents')
        ax.set_ylabel('')
        ax.legend(title='Segment', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_association_heatmap(df, path):
    all_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    n = len(all_cols)
    matrix = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        if i == j:
            matrix[i, j] = 1.0
        elif i < j:
            v = cramers_v(df[all_cols[i]], df[all_cols[j]])
            matrix[i, j] = v
            matrix[j, i] = v

    short_labels = [c.split('.', 1)[-1][:28] + ('…' if len(c) > 30 else '') for c in all_cols[:-1]] + [TARGET_COLUMN]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=short_labels, yticklabels=short_labels,
                cmap='mako', vmin=0, vmax=1, ax=ax, cbar_kws={'label': "Cramer's V"})
    ax.set_title("Association between quiz questions and segment (Cramer's V)")
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return matrix, short_labels


def chi_square_tests(df):
    results = []
    for col in FEATURE_COLUMNS:
        table = pd.crosstab(df[col], df[TARGET_COLUMN])
        chi2, p, dof, _ = chi2_contingency(table)
        results.append({
            'feature': col,
            'chi2': round(float(chi2), 3),
            'p_value': float(p),
            'degrees_of_freedom': int(dof),
            'significant_at_0.05': bool(p < 0.05),
        })
    return sorted(results, key=lambda r: r['p_value'])


def run(data_path=None):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if data_path is None:
        data_path = AUGMENTED_CSV if os.path.exists(AUGMENTED_CSV) else RAW_CSV
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No dataset found at '{data_path}'.")

    df = pd.read_csv(data_path).dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    print(f"Running EDA on {len(df)} rows from {data_path}")

    class_dist_path = os.path.join(FIGURES_DIR, 'class_distribution.png')
    plot_class_distribution(df, class_dist_path)

    breakdown_path = os.path.join(FIGURES_DIR, 'feature_breakdown_by_segment.png')
    plot_feature_breakdowns(df, breakdown_path)

    heatmap_path = os.path.join(FIGURES_DIR, 'association_heatmap.png')
    assoc_matrix, assoc_labels = plot_association_heatmap(df, heatmap_path)

    chi2_results = chi_square_tests(df)

    class_counts = df[TARGET_COLUMN].value_counts()
    class_balance_ratio = float(class_counts.min() / class_counts.max())

    summary = {
        'n_rows': int(len(df)),
        'class_counts': class_counts.to_dict(),
        'class_balance_ratio': round(class_balance_ratio, 3),
        'chi_square_tests': chi2_results,
        'most_predictive_feature': chi2_results[0]['feature'],
        'least_predictive_feature': chi2_results[-1]['feature'],
    }

    with open(os.path.join(REPORTS_DIR, 'eda_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    _write_markdown_report(summary)

    print(f"\nWrote figures to {FIGURES_DIR}")
    print(f"Wrote summary to {os.path.join(REPORTS_DIR, 'eda_summary.md')}")
    return summary


def _write_markdown_report(summary):
    lines = []
    lines.append("# Exploratory Data Analysis — Fashion Identity Classifier\n")
    lines.append(f"**Rows analyzed:** {summary['n_rows']}\n")
    lines.append("## Class distribution\n")
    for label, count in summary['class_counts'].items():
        lines.append(f"- **{label}**: {count}")
    lines.append(f"\nClass balance ratio (smallest/largest class): **{summary['class_balance_ratio']}** "
                 f"(1.0 = perfectly balanced)\n")
    lines.append("![Class distribution](figures/class_distribution.png)\n")

    lines.append("## Feature associations with segment (chi-square test)\n")
    lines.append("| Feature | chi² | p-value | Significant (p<0.05) |")
    lines.append("|---|---|---|---|")
    for r in summary['chi_square_tests']:
        lines.append(f"| {r['feature']} | {r['chi2']} | {r['p_value']:.4g} | {'Yes' if r['significant_at_0.05'] else 'No'} |")
    lines.append(f"\n**Most predictive question:** {summary['most_predictive_feature']}")
    lines.append(f"\n**Least predictive question:** {summary['least_predictive_feature']}\n")

    lines.append("![Association heatmap](figures/association_heatmap.png)\n")
    lines.append("## Feature breakdown by segment\n")
    lines.append("![Feature breakdown](figures/feature_breakdown_by_segment.png)\n")

    lines.append("## Notes\n")
    lines.append(
        "- Cramer's V measures association strength for categorical variables (0 = none, "
        "1 = perfect), used here in place of a Pearson correlation matrix, which doesn't "
        "apply to non-numeric data.\n"
        "- The chi-square test flags whether each quiz question's answers are statistically "
        "associated with the assigned segment, independent of the classifier itself — useful "
        "for spotting low-value survey questions that could be dropped or reworded.\n"
        "- This dataset is currently synthetic (rule-generated with light noise); expect "
        "these statistics to look artificially clean compared to real survey responses. "
        "Swap in real data via data/fashion_data.csv and re-run this script."
    )

    with open(os.path.join(REPORTS_DIR, 'eda_summary.md'), 'w') as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run()
