"""
Supervised Machine Learning Classification Model
=================================================
Implements a complete classification pipeline using scikit-learn:
  - Data loading (Breast Cancer dataset — a real-world medical dataset)
  - Data preprocessing (scaling, train/test split)
  - Training: Logistic Regression, Decision Tree, Random Forest, SVM
  - Performance evaluation (accuracy, classification report, confusion matrix)
  - Prediction on new/unseen samples
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_data():
    """Load the Breast Cancer Wisconsin dataset and return a DataFrame."""
    dataset = load_breast_cancer()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df["target"] = dataset.target
    print("Dataset shape:", df.shape)
    print("Class distribution:\n", df["target"].value_counts().rename(
        {i: name for i, name in enumerate(dataset.target_names)}
    ))
    print()
    return df, dataset.target_names


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------

def preprocess(df, test_size=0.2, random_state=42):
    """Split into features / labels, scale features, return train/test sets."""
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print()

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ---------------------------------------------------------------------------
# 3. Model Definitions
# ---------------------------------------------------------------------------

def build_classifiers(random_state=42):
    """Return a dict of classifier name → fitted-ready estimator."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=10000, random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state),
    }


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------

def train_classifiers(classifiers, X_train, y_train):
    """Fit every classifier on the training set."""
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        print(f"Trained: {name}")
    print()
    return classifiers


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------

def evaluate_classifiers(classifiers, X_train, X_test, y_train, y_test,
                         target_names, output_dir=None):
    """
    Print accuracy, classification report, and confusion matrix for each model.
    Optionally save confusion-matrix plots to *output_dir*.
    """
    results = {}

    for name, clf in classifiers.items():
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        cv_scores = cross_val_score(clf, X_train, y_train, cv=5,
                                    scoring="accuracy")

        print(f"{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        print(f"  Test accuracy       : {acc:.4f}")
        print(f"  CV accuracy (5-fold): {cv_scores.mean():.4f} "
              f"± {cv_scores.std():.4f}")
        print()
        print(classification_report(y_test, y_pred, target_names=target_names))

        cm = confusion_matrix(y_test, y_pred)
        _plot_confusion_matrix(cm, name, target_names, output_dir)

        results[name] = {
            "accuracy": acc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

    return results


def _plot_confusion_matrix(cm, model_name, target_names, output_dir=None):
    """Plot and optionally save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_name = model_name.replace(" ", "_")
        path = os.path.join(output_dir, f"cm_{safe_name}.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved confusion matrix to {path}")

    plt.close(fig)


def plot_accuracy_comparison(results, output_dir=None):
    """Bar chart comparing test accuracy across classifiers."""
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, accuracies, color="steelblue", edgecolor="black")
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Classifier Accuracy Comparison")
    ax.bar_label(bars, fmt="%.4f", padding=3)
    fig.tight_layout()

    if output_dir:
        path = os.path.join(output_dir, "accuracy_comparison.png")
        fig.savefig(path, dpi=150)
        print(f"\nSaved accuracy comparison chart to {path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Prediction on New / Unseen Data
# ---------------------------------------------------------------------------

def predict_new_samples(classifiers, scaler, X_new_raw, target_names):
    """
    Scale *X_new_raw* with the fitted scaler and predict with every classifier.

    Parameters
    ----------
    X_new_raw : array-like of shape (n_samples, n_features)
        Raw (unscaled) feature values for new samples.
    """
    X_new_scaled = scaler.transform(X_new_raw)

    print("=" * 60)
    print("  Predictions on new samples")
    print("=" * 60)
    for name, clf in classifiers.items():
        preds = clf.predict(X_new_scaled)
        probs = clf.predict_proba(X_new_scaled) if hasattr(clf, "predict_proba") else None
        labels = [target_names[p] for p in preds]
        print(f"\n  [{name}]")
        for i, (label, pred) in enumerate(zip(labels, preds)):
            if probs is not None:
                conf = probs[i][pred]
                print(f"    Sample {i + 1}: {label} (confidence {conf:.2%})")
            else:
                print(f"    Sample {i + 1}: {label}")


# ---------------------------------------------------------------------------
# 7. Main Entry Point
# ---------------------------------------------------------------------------

def main(save_plots=False, output_dir="outputs"):
    """Run the full supervised classification pipeline."""

    # --- Load ---
    df, target_names = load_data()

    # --- Preprocess ---
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    # --- Train ---
    classifiers = build_classifiers()
    classifiers = train_classifiers(classifiers, X_train, y_train)

    # --- Evaluate ---
    _output_dir = output_dir if save_plots else None
    results = evaluate_classifiers(
        classifiers, X_train, X_test, y_train, y_test,
        target_names, output_dir=_output_dir
    )

    # --- Accuracy comparison chart ---
    plot_accuracy_comparison(results, output_dir=_output_dir)

    # --- Predict on a couple of hand-picked test samples ---
    # Use the first two rows from the original test set as "new" samples
    dataset = load_breast_cancer()
    X_new_raw = pd.DataFrame(
        dataset.data[:2], columns=dataset.feature_names
    )
    predict_new_samples(classifiers, scaler, X_new_raw, target_names)

    # --- Summary ---
    best = max(results, key=lambda n: results[n]["accuracy"])
    print(f"\nBest model: {best} "
          f"(accuracy={results[best]['accuracy']:.4f})")

    return results


if __name__ == "__main__":
    main(save_plots=True)
