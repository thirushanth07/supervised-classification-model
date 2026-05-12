"""
Tests for the supervised classification pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

from classification_model import (
    load_data,
    preprocess,
    build_classifiers,
    train_classifiers,
    evaluate_classifiers,
    predict_new_samples,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline():
    """Run the full pipeline once and share results across tests."""
    df, target_names = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    classifiers = build_classifiers()
    classifiers = train_classifiers(classifiers, X_train, y_train)
    return {
        "df": df,
        "target_names": target_names,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "classifiers": classifiers,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def test_load_data_shape():
    df, target_names = load_data()
    assert df.shape[0] == 569, "Breast Cancer dataset should have 569 rows"
    assert "target" in df.columns


def test_load_data_target_names():
    _, target_names = load_data()
    assert set(target_names) == {"malignant", "benign"}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def test_preprocess_split_sizes():
    df, _ = load_data()
    X_train, X_test, y_train, y_test, _ = preprocess(df, test_size=0.2)
    total = len(y_train) + len(y_test)
    assert total == 569
    assert abs(len(y_test) / total - 0.2) < 0.05


def test_preprocess_scaling():
    """Training features should be approximately standard-normal."""
    df, _ = load_data()
    X_train, _, _, _, _ = preprocess(df)
    assert abs(X_train.mean()) < 0.1
    assert abs(X_train.std() - 1.0) < 0.1


def test_preprocess_no_nan():
    df, _ = load_data()
    X_train, X_test, _, _, _ = preprocess(df)
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

def test_build_classifiers_names():
    classifiers = build_classifiers()
    expected = {"Logistic Regression", "Decision Tree", "Random Forest", "SVM"}
    assert set(classifiers.keys()) == expected


def test_classifiers_fitted(pipeline):
    from sklearn.utils.validation import check_is_fitted
    for name, clf in pipeline["classifiers"].items():
        check_is_fitted(clf)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_evaluate_returns_all_models(pipeline):
    results = evaluate_classifiers(
        pipeline["classifiers"],
        pipeline["X_train"],
        pipeline["X_test"],
        pipeline["y_train"],
        pipeline["y_test"],
        pipeline["target_names"],
    )
    expected = {"Logistic Regression", "Decision Tree", "Random Forest", "SVM"}
    assert set(results.keys()) == expected


def test_evaluate_accuracy_above_threshold(pipeline):
    results = evaluate_classifiers(
        pipeline["classifiers"],
        pipeline["X_train"],
        pipeline["X_test"],
        pipeline["y_train"],
        pipeline["y_test"],
        pipeline["target_names"],
    )
    for name, metrics in results.items():
        assert metrics["accuracy"] > 0.85, (
            f"{name} accuracy {metrics['accuracy']:.4f} is below 0.85"
        )


def test_evaluate_result_keys(pipeline):
    results = evaluate_classifiers(
        pipeline["classifiers"],
        pipeline["X_train"],
        pipeline["X_test"],
        pipeline["y_train"],
        pipeline["y_test"],
        pipeline["target_names"],
    )
    for name, metrics in results.items():
        assert "accuracy" in metrics
        assert "cv_mean" in metrics
        assert "cv_std" in metrics


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def test_predict_new_samples(pipeline, capsys):
    dataset = load_breast_cancer()
    X_new_raw = pd.DataFrame(dataset.data[:3], columns=dataset.feature_names)
    predict_new_samples(
        pipeline["classifiers"],
        pipeline["scaler"],
        X_new_raw,
        pipeline["target_names"],
    )
    captured = capsys.readouterr()
    assert "Sample 1" in captured.out
    assert "Sample 2" in captured.out
    assert "Sample 3" in captured.out
