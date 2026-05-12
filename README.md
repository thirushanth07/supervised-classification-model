# Supervised Classification Model

A complete supervised machine learning classification pipeline built with **Python** and **scikit-learn**.

## What it does

| Step | Details |
|---|---|
| **Data loading** | Breast Cancer Wisconsin dataset (569 samples, 30 features, binary target) |
| **Preprocessing** | Stratified 80/20 train/test split · `StandardScaler` feature scaling |
| **Algorithms** | Logistic Regression · Decision Tree · Random Forest · SVM (RBF kernel) |
| **Evaluation** | Test accuracy · 5-fold cross-validation · Classification report · Confusion matrix |
| **Prediction** | Predict class labels and confidence scores for new/unseen samples |

## Project structure

```
supervised-classification-model/
├── classification_model.py       # Full pipeline (load → preprocess → train → evaluate → predict)
├── test_classification_model.py  # pytest test suite (11 tests)
├── requirements.txt              # Python dependencies
└── outputs/                      # Saved plots (created at runtime)
    ├── cm_Logistic_Regression.png
    ├── cm_Decision_Tree.png
    ├── cm_Random_Forest.png
    ├── cm_SVM.png
    └── accuracy_comparison.png
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the pipeline

```bash
python classification_model.py
```

This trains all four classifiers, prints per-model metrics, and saves confusion-matrix and accuracy-comparison plots to the `outputs/` folder.

## Run the tests

```bash
pytest test_classification_model.py -v
```

## Sample output

```
Dataset shape: (569, 31)
Training samples : 455  |  Test samples: 114

Logistic Regression — Test accuracy: 0.9825  |  CV: 0.9802 ± 0.0128
Decision Tree       — Test accuracy: 0.9123  |  CV: 0.9099 ± 0.0189
Random Forest       — Test accuracy: 0.9561  |  CV: 0.9538 ± 0.0235
SVM                 — Test accuracy: 0.9825  |  CV: 0.9714 ± 0.0179

Best model: Logistic Regression (accuracy=0.9825)
```
