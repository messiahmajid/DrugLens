"""
Model training, evaluation, and prediction for DrugLens.

Uses XGBoost (gradient-boosted decision trees) for binary classification
of drug-target binding. XGBoost is chosen over deep learning here because:
1. It performs very well on tabular/fingerprint data
2. It trains in minutes, not hours
3. It integrates cleanly with SHAP for explainability
4. It handles class imbalance natively via scale_pos_weight
"""

import json
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
import joblib
from pathlib import Path


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier for drug-target interaction prediction.

    Handles class imbalance (typically many more non-binders than binders)
    via scale_pos_weight, which upweights the minority class.
    """
    # Compute class imbalance ratio for weighting
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_weight = n_neg / max(n_pos, 1)

    print(f"  Class balance — positive: {n_pos}, negative: {n_neg}")
    print(f"  scale_pos_weight: {scale_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=50,
    )

    print(f"  Best iteration: {model.best_iteration}")
    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate model performance on a held-out test set.

    Returns a dictionary of metrics including accuracy, precision, recall,
    F1, AUROC, and AUPRC (average precision).
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_test, y_prob)),
        "auprc": float(average_precision_score(y_test, y_prob)),
    }

    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    print("\n  Test Set Results:")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1 Score:  {metrics['f1']:.4f}")
    print(f"    AUROC:     {metrics['auroc']:.4f}")
    print(f"    AUPRC:     {metrics['auprc']:.4f}")
    print(f"    Confusion Matrix:\n      {cm}")

    return metrics


def predict_binding(
    model: xgb.XGBClassifier,
    features: np.ndarray,
) -> tuple[int, float]:
    """
    Predict binding for a single drug-target pair.

    Args:
        model: trained XGBoost classifier
        features: feature vector for the drug-target pair

    Returns:
        prediction: 0 (non-binding) or 1 (binding)
        confidence: probability of binding (0.0 to 1.0)
    """
    X = features.reshape(1, -1)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0, 1]
    return int(pred), float(prob)


def save_artifacts(
    model: xgb.XGBClassifier,
    metrics: dict,
    reference_data: dict,
    output_dir: str = "artifacts",
) -> None:
    """Save trained model and associated artifacts."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    joblib.dump(model, out / "model.joblib")
    joblib.dump(reference_data, out / "reference_db.joblib")

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Artifacts saved to {output_dir}/")


def load_artifacts(artifact_dir: str = "artifacts") -> tuple:
    """Load trained model and associated artifacts."""
    path = Path(artifact_dir)
    model = joblib.load(path / "model.joblib")
    reference_data = joblib.load(path / "reference_db.joblib")

    with open(path / "metrics.json", "r") as f:
        metrics = json.load(f)

    return model, metrics, reference_data
