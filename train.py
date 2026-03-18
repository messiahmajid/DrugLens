"""
DrugLens Training Pipeline
===========================

Run this script to:
1. Download the Davis kinase binding dataset
2. Engineer molecular and protein features
3. Train an XGBoost classifier
4. Evaluate on held-out test set
5. Save all artifacts for the Streamlit app

Usage:
    python train.py

Outputs saved to artifacts/:
    - model.joblib          (trained XGBoost model)
    - reference_db.joblib   (drug fingerprints for similarity search)
    - metrics.json          (test set evaluation metrics)
"""

import time
import numpy as np
from src.data import load_davis_dataset, get_dataset_stats
from src.features import featurize_dataset, get_all_feature_names
from src.model import train_model, evaluate_model, save_artifacts
from src.similarity import build_reference_database


def main():
    print("=" * 60)
    print("  DrugLens — Training Pipeline")
    print("=" * 60)
    start = time.time()

    # ── Step 1: Load Data ──────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    data = load_davis_dataset(threshold=30.0)
    stats = get_dataset_stats(data)
    print(f"  Dataset: {stats['total_pairs']} pairs, "
          f"{stats['unique_drugs']} drugs, {stats['unique_targets']} targets")
    print(f"  Binding ratio: {stats['binding_ratio']:.3f}")

    # ── Step 2: Feature Engineering ────────────────────────────
    print("\n[2/5] Engineering features...")
    print("  Processing training set...")
    X_train, y_train, _ = featurize_dataset(data["train"])
    print("  Processing validation set...")
    X_valid, y_valid, _ = featurize_dataset(data["valid"])
    print("  Processing test set...")
    X_test, y_test, _ = featurize_dataset(data["test"])

    feature_names = get_all_feature_names()
    print(f"  Feature vector dimension: {X_train.shape[1]}")
    print(f"    Drug features: 2048 (Morgan) + 10 (descriptors) = 2058")
    print(f"    Protein features: 20 (AAC) + 400 (dipeptide) + 5 (properties) = 425")
    print(f"    Total: {len(feature_names)}")

    # ── Step 3: Train Model ────────────────────────────────────
    print("\n[3/5] Training XGBoost model...")
    model = train_model(X_train, y_train, X_valid, y_valid)

    # ── Step 4: Evaluate ───────────────────────────────────────
    print("\n[4/5] Evaluating on test set...")
    metrics = evaluate_model(model, X_test, y_test)
    metrics["dataset_stats"] = stats
    metrics["feature_dim"] = int(X_train.shape[1])

    # ── Step 5: Save Artifacts ─────────────────────────────────
    print("\n[5/5] Saving artifacts...")

    # Build reference database for similarity search
    all_smiles = list(data["train"]["Drug"]) + list(data["valid"]["Drug"])
    all_labels = np.concatenate([y_train, y_valid])
    ref_db = build_reference_database(all_smiles, all_labels)

    save_artifacts(
        model=model,
        metrics=metrics,
        reference_data=ref_db,
        output_dir="artifacts",
    )

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  Training complete in {elapsed:.1f} seconds.")
    print(f"  AUROC: {metrics['auroc']:.4f} | F1: {metrics['f1']:.4f}")
    print(f"  Run 'streamlit run app.py' to launch the app.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
