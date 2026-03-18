"""
Data loading and processing for DrugLens.

Downloads the Davis kinase binding dataset directly from the DeepDTA
repository (Ozturk et al.), a well-known academic source for this benchmark.

The Davis dataset contains ~30,000 experimentally measured binding affinities
(Kd values) between 68 drugs and 442 protein kinase targets.
"""

import os
import pickle
import numpy as np
import pandas as pd
import json
import urllib.request
from pathlib import Path
from sklearn.model_selection import train_test_split


# Davis dataset uses Kd (dissociation constant) in nM.
# Lower Kd = stronger binding. Standard threshold: Kd <= 30 nM = active binder.
BINDING_THRESHOLD_NM = 30.0

DATA_DIR = Path("data")

# DeepDTA repository — reliable academic source for the Davis dataset
BASE_URL = "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis"
DAVIS_FILES = {
    "Y": f"{BASE_URL}/Y",
    "ligands": f"{BASE_URL}/ligands_can.txt",
    "proteins": f"{BASE_URL}/proteins.txt",
}


def _download_file(url: str, filepath: Path) -> None:
    """Download a file if it doesn't already exist."""
    if filepath.exists():
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)
    print(f"    Downloading {filepath.name} ...")
    urllib.request.urlretrieve(url, filepath)


def _load_davis_raw() -> tuple:
    """
    Download and parse the raw Davis dataset files.

    The DeepDTA format stores:
      - ligands_can.txt: JSON dict {drug_name: SMILES}
      - proteins.txt: JSON dict {target_name: amino_acid_sequence}
      - Y: pickle file containing a 2D list of Kd values (drugs x targets)

    Returns:
        drugs: list of SMILES strings
        target_names: list of target identifiers
        target_seqs: list of amino acid sequences
        affinity_matrix: numpy array of Kd values (drugs x targets)
    """
    for name, url in DAVIS_FILES.items():
        _download_file(url, DATA_DIR / name)

    # Load drug SMILES (JSON dict: name -> SMILES)
    with open(DATA_DIR / "ligands", "r") as f:
        ligand_dict = json.load(f)
    drug_names = list(ligand_dict.keys())
    drugs = list(ligand_dict.values())

    # Load protein sequences (JSON dict: name -> sequence)
    with open(DATA_DIR / "proteins", "r") as f:
        protein_dict = json.load(f)
    target_names = list(protein_dict.keys())
    target_seqs = list(protein_dict.values())

    # Load affinity matrix (pickle file containing a 2D list)
    with open(DATA_DIR / "Y", "rb") as f:
        affinity_matrix = np.array(pickle.load(f, encoding="latin1"))

    print(f"    Raw data: {len(drugs)} drugs x {len(target_seqs)} targets")
    print(f"    Affinity matrix shape: {affinity_matrix.shape}")

    return drugs, target_names, target_seqs, affinity_matrix


def load_davis_dataset(threshold: float = BINDING_THRESHOLD_NM) -> dict:
    """
    Load the Davis kinase binding dataset and convert to binary classification.

    Downloads the dataset on first run, then caches locally in data/.

    Args:
        threshold: Kd threshold in nM. Pairs with Kd <= threshold are labeled
                   as binding (1), others as non-binding (0).

    Returns:
        Dictionary with 'train', 'valid', 'test' DataFrames, each containing:
            - Drug: SMILES string
            - Target: amino acid sequence
            - Y: binary label (1 = binds, 0 = does not bind)
    """
    print("  Loading Davis dataset...")
    drugs, target_names, target_seqs, affinity_matrix = _load_davis_raw()

    # Flatten the matrix into drug-target pairs
    rows = []
    for i, drug in enumerate(drugs):
        for j, target_seq in enumerate(target_seqs):
            if i < affinity_matrix.shape[0] and j < affinity_matrix.shape[1]:
                kd = affinity_matrix[i][j]
                label = 1 if kd <= threshold else 0
                rows.append({
                    "Drug": drug,
                    "Target": target_seq,
                    "Y": label,
                })

    df = pd.DataFrame(rows)
    print(f"    Total pairs: {len(df)}")
    print(f"    Binding (Y=1): {df['Y'].sum()}, Non-binding (Y=0): {(df['Y'] == 0).sum()}")

    # Split: 70% train, 15% valid, 15% test
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["Y"]
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["Y"]
    )

    processed = {"train": train_df, "valid": valid_df, "test": test_df}

    for split_name, split_df in processed.items():
        n_pos = split_df["Y"].sum()
        n_neg = len(split_df) - n_pos
        print(f"    {split_name}: {len(split_df)} pairs | {n_pos} binding, {n_neg} non-binding")

    return processed


def get_dataset_stats(data: dict) -> dict:
    """Compute summary statistics for the processed dataset."""
    all_data = pd.concat(data.values())
    return {
        "total_pairs": len(all_data),
        "unique_drugs": all_data["Drug"].nunique(),
        "unique_targets": all_data["Target"].nunique(),
        "binding_ratio": float(all_data["Y"].mean()),
        "train_size": len(data["train"]),
        "valid_size": len(data["valid"]),
        "test_size": len(data["test"]),
    }
