"""
Similar drug search for DrugLens.

Given a query molecule, finds the most structurally similar compounds in the
reference database using Tanimoto similarity on Morgan fingerprints.

Tanimoto similarity (also called Jaccard index for binary vectors) measures
the overlap between two fingerprints:

    T(A, B) = |A ∩ B| / |A ∪ B|

A value of 1.0 means identical fingerprints; 0.0 means no overlap.
In drug discovery, molecules with Tanimoto > 0.7 are generally considered
structurally similar and may share biological activity.
"""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def compute_tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute Tanimoto similarity between two binary fingerprint vectors."""
    intersection = np.sum(np.logical_and(fp1, fp2))
    union = np.sum(np.logical_or(fp1, fp2))
    if union == 0:
        return 0.0
    return float(intersection / union)


def find_similar_drugs(
    query_smiles: str,
    reference_smiles: list[str],
    reference_fingerprints: np.ndarray,
    reference_labels: np.ndarray | None = None,
    top_k: int = 5,
    fp_radius: int = 2,
    fp_nbits: int = 2048,
) -> list[dict]:
    """
    Find the top-K most similar drugs to a query molecule.

    Args:
        query_smiles: SMILES string of the query molecule
        reference_smiles: list of SMILES strings in the reference database
        reference_fingerprints: precomputed Morgan fingerprints for reference
        reference_labels: optional binding labels for reference compounds
        top_k: number of similar drugs to return
        fp_radius: Morgan fingerprint radius (must match reference)
        fp_nbits: Morgan fingerprint bit length (must match reference)

    Returns:
        List of dicts with keys: smiles, similarity, label (if available)
    """
    # Generate fingerprint for query
    mol = Chem.MolFromSmiles(query_smiles)
    if mol is None:
        return []

    query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_nbits)
    query_arr = np.array(query_fp, dtype=np.float32)

    # Compute similarity against all reference compounds
    similarities = np.array([
        compute_tanimoto_similarity(query_arr, ref_fp)
        for ref_fp in reference_fingerprints
    ])

    # Get top-K indices (excluding exact matches with similarity 1.0 if same molecule)
    top_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in top_indices:
        if len(results) >= top_k:
            break
        # Skip if it's the exact same molecule
        if similarities[idx] >= 0.999 and reference_smiles[idx] == query_smiles:
            continue
        entry = {
            "smiles": reference_smiles[idx],
            "similarity": round(float(similarities[idx]), 4),
        }
        if reference_labels is not None:
            entry["binds"] = bool(reference_labels[idx])
        results.append(entry)

    return results


def build_reference_database(
    smiles_list: list[str],
    labels: np.ndarray,
    fp_radius: int = 2,
    fp_nbits: int = 2048,
) -> dict:
    """
    Build a reference database of unique drugs with precomputed fingerprints.

    This is saved as an artifact and loaded by the Streamlit app for fast
    similarity search at prediction time.
    """
    unique_drugs = {}
    fingerprints = []
    smiles_out = []
    labels_out = []

    for smiles, label in zip(smiles_list, labels):
        if smiles in unique_drugs:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_nbits)
        fp_arr = np.array(fp, dtype=np.float32)

        unique_drugs[smiles] = True
        fingerprints.append(fp_arr)
        smiles_out.append(smiles)
        labels_out.append(int(label))

    print(f"  Reference database: {len(smiles_out)} unique drugs")

    return {
        "smiles": smiles_out,
        "fingerprints": np.array(fingerprints),
        "labels": np.array(labels_out),
    }
