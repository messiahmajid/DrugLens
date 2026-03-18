"""
Feature engineering for DrugLens.

Converts raw SMILES strings and amino acid sequences into numerical feature
vectors suitable for machine learning.

Drug features:
    - Morgan fingerprints (2048-bit circular fingerprints capturing molecular
      substructure patterns within a given radius)
    - Physicochemical descriptors (molecular weight, LogP, H-bond donors/
      acceptors, topological polar surface area, rotatable bonds, aromatic rings)

Protein features:
    - Amino acid composition (frequency of each of 20 standard amino acids)
    - Dipeptide composition (frequency of each possible pair of adjacent amino
      acids — captures local sequence patterns)
    - Bulk sequence properties (length, molecular weight estimate, charge metrics)
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from itertools import product as iter_product


# ─── Constants ─────────────────────────────────────────────────────────────

MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# All 400 possible dipeptides
DIPEPTIDES = [a + b for a, b in iter_product(AMINO_ACIDS, repeat=2)]


# ─── Drug Featurization ───────────────────────────────────────────────────

def smiles_to_fingerprint(smiles: str) -> np.ndarray | None:
    """
    Convert a SMILES string to a Morgan fingerprint bit vector.

    Morgan fingerprints encode circular substructures around each atom up to
    a given radius. They are the most widely used molecular representation
    in cheminformatics and drug discovery ML.

    Returns None if the SMILES string is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
    return np.array(fp, dtype=np.float32)


def smiles_to_descriptors(smiles: str) -> np.ndarray | None:
    """
    Compute physicochemical descriptors for a molecule.

    These capture global molecular properties that influence binding:
    - Molecular weight: larger molecules have more potential interaction points
    - LogP: lipophilicity affects membrane permeability and binding pockets
    - H-bond donors/acceptors: key for specific protein-ligand interactions
    - TPSA: topological polar surface area, relates to bioavailability
    - Rotatable bonds: molecular flexibility, affects binding entropy
    - Aromatic rings: pi-stacking interactions with protein residues

    Returns None if the SMILES string is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.NumValenceElectrons(mol),
    ], dtype=np.float32)


DESCRIPTOR_NAMES = [
    "MolWeight", "LogP", "HBondDonors", "HBondAcceptors", "TPSA",
    "RotatableBonds", "AromaticRings", "HeavyAtoms", "FractionCSP3",
    "ValenceElectrons",
]


def featurize_drug(smiles: str) -> np.ndarray | None:
    """
    Full drug featurization: Morgan fingerprint + physicochemical descriptors.
    Returns a single concatenated feature vector, or None if SMILES is invalid.
    """
    fp = smiles_to_fingerprint(smiles)
    desc = smiles_to_descriptors(smiles)
    if fp is None or desc is None:
        return None
    return np.concatenate([fp, desc])


def get_drug_feature_names() -> list[str]:
    """Return human-readable names for all drug features."""
    fp_names = [f"MorganBit_{i}" for i in range(MORGAN_NBITS)]
    return fp_names + DESCRIPTOR_NAMES


# ─── Protein Featurization ─────────────────────────────────────────────────

def sequence_to_aac(sequence: str) -> np.ndarray:
    """
    Compute amino acid composition (AAC) — the frequency of each standard
    amino acid in the sequence. This is a simple but effective protein
    representation that captures bulk compositional properties.
    """
    seq = sequence.upper()
    length = max(len(seq), 1)
    return np.array([seq.count(aa) / length for aa in AMINO_ACIDS], dtype=np.float32)


def sequence_to_dipeptide(sequence: str) -> np.ndarray:
    """
    Compute dipeptide composition — the frequency of each pair of adjacent
    amino acids. This captures local sequence patterns and is more expressive
    than single amino acid composition alone. Produces 400 features (20x20).
    """
    seq = sequence.upper()
    total = max(len(seq) - 1, 1)
    dipeptide_counts = {}
    for i in range(len(seq) - 1):
        dp = seq[i:i+2]
        dipeptide_counts[dp] = dipeptide_counts.get(dp, 0) + 1
    return np.array(
        [dipeptide_counts.get(dp, 0) / total for dp in DIPEPTIDES],
        dtype=np.float32
    )


def sequence_to_properties(sequence: str) -> np.ndarray:
    """
    Compute bulk sequence properties: length, estimated molecular weight,
    fraction of charged/hydrophobic/polar residues.
    """
    seq = sequence.upper()
    length = max(len(seq), 1)

    # Amino acid property groups
    charged = set("DEKRH")
    hydrophobic = set("AVILMFYW")
    polar = set("STNQ")

    return np.array([
        length,
        length * 110.0,  # Average residue MW ~110 Da
        sum(1 for aa in seq if aa in charged) / length,
        sum(1 for aa in seq if aa in hydrophobic) / length,
        sum(1 for aa in seq if aa in polar) / length,
    ], dtype=np.float32)


PROTEIN_PROPERTY_NAMES = [
    "SeqLength", "EstMolWeight", "FracCharged", "FracHydrophobic", "FracPolar"
]


def featurize_protein(sequence: str) -> np.ndarray:
    """
    Full protein featurization: AAC + dipeptide composition + bulk properties.
    Returns a single concatenated feature vector.
    """
    aac = sequence_to_aac(sequence)
    dipeptide = sequence_to_dipeptide(sequence)
    props = sequence_to_properties(sequence)
    return np.concatenate([aac, dipeptide, props])


def get_protein_feature_names() -> list[str]:
    """Return human-readable names for all protein features."""
    aac_names = [f"AA_{aa}" for aa in AMINO_ACIDS]
    dp_names = [f"DP_{dp}" for dp in DIPEPTIDES]
    return aac_names + dp_names + PROTEIN_PROPERTY_NAMES


# ─── Combined Featurization ───────────────────────────────────────────────

def featurize_pair(smiles: str, sequence: str) -> np.ndarray | None:
    """
    Featurize a drug-target pair by concatenating drug and protein features.
    Returns None if the SMILES string is invalid.
    """
    drug_feat = featurize_drug(smiles)
    if drug_feat is None:
        return None
    protein_feat = featurize_protein(sequence)
    return np.concatenate([drug_feat, protein_feat])


def get_all_feature_names() -> list[str]:
    """Return names for all features in the combined vector."""
    return get_drug_feature_names() + get_protein_feature_names()


def featurize_dataset(df: pd.DataFrame, show_progress: bool = True) -> tuple:
    """
    Featurize an entire DataFrame of drug-target pairs.

    Args:
        df: DataFrame with 'Drug' (SMILES), 'Target' (sequence), 'Y' (label)
        show_progress: whether to print progress updates

    Returns:
        X: feature matrix (n_valid_samples, n_features)
        y: label vector (n_valid_samples,)
        valid_idx: indices of successfully featurized rows
    """
    X_list = []
    y_list = []
    valid_idx = []
    total = len(df)
    failed = 0

    for i, (_, row) in enumerate(df.iterrows()):
        if show_progress and (i + 1) % 2000 == 0:
            print(f"    Featurized {i+1}/{total} pairs...")

        feat = featurize_pair(row["Drug"], row["Target"])
        if feat is not None:
            X_list.append(feat)
            y_list.append(row["Y"])
            valid_idx.append(i)
        else:
            failed += 1

    if show_progress:
        print(f"    Done. {len(X_list)} valid, {failed} failed (invalid SMILES).")

    return np.array(X_list), np.array(y_list), valid_idx
