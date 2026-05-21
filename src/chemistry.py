"""
Chemistry utilities for DrugLens Screening Studio.

Provides SMILES validation, canonicalization, molecular descriptor computation,
Lipinski rule-of-five checking, and compound input parsing (text / CSV).
"""

import io
import csv
from typing import Optional

import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, Lipinski, rdMolDescriptors


# ── SMILES helpers ─────────────────────────────────────────────────────────

def validate_smiles(smiles: str) -> bool:
    """Return True if *smiles* can be parsed by RDKit."""
    if not smiles or not isinstance(smiles, str):
        return False
    return Chem.MolFromSmiles(smiles.strip()) is not None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Return canonical SMILES or None if input is invalid."""
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def smiles_to_image(smiles: str, size: tuple = (300, 220)) -> Optional[Image.Image]:
    """Render a SMILES string as a PIL Image."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


# ── Molecular descriptors ─────────────────────────────────────────────────

def compute_descriptors(smiles: str) -> Optional[dict]:
    """Compute basic chemistry descriptors for a valid SMILES.

    Returns a dict with keys:
        mw, logp, tpsa, hbd, hba, rotatable_bonds, aromatic_rings, heavy_atoms
    or None if SMILES is invalid.
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return {
        "mw": round(Descriptors.MolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "aromatic_rings": Descriptors.NumAromaticRings(mol),
        "heavy_atoms": Lipinski.HeavyAtomCount(mol),
    }


def count_lipinski_violations(descriptors: dict) -> int:
    """Count Lipinski rule-of-five violations.

    Rules: MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10.
    Returns the number of rules violated (0–4).
    """
    if not descriptors:
        return 0
    violations = 0
    if descriptors.get("mw", 0) > 500:
        violations += 1
    if descriptors.get("logp", 0) > 5:
        violations += 1
    if descriptors.get("hbd", 0) > 5:
        violations += 1
    if descriptors.get("hba", 0) > 10:
        violations += 1
    return violations


# ── Input parsing ──────────────────────────────────────────────────────────

def parse_smiles_lines(text: str) -> tuple[list[dict], list[dict]]:
    """Parse multi-line text into compound dicts.

    Accepted formats per line:
        SMILES
        name SMILES       (whitespace-separated, name first)
        name<tab>SMILES

    Returns ``(valid_compounds, invalid_rows)`` where each item is a list of
    dicts.  Invalid or unparseable lines are collected with a reason.
    """
    if not text or not text.strip():
        return [], []

    compounds = []
    invalid: list[dict] = []
    for i, raw_line in enumerate(text.strip().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Try tab-split first, then whitespace
        parts = line.split("\t") if "\t" in line else line.split(None, 1)
        if len(parts) == 2:
            name, smi = parts[0].strip(), parts[1].strip()
            # If the first part is a valid SMILES itself, treat whole line as one SMILES
            if validate_smiles(name) and not validate_smiles(smi):
                name, smi = f"Compound_{i}", line.strip()
            elif not validate_smiles(smi):
                # Maybe they're reversed
                if validate_smiles(name):
                    name, smi = smi, name
                else:
                    invalid.append({"row": i, "name": parts[0].strip(), "smiles": parts[1].strip(), "reason": "Invalid SMILES"})
                    continue
        elif len(parts) == 1:
            name, smi = f"Compound_{i}", parts[0].strip()
        else:
            continue

        canon = canonicalize_smiles(smi)
        if canon is None:
            invalid.append({"row": i, "name": name, "smiles": smi, "reason": "Invalid SMILES"})
            continue

        compounds.append({
            "name": name,
            "smiles": smi,
            "canonical_smiles": canon,
        })
    return compounds, invalid


def parse_compound_csv(file) -> tuple[list[dict], list[dict]]:
    """Parse a CSV file (or file-like / UploadedFile) with compound data.

    The CSV must have a ``smiles`` column. An optional ``name`` column provides
    compound names. All other columns are ignored.

    Returns ``(valid_compounds, invalid_rows)`` where each item is a list of
    dicts.
    """
    valid: list[dict] = []
    invalid: list[dict] = []

    try:
        # Streamlit UploadedFile → bytes
        if hasattr(file, "read"):
            raw = file.read()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            file.seek(0)  # reset for potential re-reads
        else:
            raw = str(file)

        reader = csv.DictReader(io.StringIO(raw))
        # Normalise header names to lowercase
        if reader.fieldnames is None:
            return [], []
        reader.fieldnames = [f.strip().lower() for f in reader.fieldnames]

        if "smiles" not in reader.fieldnames:
            return [], [{"row": 0, "reason": "CSV has no 'smiles' column"}]

        for i, row in enumerate(reader, start=2):  # header is row 1
            smi = (row.get("smiles") or "").strip()
            name = (row.get("name") or f"CSV_{i}").strip()

            if not smi:
                invalid.append({"row": i, "name": name, "smiles": "", "reason": "Empty SMILES"})
                continue

            canon = canonicalize_smiles(smi)
            if canon is None:
                invalid.append({"row": i, "name": name, "smiles": smi, "reason": "Invalid SMILES"})
                continue

            valid.append({
                "name": name,
                "smiles": smi,
                "canonical_smiles": canon,
            })
    except Exception as exc:
        invalid.append({"row": 0, "name": "", "smiles": "", "reason": f"CSV parse error: {exc}"})

    return valid, invalid


def deduplicate_compounds(compounds: list[dict]) -> list[dict]:
    """Remove duplicate compounds by canonical SMILES, keeping the first occurrence."""
    seen: set[str] = set()
    unique: list[dict] = []
    for comp in compounds:
        canon = comp.get("canonical_smiles") or canonicalize_smiles(comp.get("smiles", ""))
        if canon is None:
            continue
        if canon not in seen:
            seen.add(canon)
            unique.append(comp)
    return unique
