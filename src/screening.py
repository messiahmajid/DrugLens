"""
Batch screening engine for DrugLens Screening Studio.

Orchestrates compound screening against a protein target: featurization,
prediction, descriptor computation, similarity lookup — all with per-row
error isolation so a single bad compound never crashes the whole run.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.features import featurize_pair, get_all_feature_names
from src.model import predict_binding
from src.similarity import find_similar_drugs
from src.chemistry import (
    canonicalize_smiles,
    compute_descriptors,
    count_lipinski_violations,
)

logger = logging.getLogger(__name__)


# ── Priority bucketing ─────────────────────────────────────────────────────

def assign_priority(binding_probability: float) -> str:
    """Map a binding probability to a human-readable priority bucket.

    - ``"High priority"`` for prob ≥ 0.75
    - ``"Review"``        for 0.50 ≤ prob < 0.75
    - ``"Low priority"``  for prob < 0.50
    """
    if binding_probability >= 0.75:
        return "High priority"
    elif binding_probability >= 0.50:
        return "Review"
    else:
        return "Low priority"


# ── Trust assessment ──────────────────────────────────────────────────────

def assess_trust(
    binding_prob: float,
    is_kinase: bool,
    similar_score: float | None,
    lipinski_violations: int | None,
) -> tuple[str, str]:
    """Compute a trust level and human-readable reason for a prediction.

    Returns (level, reason) where level is "High", "Medium", or "Low".
    """
    factors_good: list[str] = []
    factors_bad: list[str] = []

    if is_kinase:
        factors_good.append("kinase target (in-domain)")
    else:
        factors_bad.append("out-of-domain target")

    score_margin = abs(binding_prob - 0.5)
    if score_margin >= 0.25:
        factors_good.append("strong score margin")
    elif score_margin < 0.1:
        factors_bad.append("score near decision threshold")

    if similar_score is not None and similar_score >= 0.5:
        factors_good.append("similar known compound found")
    elif similar_score is not None and similar_score >= 0.3:
        pass  # neutral
    else:
        factors_bad.append("no close known compound in reference set")

    if lipinski_violations is not None and lipinski_violations == 0:
        factors_good.append("no Lipinski violations")
    elif lipinski_violations is not None and lipinski_violations >= 2:
        factors_bad.append(f"{lipinski_violations} Lipinski violations")

    if len(factors_bad) == 0 and len(factors_good) >= 3:
        level = "High"
    elif len(factors_bad) >= 2 or (not is_kinase and len(factors_good) < 2):
        level = "Low"
    else:
        level = "Medium"

    parts = factors_good + factors_bad
    reason = ", ".join(parts) + "." if parts else "Insufficient data."
    return level, reason


# ── Single-compound screening ──────────────────────────────────────────────

def screen_single(
    compound: dict,
    target_sequence: str,
    model,
    ref_db: dict,
    feature_names: Optional[list] = None,
    is_kinase: bool = True,
) -> Optional[dict]:
    smiles = compound.get("smiles", "")
    name = compound.get("name", "Unknown")
    canon = compound.get("canonical_smiles") or canonicalize_smiles(smiles)

    if canon is None:
        return None

    # Featurize
    features = featurize_pair(canon, target_sequence)
    if features is None:
        logger.warning("Featurization failed for %s (%s)", name, canon)
        return None

    # Predict
    prediction, confidence = predict_binding(model, features)
    binding_prob = float(confidence)

    # Descriptors
    desc = compute_descriptors(canon) or {}
    lipinski = count_lipinski_violations(desc) if desc else None

    # Similar known compounds (top-1)
    similar_score = None
    nearest_smiles = None
    nearest_label = None
    try:
        similar = find_similar_drugs(
            query_smiles=canon,
            reference_smiles=ref_db.get("smiles", []),
            reference_fingerprints=ref_db.get("fingerprints", []),
            reference_labels=ref_db.get("labels", []),
            top_k=1,
        )
        if similar:
            similar_score = round(similar[0]["similarity"], 3)
            nearest_smiles = similar[0]["smiles"]
            nearest_label = "Binder" if similar[0].get("binds") else "Non-binder"
    except Exception:
        logger.debug("Similarity lookup failed for %s", name, exc_info=True)

    trust_level, trust_reason = assess_trust(
        binding_prob, is_kinase, similar_score, lipinski,
    )

    return {
        "Name": name,
        "SMILES": canon,
        "Binding Prob": round(binding_prob, 4),
        "Priority": assign_priority(binding_prob),
        "Trust": trust_level,
        "Trust Reason": trust_reason,
        "Lipinski Violations": lipinski,
        "MW": desc.get("mw"),
        "LogP": desc.get("logp"),
        "TPSA": desc.get("tpsa"),
        "HBD": desc.get("hbd"),
        "HBA": desc.get("hba"),
        "Rotatable Bonds": desc.get("rotatable_bonds"),
        "Aromatic Rings": desc.get("aromatic_rings"),
        "Heavy Atoms": desc.get("heavy_atoms"),
        "Nearest Known Compound": nearest_smiles,
        "Nearest Known Label": nearest_label,
        "Similar Known Score": similar_score,
    }


# ── Batch screening ───────────────────────────────────────────────────────

def screen_compounds(
    compounds: list[dict],
    target_sequence: str,
    model,
    ref_db: dict,
    feature_names: Optional[list] = None,
    progress_callback=None,
    is_kinase: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = []
    total = len(compounds)

    for idx, compound in enumerate(compounds):
        try:
            row = screen_single(compound, target_sequence, model, ref_db, feature_names, is_kinase=is_kinase)
            if row is not None:
                rows.append(row)
        except Exception:
            name = compound.get("name", f"Compound_{idx + 1}")
            logger.warning("Screening failed for %s", name, exc_info=True)

        if progress_callback:
            progress_callback(idx + 1, total)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("Binding Prob", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-based rank
    df.index.name = "Rank"
    return df


# ── Invalid compound summary ──────────────────────────────────────────────

def summarize_invalid_compounds(invalid_rows: list[dict]) -> pd.DataFrame:
    """Format a list of invalid compound dicts into a summary DataFrame."""
    if not invalid_rows:
        return pd.DataFrame()
    return pd.DataFrame(invalid_rows)
