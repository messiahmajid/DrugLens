"""
Lightweight tests for DrugLens Screening Studio.

Tests SMILES validation, descriptor computation, priority bucketing,
and batch screening with a small set of molecules.

Run with:  python -m pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np


# ── Chemistry tests ───────────────────────────────────────────────────────

class TestValidateSmiles:
    def test_valid_aspirin(self):
        from src.chemistry import validate_smiles
        assert validate_smiles("CC(=O)OC1=CC=CC=C1C(=O)O") is True

    def test_valid_caffeine(self):
        from src.chemistry import validate_smiles
        assert validate_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C") is True

    def test_invalid_smiles(self):
        from src.chemistry import validate_smiles
        assert validate_smiles("NOT_A_SMILES") is False

    def test_empty_string(self):
        from src.chemistry import validate_smiles
        assert validate_smiles("") is False

    def test_none_input(self):
        from src.chemistry import validate_smiles
        assert validate_smiles(None) is False

    def test_partial_smiles(self):
        from src.chemistry import validate_smiles
        # Partial/broken structure that RDKit cannot parse
        assert validate_smiles("C(C)(") is False

    def test_numeric_input(self):
        from src.chemistry import validate_smiles
        assert validate_smiles(123) is False


class TestCanonicalizeSmiles:
    def test_canonical_form(self):
        from src.chemistry import canonicalize_smiles
        # Different representations of aspirin should canonicalize to the same thing
        c1 = canonicalize_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        c2 = canonicalize_smiles("CC(=O)Oc1ccccc1C(=O)O")
        assert c1 is not None
        assert c1 == c2

    def test_invalid_returns_none(self):
        from src.chemistry import canonicalize_smiles
        assert canonicalize_smiles("INVALID") is None

    def test_empty_returns_none(self):
        from src.chemistry import canonicalize_smiles
        assert canonicalize_smiles("") is None


class TestComputeDescriptors:
    def test_aspirin_descriptors(self):
        from src.chemistry import compute_descriptors
        desc = compute_descriptors("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert desc is not None
        assert "mw" in desc
        assert "logp" in desc
        assert "tpsa" in desc
        assert "hbd" in desc
        assert "hba" in desc
        assert "rotatable_bonds" in desc
        assert "aromatic_rings" in desc
        assert "heavy_atoms" in desc
        # Aspirin MW is ~180
        assert 170 < desc["mw"] < 190

    def test_invalid_returns_none(self):
        from src.chemistry import compute_descriptors
        assert compute_descriptors("NOT_VALID") is None

    def test_empty_returns_none(self):
        from src.chemistry import compute_descriptors
        assert compute_descriptors("") is None


class TestLipinskiViolations:
    def test_aspirin_no_violations(self):
        from src.chemistry import compute_descriptors, count_lipinski_violations
        desc = compute_descriptors("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert count_lipinski_violations(desc) == 0

    def test_empty_dict(self):
        from src.chemistry import count_lipinski_violations
        assert count_lipinski_violations({}) == 0

    def test_none_input(self):
        from src.chemistry import count_lipinski_violations
        assert count_lipinski_violations(None) == 0

    def test_high_mw_violation(self):
        from src.chemistry import count_lipinski_violations
        desc = {"mw": 600, "logp": 2, "hbd": 1, "hba": 3}
        assert count_lipinski_violations(desc) == 1

    def test_all_violations(self):
        from src.chemistry import count_lipinski_violations
        desc = {"mw": 600, "logp": 6, "hbd": 7, "hba": 12}
        assert count_lipinski_violations(desc) == 4


class TestParseSmiles:
    def test_single_smiles(self):
        from src.chemistry import parse_smiles_lines
        valid, invalid = parse_smiles_lines("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert len(valid) == 1
        assert valid[0]["canonical_smiles"] is not None
        assert len(invalid) == 0

    def test_multiple_smiles(self):
        from src.chemistry import parse_smiles_lines
        text = "CC(=O)OC1=CC=CC=C1C(=O)O\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        valid, invalid = parse_smiles_lines(text)
        assert len(valid) == 2
        assert len(invalid) == 0

    def test_named_smiles_tab(self):
        from src.chemistry import parse_smiles_lines
        text = "Aspirin\tCC(=O)OC1=CC=CC=C1C(=O)O"
        valid, invalid = parse_smiles_lines(text)
        assert len(valid) == 1
        assert valid[0]["name"] == "Aspirin"

    def test_empty_input(self):
        from src.chemistry import parse_smiles_lines
        valid, invalid = parse_smiles_lines("")
        assert valid == [] and invalid == []
        valid, invalid = parse_smiles_lines(None)
        assert valid == [] and invalid == []

    def test_comment_lines_skipped(self):
        from src.chemistry import parse_smiles_lines
        text = "# Comment\nCC(=O)OC1=CC=CC=C1C(=O)O"
        valid, invalid = parse_smiles_lines(text)
        assert len(valid) == 1

    def test_invalid_lines_reported(self):
        from src.chemistry import parse_smiles_lines
        text = "CC(=O)OC1=CC=CC=C1C(=O)O\nINVALID_SMILES\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        valid, invalid = parse_smiles_lines(text)
        assert len(valid) == 2
        assert len(invalid) == 1
        assert invalid[0]["reason"] == "Invalid SMILES"
        assert invalid[0]["row"] == 2


class TestDeduplication:
    def test_dedup_removes_duplicates(self):
        from src.chemistry import deduplicate_compounds
        compounds = [
            {"name": "A", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "canonical_smiles": "CC(=O)Oc1ccccc1C(O)=O"},
            {"name": "B", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "canonical_smiles": "CC(=O)Oc1ccccc1C(O)=O"},
        ]
        result = deduplicate_compounds(compounds)
        assert len(result) == 1
        assert result[0]["name"] == "A"

    def test_dedup_keeps_unique(self):
        from src.chemistry import deduplicate_compounds
        compounds = [
            {"name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "canonical_smiles": "CC(=O)Oc1ccccc1C(O)=O"},
            {"name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "canonical_smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O"},
        ]
        result = deduplicate_compounds(compounds)
        assert len(result) == 2


class TestParseCsv:
    def test_valid_csv(self):
        import io
        from src.chemistry import parse_compound_csv
        csv_content = "name,smiles\nAspirin,CC(=O)OC1=CC=CC=C1C(=O)O\nCaffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C\n"
        f = io.StringIO(csv_content)
        valid, invalid = parse_compound_csv(f)
        assert len(valid) == 2
        assert len(invalid) == 0

    def test_csv_with_invalid_rows(self):
        import io
        from src.chemistry import parse_compound_csv
        csv_content = "name,smiles\nAspirin,CC(=O)OC1=CC=CC=C1C(=O)O\nBad,INVALID\n"
        f = io.StringIO(csv_content)
        valid, invalid = parse_compound_csv(f)
        assert len(valid) == 1
        assert len(invalid) == 1

    def test_csv_no_smiles_column(self):
        import io
        from src.chemistry import parse_compound_csv
        csv_content = "name,formula\nAspirin,C9H8O4\n"
        f = io.StringIO(csv_content)
        valid, invalid = parse_compound_csv(f)
        assert len(valid) == 0
        assert len(invalid) == 1
        assert "smiles" in invalid[0]["reason"].lower()


# ── Screening tests ───────────────────────────────────────────────────────

class TestAssignPriority:
    def test_high_priority(self):
        from src.screening import assign_priority
        assert assign_priority(0.75) == "High priority"
        assert assign_priority(0.90) == "High priority"
        assert assign_priority(1.0) == "High priority"

    def test_review(self):
        from src.screening import assign_priority
        assert assign_priority(0.50) == "Review"
        assert assign_priority(0.74) == "Review"

    def test_low_priority(self):
        from src.screening import assign_priority
        assert assign_priority(0.49) == "Low priority"
        assert assign_priority(0.0) == "Low priority"


class TestSummarizeInvalid:
    def test_empty_list(self):
        from src.screening import summarize_invalid_compounds
        result = summarize_invalid_compounds([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_with_rows(self):
        from src.screening import summarize_invalid_compounds
        rows = [{"row": 2, "name": "Bad", "smiles": "XYZ", "reason": "Invalid"}]
        result = summarize_invalid_compounds(rows)
        assert len(result) == 1


# ── Screening integration (requires model artifacts) ──────────────────────

class TestBatchScreening:
    """Integration tests that require model artifacts to exist."""

    @pytest.fixture(autouse=True)
    def check_artifacts(self):
        """Skip if artifacts aren't available."""
        from pathlib import Path
        if not Path("artifacts/model.joblib").exists():
            pytest.skip("Model artifacts not found — run train.py first")

    def test_screen_aspirin_and_caffeine(self):
        from src.model import load_artifacts
        from src.screening import screen_compounds

        model, metrics, ref_db = load_artifacts("artifacts")
        compounds = [
            {"name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
            {"name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
        ]
        target_seq = "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLESEEEGVPSTAIREISLLKELKHDNIVRLYDIVHSDAHKLYLVFEFLDLDLKRYMEGIPKDQPLGADIVKKFMMQLCKGIAYCHSHRILHRDLKPQNLLIDKEGNLKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL"

        df = screen_compounds(compounds, target_seq, model, ref_db)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "Binding Prob" in df.columns
        assert "Priority" in df.columns
        assert "MW" in df.columns
        assert "LogP" in df.columns
        # Results should be sorted by binding probability descending
        probs = df["Binding Prob"].tolist()
        assert probs == sorted(probs, reverse=True)

    def test_screen_with_invalid_compound(self):
        """Invalid compound should be skipped, not crash the batch."""
        from src.model import load_artifacts
        from src.screening import screen_compounds

        model, metrics, ref_db = load_artifacts("artifacts")
        compounds = [
            {"name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
            {"name": "Bad", "smiles": "INVALID_SMILES"},
            {"name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
        ]
        target_seq = "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLESEEEGVPSTAIREISLLKELKHDNIVRLYDIVHSDAHKLYLVFEFLDLDLKRYMEGIPKDQPLGADIVKKFMMQLCKGIAYCHSHRILHRDLKPQNLLIDKEGNLKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL"

        df = screen_compounds(compounds, target_seq, model, ref_db)
        assert len(df) == 2  # Only 2 valid compounds
        names = df["Name"].tolist()
        assert "Bad" not in names

    def test_screen_empty_list(self):
        from src.model import load_artifacts
        from src.screening import screen_compounds

        model, metrics, ref_db = load_artifacts("artifacts")
        df = screen_compounds([], "ACGT", model, ref_db)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_progress_callback(self):
        """Verify progress callback is called correctly."""
        from src.model import load_artifacts
        from src.screening import screen_compounds

        model, metrics, ref_db = load_artifacts("artifacts")
        compounds = [
            {"name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
        ]
        target_seq = "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLESEEEGVPSTAIREISLLKELKHDNIVRLYDIVHSDAHKLYLVFEFLDLDLKRYMEGIPKDQPLGADIVKKFMMQLCKGIAYCHSHRILHRDLKPQNLLIDKEGNLKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL"

        progress_calls = []
        df = screen_compounds(
            compounds, target_seq, model, ref_db,
            progress_callback=lambda c, t: progress_calls.append((c, t)),
        )
        assert len(progress_calls) == 1
        assert progress_calls[0] == (1, 1)


# ── SMILES Image tests ───────────────────────────────────────────────────

class TestSmilesToImage:
    def test_valid_smiles_image(self):
        from src.chemistry import smiles_to_image
        from PIL import Image
        img = smiles_to_image("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert isinstance(img, Image.Image)

    def test_invalid_smiles_none(self):
        from src.chemistry import smiles_to_image
        assert smiles_to_image("INVALID") is None

    def test_empty_smiles_none(self):
        from src.chemistry import smiles_to_image
        assert smiles_to_image("") is None
