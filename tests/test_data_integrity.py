"""
Data integrity tests for DrugLens.

Verifies that target and ligand data files are consistent with what the
app exposes, preventing the class of bugs where hardcoded sequences
diverge from the actual Davis dataset.

Run with:  python -m pytest tests/test_data_integrity.py -v
"""

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def davis_proteins():
    with open("data/proteins") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def davis_ligands():
    with open("data/ligands") as f:
        return json.load(f)


class TestFeaturedTargets:
    def test_all_featured_keys_exist_in_davis(self, davis_proteins):
        from app import _KINASE_TARGET_META

        for display_name, (davis_key, _desc) in _KINASE_TARGET_META.items():
            assert davis_key in davis_proteins, (
                f"Featured target '{display_name}' maps to key '{davis_key}' "
                f"which is missing from data/proteins"
            )

    def test_featured_sequences_match_davis(self, davis_proteins):
        from app import FEATURED_KINASE_TARGETS, _KINASE_TARGET_META

        for display_name, (davis_key, _) in _KINASE_TARGET_META.items():
            if display_name not in FEATURED_KINASE_TARGETS:
                continue
            app_seq = FEATURED_KINASE_TARGETS[display_name]["sequence"]
            davis_seq = davis_proteins[davis_key]
            assert app_seq == davis_seq, (
                f"Sequence mismatch for '{display_name}' (key={davis_key}): "
                f"app has {len(app_seq)} residues, Davis has {len(davis_seq)}"
            )

    def test_featured_targets_marked_in_training(self):
        from app import FEATURED_KINASE_TARGETS

        for name, info in FEATURED_KINASE_TARGETS.items():
            assert info["in_training"] is True, f"{name} should be in_training=True"


class TestAllDavisTargets:
    def test_count_matches_proteins_file(self, davis_proteins):
        from app import ALL_DAVIS_TARGETS

        assert len(ALL_DAVIS_TARGETS) == len(davis_proteins), (
            f"ALL_DAVIS_TARGETS has {len(ALL_DAVIS_TARGETS)} entries "
            f"but data/proteins has {len(davis_proteins)}"
        )

    def test_all_sequences_match_davis(self, davis_proteins):
        from app import ALL_DAVIS_TARGETS

        for name, info in ALL_DAVIS_TARGETS.items():
            davis_key = name
            for _display, (_key, _) in __import__("app")._KINASE_TARGET_META.items():
                if _display == name:
                    davis_key = _key
                    break
            assert davis_key in davis_proteins, f"Key '{davis_key}' not in data/proteins"
            assert info["sequence"] == davis_proteins[davis_key]


class TestDavisLigands:
    def test_ligands_file_exists(self):
        assert Path("data/ligands").exists()

    def test_ligand_count(self, davis_ligands):
        from app import load_davis_compounds

        loaded = load_davis_compounds()
        assert len(loaded) == len(davis_ligands), (
            f"load_davis_compounds returned {len(loaded)} but "
            f"data/ligands has {len(davis_ligands)}"
        )

    def test_ligands_are_nonempty_smiles(self, davis_ligands):
        for cid, smiles in davis_ligands.items():
            assert isinstance(smiles, str) and len(smiles) > 0, (
                f"Ligand {cid} has invalid SMILES"
            )


class TestOutOfDomainTargets:
    def test_marked_not_in_training(self):
        from app import OUT_OF_DOMAIN_TARGETS

        for name, info in OUT_OF_DOMAIN_TARGETS.items():
            assert info["in_training"] is False, f"{name} should be in_training=False"
