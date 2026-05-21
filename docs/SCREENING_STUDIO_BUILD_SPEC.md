# DrugLens Screening Studio Build Spec

## Goal

Rebuild DrugLens from a single-prediction demo into a practical compound screening workflow.

The product should help a user answer:

> Given a target protein and a set of candidate compounds, which molecules should I investigate first, why, and what basic chemistry risks should I notice?

This is not a clinical tool and should not claim to discover drugs. It is an early-stage, interpretable compound prioritization tool focused on kinase-style drug-target screening.

## Product Positioning

**Name:** DrugLens Screening Studio

**One-line story:**
Interpretable compound screening for early-stage kinase drug discovery.

**Primary workflow:**

1. Select a protein target.
2. Select example compounds or upload a CSV/SMILES list.
3. Run batch screening.
4. View ranked candidate compounds.
5. Filter by predicted binding score and chemistry properties.
6. Inspect individual compound evidence.
7. Export screening results.

## MVP Scope

Build this first inside the current Streamlit app unless a rewrite is explicitly requested later. The MVP should be robust and useful before switching stacks.

### Keep From Existing App

- RDKit molecule validation and rendering.
- `featurize_pair` from `src/features.py`.
- `predict_binding` and artifact loading from `src/model.py`.
- SHAP explanation helpers from `src/explainability.py`.
- Tanimoto similarity helpers from `src/similarity.py`.
- Existing example targets and example drugs.

### Add In MVP

1. Batch compound input:
   - Built-in example library.
   - Manual multi-line SMILES input.
   - CSV upload with at least a `smiles` column and optional `name` column.

2. Compound standardization and validation:
   - Validate SMILES.
   - Canonicalize valid molecules.
   - Drop duplicates.
   - Report invalid rows clearly.

3. Chemistry descriptors:
   - Molecular weight.
   - LogP.
   - TPSA.
   - H-bond donors.
   - H-bond acceptors.
   - Rotatable bonds.
   - Aromatic rings.
   - Heavy atoms.
   - Lipinski rule-of-five violation count.

4. Batch prediction:
   - Predict binding probability for each valid compound against the selected target.
   - Rank compounds descending by binding probability.
   - Add a readable bucket:
     - `High priority` >= 0.75
     - `Review` >= 0.50 and < 0.75
     - `Low priority` < 0.50

5. Ranked results table:
   - Compound name.
   - Canonical SMILES.
   - Binding probability.
   - Priority bucket.
   - Lipinski violations.
   - Molecular weight.
   - LogP.
   - TPSA.
   - Similar known compound score if available.

6. Filters:
   - Minimum binding probability.
   - Maximum Lipinski violations.
   - Molecular weight range.
   - LogP range.

7. Compound detail view:
   - Molecule image.
   - Prediction score and priority bucket.
   - Chemistry descriptor summary.
   - Similar known compounds.
   - SHAP explanation for selected compound.
   - Plain-language caveat that results are model-based prioritization only.

8. Export:
   - Download ranked results as CSV.
   - Include model AUROC/F1, dataset stats, target name, and timestamp in the exported table metadata if feasible.

## UX Direction

The app should feel like a scientific screening workspace, not a landing page.

Recommended layout:

- Sidebar:
  - Target selection.
  - Compound source controls.
  - Filters.
  - Dataset/model metrics.

- Main area:
  - Title: `DrugLens Screening Studio`.
  - Compact intro sentence, not a marketing hero.
  - Batch input area.
  - Ranked screening results table.
  - Selected compound evidence panel.

Avoid large decorative hero sections. Prioritize dense, readable tables and clear analysis states.

## Suggested File-Level Changes

### New: `src/chemistry.py`

Add chemistry utility functions:

- `canonicalize_smiles(smiles: str) -> str | None`
- `validate_smiles(smiles: str) -> bool`
- `smiles_to_image(smiles: str, size=(300, 220))`
- `compute_descriptors(smiles: str) -> dict | None`
- `count_lipinski_violations(descriptors: dict) -> int`
- `parse_smiles_lines(text: str) -> list[dict]`
- `parse_compound_csv(file) -> tuple[list[dict], list[dict]]`

Expected compound dict:

```python
{
    "name": "Aspirin",
    "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "canonical_smiles": "...",
}
```

### New: `src/screening.py`

Add batch screening logic:

- `screen_compounds(compounds, target_sequence, model, ref_db, feature_names=None) -> pd.DataFrame`
- `assign_priority(binding_probability: float) -> str`
- `summarize_invalid_compounds(invalid_rows) -> pd.DataFrame`

The screening function should:

- Skip invalid compounds.
- Featurize each valid compound.
- Predict binding probability.
- Compute descriptors.
- Compute similar known compound max score.
- Return a ranked DataFrame.

### Update: `app.py`

Refactor from single-prediction flow to screening workflow.

Expected top-level sections:

- Load artifacts.
- Sidebar target and filters.
- Compound input section.
- Run screening button.
- Ranked results.
- Selected compound details.
- Export CSV.

Keep CSS smaller and less fragile than the current version.

### Optional New: `tests/`

Add lightweight tests if the repo has a test runner or if installing pytest is already supported:

- SMILES validation.
- Descriptor computation.
- Priority bucket logic.
- Batch screening with 2-3 example molecules.

Do not block MVP completion on tests if environment setup is painful, but write testable functions.

## Data Plan

For this first build, use the existing saved Davis-based artifacts.

Later, add:

1. TDC Davis rebuild.
2. TDC KIBA model.
3. Affinity regression instead of binary-only classification.
4. Cold-drug, cold-target, and scaffold-split evaluation.
5. Curated BindingDB kinase subset.

Do not attempt BindingDB/ChEMBL integration in the MVP.

## Robustness Requirements

- Invalid SMILES should not crash the app.
- Empty upload/input should show a useful message.
- Duplicate molecules should be removed after canonicalization.
- Batch prediction should fail per-row, not crash the whole run.
- Export should reflect the filtered or full ranked table clearly.
- The app must clearly state that predictions are computational prioritization, not experimental validation or medical advice.

## Acceptance Criteria

The MVP is done when:

1. A user can select a target.
2. A user can provide multiple compounds through examples, text, or CSV.
3. The app returns a ranked screening table.
4. The user can filter the table.
5. The user can select one compound and inspect:
   - molecule image
   - binding score
   - priority bucket
   - descriptors
   - similar known compounds
   - SHAP explanation
6. Invalid compounds are reported without crashing.
7. Results can be downloaded as CSV.
8. The UI no longer feels like a single prediction demo.

## Suggested Claude CLI Prompt

Use this prompt from the repository root:

```text
You are working in this repo: DrugLens.

Read docs/SCREENING_STUDIO_BUILD_SPEC.md first. Rebuild the current Streamlit app into the MVP described there. Keep the existing model artifacts and ML helpers. Do not attempt a full FastAPI/React rewrite yet.

Implement:
- src/chemistry.py
- src/screening.py
- app.py refactor around batch screening
- lightweight tests if practical

Respect existing user changes. Do not delete unrelated files. Make the app robust against invalid SMILES, empty input, duplicate compounds, and row-level prediction failures.

When finished, run the app or at least run import/compile checks and summarize changed files, verification performed, and any limitations.
```

