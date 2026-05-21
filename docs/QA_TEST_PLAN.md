# DrugLens Screening Studio QA Test Plan

Use this checklist to verify the MVP manually.

## Start The App

```bash
streamlit run app.py
```

Expected:

- The app opens without crashing.
- Sidebar shows demo preset selector, target selection, filters, model performance, and dataset stats.
- Main page title is `DrugLens Screening Studio`.

## Test 1: Example Library

1. Set target group to `Featured kinases`.
2. Select target: `CDK2 — cell cycle regulation`.
3. Keep compound source as `Example library` → `Curated examples`.
4. Leave the default selected compounds.
5. Click `Screen ... compounds against CDK2`.

Expected:

- A progress bar appears and completes.
- A ranked results table appears.
- Each row has:
  - `Name`
  - `SMILES`
  - `Binding Prob`
  - `Priority`
  - `Trust`
  - `Lipinski Violations`
  - `MW`
  - `LogP`
  - `TPSA`
  - `Nearest Known Compound`
  - `Nearest Known Label`
  - `Similar Known Score`
- `Binding Prob` values are between `0.0` and `1.0`.
- Selecting a compound shows molecule image, score, descriptors, similar compounds, and SHAP explanation.

## Test 2: CSV Upload

1. Select compound source: `Upload CSV`.
2. Upload `examples/sample_compounds.csv`.
3. Click the screen button.

Expected:

- The app reports valid compounds.
- It reports invalid rows for:
  - `Invalid Molecule`
  - `Empty Molecule`
- It reports one duplicate removed for `Duplicate Aspirin`.
- The results table screens only valid unique molecules.
- The app does not crash.

## Test 3: Pasted SMILES

Select compound source: `Paste SMILES` and paste:

```text
Aspirin	CC(=O)OC1=CC=CC=C1C(=O)O
Caffeine	CN1C=NC2=C1C(=O)N(C(=O)N2C)C
BadOne	NOT_A_SMILES
Metformin	CN(C)C(=N)NC(=N)N
```

Expected:

- Three valid compounds are found.
- One invalid row is reported.
- Screening works for the three valid compounds.

## Test 1b: All Davis Targets

1. Set target group to `All Davis targets`.
2. Type `MAPK` in the dropdown to search.
3. Select any matched target.
4. Screen curated examples against it.

Expected:

- Dropdown shows all 442 Davis targets (featured first, then wild-type, then mutants).
- Typing filters the list.
- Mutant variants (e.g., `BRAF(V600E)`) appear separately from wild-type.
- Screening works and results include `In training set: Yes`.

## Test 1c: Davis Training Ligands

1. Set compound source to `Example library` → `Davis training ligands`.
2. Leave defaults (25 selected).
3. Screen against any featured kinase.

Expected:

- 68 Davis ligands are available (labeled `Davis <CID>`).
- Screening works normally.

## Test 1d: Batch Size Warning & Hard Cap

1. Select `Davis training ligands` and select all 68.

Expected:

- A warning appears: "68 compounds selected — screening may take a while."

To test the hard cap (500 compounds): upload a large CSV exceeding 500 rows.

Expected:

- An error message says only the first 500 will be screened.
- Screening proceeds with 500 compounds.

## Test 1e: Demo Presets

1. In the sidebar, select preset: `Quick demo — 10 curated vs CDK2`.

Expected:

- Target group switches to `Featured kinases`.
- Target switches to `CDK2 — cell cycle regulation`.
- Example mode switches to `Curated examples`.

2. Select preset: `In-domain batch — Davis ligands vs EGFR`.

Expected:

- Target switches to `EGFR — non-small cell lung cancer`.
- Example mode switches to `Davis training ligands`.

3. Select preset: `Out-of-domain — 10 curated vs ACE2`.

Expected:

- Target group switches to `Out-of-domain examples`.
- Target switches to `ACE2 — SARS-CoV-2 entry receptor`.
- Out-of-domain warning appears.

4. Select preset: `Custom`.

Expected:

- All controls revert to manual selection.

## Test 1f: Nearest Known Compound

After screening any compounds:

Expected:

- Results table includes `Nearest Known Compound` (SMILES of closest reference molecule).
- Results table includes `Nearest Known Label` (`Binder` or `Non-binder`).
- Compound detail view shows similar compounds with SMILES and binding status.

## Test 4: Stale Results

1. Screen the example library against `CDK2`.
2. Change target to `EGFR`.

Expected:

- Old results disappear until you screen again.

Then:

1. Screen the example library.
2. Remove one selected example compound.

Expected:

- Old results disappear until you screen again.

## Test 5: Filters

After screening:

1. Raise `Min binding probability`.
2. Lower `Max Lipinski violations`.
3. Adjust molecular weight range.
4. Adjust LogP range.

Expected:

- `After filters` count changes.
- Table updates.
- Detail selectbox only shows filtered compounds.
- If no compounds match, app shows a clear warning.

## Test 6: Export

After screening:

1. Click `Download results as CSV`.
2. Open the downloaded file.

Expected:

- File contains comment metadata lines:
  - target
  - date
  - model metrics
  - dataset source
  - computational-prioritization caveat
- File contains the filtered ranked results.

3. Click `Download HTML report`.
4. Open the downloaded `.html` file in a browser.

Expected:

- Report table includes `Trust`, `Trust Reason`, `Nearest Known Compound`, `Nearest Known Label`, and `Similar Known Score` columns.
- Report is self-contained and readable without the app.

## Test 7: Non-Kinase Warning

1. Set target group to `Out-of-domain examples`.
2. Select target: `ACE2 — SARS-CoV-2 entry receptor`.

Expected:

- Sidebar shows a warning that this target is outside the Davis kinase model scope.
- Screening still works, but the limitation is visible.

## Quick Model Sanity Check

Run:

```bash
python -m pytest tests/ -q
python -m py_compile app.py src/chemistry.py src/screening.py
```

Expected:

- Tests pass.
- Compile check exits without errors.

You can also run this quick probability check:

```bash
python - <<'PY'
from src.model import load_artifacts, predict_binding
from src.features import featurize_pair
from src.screening import screen_single
from app import EXAMPLE_TARGETS

model, metrics, ref_db = load_artifacts("artifacts")
seq = EXAMPLE_TARGETS["CDK2 — cell cycle regulation"]["sequence"]

for name, smi in [
    ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
]:
    pred, prob = predict_binding(model, featurize_pair(smi, seq))
    row = screen_single({"name": name, "smiles": smi}, seq, model, ref_db)
    print(name, round(prob, 4), row["Binding Prob"], row["Priority"])
PY
```

Expected:

- The raw probability and `Binding Prob` should match.
- Example output from current artifacts:

```text
Aspirin 0.0039 0.0039 Low priority
Caffeine 0.0058 0.0058 Low priority
```

