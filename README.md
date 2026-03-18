# DrugLens 🔬

**An AI-powered drug-target interaction predictor that predicts whether a drug molecule will bind to a protein target — the core question behind every drug discovery program.**

DrugLens uses molecular fingerprinting, protein sequence features, and gradient-boosted machine learning to predict binding interactions, explain *why* a molecule is predicted to bind, and suggest similar known drugs.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- **Binding Prediction**: Input any drug molecule (SMILES) and protein target to get a binding probability with confidence score
- **Explainability**: SHAP-powered feature importance showing *which molecular and protein properties* drive each prediction
- **Similar Drugs Panel**: Finds structurally similar compounds from the training database using Tanimoto similarity
- **Molecule Visualization**: 2D structure rendering for any input molecule
- **Curated Target Library**: Pre-loaded panel of well-known drug targets (EGFR, ACE2, COX-2, etc.) for quick demos

## Tech Stack

- **Data**: Therapeutics Data Commons (TDC) — Davis Kinase Binding Dataset
- **Molecular Features**: Morgan fingerprints (2048-bit) via RDKit + physicochemical descriptors
- **Protein Features**: Amino acid composition + dipeptide frequencies + sequence properties
- **Model**: XGBoost gradient-boosted classifier with Bayesian-optimized hyperparameters
- **Explainability**: SHAP TreeExplainer
- **Frontend**: Streamlit with custom theming
- **Similarity**: Tanimoto coefficient on Morgan fingerprints

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/messiahmajid/DrugLens.git
cd DrugLens
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

This downloads the Davis dataset via TDC, engineers features, trains the model, and saves all artifacts to `artifacts/`. Takes ~5-15 minutes depending on your machine.

### 3. Launch the app

```bash
streamlit run app.py
```

### 4. Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set `app.py` as the main file
4. Deploy — your `artifacts/` folder must be committed to the repo

## Project Structure

```
DrugLens/
├── app.py                    # Streamlit web application
├── train.py                  # Model training pipeline
├── requirements.txt          # Python dependencies
├── packages.txt              # System deps for Streamlit Cloud
├── .streamlit/
│   └── config.toml           # App theming
├── src/
│   ├── __init__.py
│   ├── data.py               # Dataset loading and processing
│   ├── features.py           # Molecular and protein featurization
│   ├── model.py              # Model training, evaluation, prediction
│   ├── similarity.py         # Tanimoto similar drug search
│   └── explainability.py     # SHAP explanations
└── artifacts/                # Created by train.py
    ├── model.joblib
    ├── metrics.json
    └── reference_db.joblib
```

## How It Works

1. **Molecular Representation**: Each drug is converted from its SMILES string into a 2048-bit Morgan fingerprint (capturing circular substructures) plus physicochemical descriptors (molecular weight, LogP, H-bond donors/acceptors, etc.)

2. **Protein Representation**: Each protein target's amino acid sequence is converted into compositional features — amino acid frequencies, dipeptide frequencies, and bulk sequence properties (length, charge, hydrophobicity)

3. **Combined Feature Vector**: Drug features and protein features are concatenated into a single vector that represents the drug-target pair

4. **Classification**: An XGBoost model trained on ~30,000 experimentally measured interactions predicts binding probability

5. **Explanation**: SHAP values decompose each prediction into per-feature contributions, showing which molecular or protein properties pushed the prediction toward binding or non-binding

## Author

**Messiah Godfred Majid**
University of Miami | Computer Science, Mathematics, and Biology
[messiahmajid.dev](https://messiahmajid.dev) | [LinkedIn](https://linkedin.com/in/messiahmajid) | [GitHub](https://github.com/messiahmajid)

## License

MIT License — see LICENSE for details.
