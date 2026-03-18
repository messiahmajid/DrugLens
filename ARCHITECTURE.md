# DrugLens - Architecture

## What It Does

DrugLens predicts whether a drug molecule will bind to a protein target in the human body - **before** anyone steps into a lab.

Drug discovery is one of the most expensive processes in science. Testing whether a drug interacts with a specific protein traditionally requires physical experiments that cost millions and take years. DrugLens uses machine learning to computationally screen drug-protein pairs in under half a second, helping researchers shortlist the most promising candidates before committing to expensive lab work.

You give it a drug's chemical structure and a protein's amino acid sequence. It gives you back a binding probability, an explanation of *why* it thinks they'll interact, and a list of similar known drugs for reference.

---

## The Problem It Solves

Bringing a new drug to market costs **$2-3 billion** and takes **10-15 years** on average. A huge chunk of that time and money is spent testing drug candidates that ultimately don't work. If you could computationally filter out the non-starters early, you'd dramatically reduce waste and accelerate the path to treatments that help people.

DrugLens targets the earliest stage of this pipeline: **"Will this molecule even bind to the target protein?"** If the answer is no, there's no point running further experiments.

---

## How It Works (The Architecture)

The system has three main stages:

```
 ┌──────────────────────────────────────────────┐
 │           STREAMLIT WEB APPLICATION           │
 │   Input drugs + proteins, see predictions     │
 ├──────────────────────────────────────────────┤
 │          PREDICTION + EXPLANATION              │
 │   XGBoost model · SHAP explainability         │
 ├──────────────────────────────────────────────┤
 │           FEATURE ENGINEERING                  │
 │   Drug fingerprints · Protein descriptors     │
 └──────────────────────────────────────────────┘
```

### Stage 1: Feature Engineering (Translating Chemistry into Numbers)

Machine learning models can't read molecular diagrams or protein sequences directly. This stage converts them into numerical representations:

**For the drug molecule (2,058 features):**
- A **Morgan Fingerprint** (2,048 bits) - a binary map of the molecular substructures present in the drug. Think of it as a barcode that captures the drug's shape and chemical neighborhoods.
- **10 physical/chemical descriptors** - molecular weight, how water-soluble it is, how many hydrogen bonds it can form, etc.

**For the protein target (425 features):**
- **Amino acid composition** (20 values) - how frequently each of the 20 amino acids appears
- **Dipeptide composition** (400 values) - patterns of amino acid pairs, capturing local sequence structure
- **Bulk properties** (5 values) - overall protein length, estimated weight, charge distribution

Combined, each drug-protein pair becomes a **2,483-dimensional numerical vector** that the model can work with.

### Stage 2: Prediction + Explanation

**The Model: XGBoost**
- A gradient-boosted decision tree classifier trained on **30,000+ experimentally measured drug-protein interactions** from the Davis Kinase dataset
- Achieves **0.935 AUROC** (meaning it correctly ranks true binders above non-binders 93.5% of the time)
- Prediction takes ~2 milliseconds

**The Explanation: SHAP**
- After every prediction, SHAP (SHapley Additive exPlanations) shows *which features* pushed the prediction toward binding or not-binding
- This is critical for scientific trust - researchers need to understand *why*, not just get a yes/no

**Similar Drug Search:**
- Finds the 5 most structurally similar drugs from the training database using Tanimoto similarity
- Gives researchers context: "Your molecule looks like Imatinib, which is a known binder"

### Stage 3: Web Application (Streamlit)

An interactive web interface where users can:
- Enter a drug (as a SMILES chemical notation string) or pick from famous examples (Imatinib, Erlotinib, Aspirin)
- Select a protein target from a library of 6 well-known targets (EGFR, ACE2, COX-2, etc.) or paste a custom sequence
- See the binding prediction, probability score, SHAP explanation chart, similar drugs, and 2D molecular visualization

Total prediction time: **under 500 milliseconds** from click to result.

---

## Tech Stack

| Category | Technology | Why This Choice |
|----------|-----------|-----------------|
| ML Model | XGBoost | Best-in-class for tabular data, fast, explainable |
| Chemistry | RDKit | Industry-standard molecular processing toolkit |
| Biology | BioPython | Protein sequence analysis |
| Explainability | SHAP | Gold standard for model interpretability |
| Web App | Streamlit | Fast to build, interactive, easy to deploy |
| Data Source | Davis Kinase Dataset (TDC) | 30K+ experimentally validated interactions |
| Training Data | scikit-learn | Evaluation metrics and train/test splitting |

---

## Project Structure

```
DrugLens/
├── app.py               # Streamlit web application
├── train.py             # Training pipeline (run once)
├── src/
│   ├── data.py          # Dataset loading and processing
│   ├── features.py      # Drug + protein featurization
│   ├── model.py         # XGBoost training and prediction
│   ├── explainability.py # SHAP analysis and visualization
│   └── similarity.py    # Similar drug search
├── artifacts/           # Saved model + reference database
└── .streamlit/          # UI theme configuration
```

---

## What Makes It Interesting

- **Sub-second predictions** for a process that takes weeks in a lab
- **Explainable AI** - every prediction comes with a "here's why" breakdown, critical for scientific credibility
- **End-to-end pipeline** - from raw chemical data to interactive predictions in one system
- **Honest about limitations** - trained on kinase proteins only; clearly documents where it won't generalize
- **Production-ready** - deployed on Streamlit Cloud, no GPU needed, runs on any laptop
