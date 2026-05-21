# Model Card — DrugLens Kinase Binding Classifier

## Overview

Binary classifier predicting whether a small molecule (drug) binds to a protein target, trained on kinase interaction data.

## Dataset

- **Source**: Davis kinase binding affinity dataset via Therapeutics Data Commons (TDC)
- **Pairs**: 30,056 drug-target pairs
- **Unique drugs**: 68
- **Unique targets**: 379
- **Binding ratio**: ~5.1% positive (highly imbalanced)
- **Split**: Train 21,039 / Validation 4,508 / Test 4,509
- **Binarization**: Continuous Kd values converted to binary labels using standard TDC threshold

## Model Type

- XGBoost gradient boosted decision trees (`xgboost.XGBClassifier`)
- Feature dimension: 2,483

## Features

- **Drug features**: Morgan fingerprints (2,048-bit circular fingerprints, radius 2) + 10 physicochemical descriptors (MW, LogP, HBD, HBA, TPSA, rotatable bonds, aromatic rings, heavy atoms, fraction CSP3, valence electrons)
- **Protein features**: Amino acid composition (20 features) + dipeptide composition (400 features capturing adjacent residue pair frequencies) + bulk sequence properties (length, estimated MW, fraction charged, fraction hydrophobic, fraction polar)
- **Combined**: Concatenation of drug and protein feature vectors

## Performance (Test Set)

| Metric    | Value |
|-----------|-------|
| AUROC     | 0.935 |
| AUPRC     | 0.629 |
| Accuracy  | 0.960 |
| Precision | 0.593 |
| Recall    | 0.655 |
| F1        | 0.622 |

Note: High accuracy is partly due to class imbalance (95% negative). AUROC and AUPRC are more informative metrics for this task.

## Intended Use

- Computational prioritization of compound libraries against kinase targets
- Early-stage virtual screening to narrow candidates for experimental follow-up
- Educational demonstration of ML-based drug-target interaction prediction

## Not Intended For

- Clinical decision-making or patient treatment
- Replacing experimental binding assays
- Predicting binding affinity magnitude (this is classification, not regression)
- Regulatory submissions or drug approval processes

## Limitations

- **Kinase-specific**: Trained exclusively on kinase targets from the Davis dataset. Predictions for non-kinase proteins (e.g., GPCRs, ion channels, proteases) are out-of-domain extrapolations.
- **Small drug space**: Only 68 unique drugs in training. Structurally novel compounds may produce unreliable predictions.
- **No 3D information**: Uses sequence-based protein features and 2D molecular fingerprints. Binding site geometry, conformational flexibility, and protein-ligand docking are not captured.
- **Binary only**: Predicts bind/no-bind, not binding strength (Kd/Ki/IC50).
- **No calibration**: Predicted probabilities may not reflect true binding likelihood. Use for ranking, not absolute probability interpretation.
- **Temporal bias**: No temporal split — some information leakage between similar drug-target pairs in train/test is possible.

## Known Failure Modes

- Non-kinase targets produce predictions with no empirical grounding
- Very large or unusual molecules (outside Morgan fingerprint training distribution) may get default predictions
- Targets with sequences very different from Davis kinase set will rely on generic protein descriptors
- Class imbalance means the model is conservative — it may miss true binders (recall = 0.655)

## Ethical Considerations

This model is a portfolio/educational project. Its predictions should never be used as the sole basis for any medical, pharmaceutical, or clinical decision. All computational hits require wet-lab experimental validation.
