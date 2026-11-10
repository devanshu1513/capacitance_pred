# Supercapacitor Capacitance Prediction

An independent CatBoost pipeline for predicting gravimetric specific capacitance (F/g) from literature-derived electrode data. The evaluation holds out entire publications so test papers do not guide feature choice or early stopping.

## Implementation

- DOI-disjoint train, validation, and test partitions with overlap assertions.
- Unit filtering and conversion, range parsing, categorical normalization, and material/electrochemical feature engineering.
- CatBoost early stopping on validation papers; the held-out test papers are scored only after training.
- Reproducible split seed, saved model, metrics, and test predictions.

The separate implementation is in train_model.py. The original notebook and datasets remain in the repository as upstream reference material and are credited below.

## Measured result

Run on composites.csv with seed 42, CatBoost 1.2.10, scikit-learn 1.9.1, pandas 3.1.1, and NumPy 2.5.4. DOI groups were disjoint across all partitions.

| Partition | Records | Papers (DOIs) |
|---|---:|---:|
| Train | 339 | 24 |
| Validation | 114 | 6 |
| Test | 95 | 8 |

The model stopped at iteration 114 using validation RMSE. On the held-out test papers it achieved **R² 0.252**, **RMSE 464.79 F/g**, and **MAE 371.88 F/g**. These results are from one deterministic split and should be described as a baseline, not as broad evidence of generalization.

## Run

Install catboost, scikit-learn, pandas, and numpy, then run:

    python train_model.py --data composites.csv --out artifacts

The script writes the trained model, metrics.json, and test_predictions.csv to the output directory. The metrics shown above were produced by running the implementation locally.

## Data and attribution

This repository is a public fork of [khub-crystalparadigm/capacitance_pred](https://github.com/khub-crystalparadigm/capacitance_pred). The original notebook and literature-derived datasets are retained from upstream and remain attributed to that project. The CatBoost pipeline in train_model.py is a separate implementation.

## Resume bullet

> Implemented a CatBoost capacitance-prediction baseline with DOI-disjoint train/validation/test splits and validation-only early stopping; evaluated 95 measurements from 8 unseen papers (R² 0.252, RMSE 464.79 F/g, MAE 371.88 F/g).
