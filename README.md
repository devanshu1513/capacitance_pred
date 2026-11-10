# Supercapacitor Capacitance Prediction

An independent CatBoost pipeline for predicting gravimetric specific capacitance (F/g) from literature-derived electrode data. The pipeline reserves entire publications for validation and final testing, so held-out test papers do not guide early stopping.

## What I implemented

- DOI-disjoint train, validation, and test partitioning with explicit overlap checks.
- Unit filtering and conversion, numeric range parsing, categorical normalization, and material/electrochemical feature engineering.
- CatBoost early stopping on validation papers; the held-out test papers are evaluated once after training.
- Saved test metrics, predictions, and model artifact for reproducibility.

The implementation is in train_model.py. It was written for this portfolio and has not yet been run; reported model metrics are not available until it is executed.

## Run

Install catboost, scikit-learn, pandas, and numpy, then run:

    python train_model.py --data composites.csv --out artifacts

The script writes the trained model, metrics.json, and test_predictions.csv to the output directory. Its DOI split is deterministic with seed 42.

## Data and attribution

This repository is a public fork of [khub-crystalparadigm/capacitance_pred](https://github.com/khub-crystalparadigm/capacitance_pred). The original notebook and both literature-derived datasets are retained from upstream and remain attributed to that project. My implementation is a separate baseline; it does not reproduce or replace the upstream notebook.

## Validation notes

The upstream notebook's saved outputs report R² = 0.930 on a random row split, 0.150 on a DOI-grouped split, and 0.235 on a 31-row external set. It also uses test data for early stopping, reports only 76.8% coverage for DOI-grouped nominal 90% intervals, and appears to create an always-zero three-electrode flag. Those are saved upstream results, not results from my new pipeline. The new script avoids the test-set early-stopping issue; it still needs to be run and its performance independently assessed before drawing conclusions.
