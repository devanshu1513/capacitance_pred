# Cross-Paper Review: Capacitance Prediction for Supercapacitor Electrodes

A portfolio review of a machine-learning project predicting specific capacitance of biomass-derived carbon and composite supercapacitor electrodes. The upstream implementation uses CatBoost, engineered electrochemical and material features, paper-wise evaluation, external validation, feature analysis, and split conformal prediction intervals.

> **Attribution:** This is a public fork of [khub-crystalparadigm/capacitance_pred](https://github.com/khub-crystalparadigm/capacitance_pred). The model implementation, notebook, and source datasets remain from that project. My work in this portfolio is the independent evaluation review documented below; I did not create the original model or dataset.

## My contribution: cross-paper validation review

I inspected the notebook and its saved outputs to assess whether its evaluation supports claims about predictions on new publications. The notebook was not rerun for this review, so the values below are reported outputs rather than independently reproduced results.

### Results reported by the notebook

| Evaluation | Records | R² | RMSE (F/g) | MAE (F/g) |
|---|---:|---:|---:|---:|
| Random held-out split | 110 | 0.930 | 86.21 | — |
| DOI-grouped held-out split | 95 | 0.150 | 495.61 | — |
| External workbook | 31 | 0.235 | 472.20 | 386.34 |

The gap between the random split and DOI-grouped/external results suggests the random split overstates performance on unseen papers. External validation also reports Pearson *r* = 0.712 and Spearman ρ = 0.552; correlation does not imply small absolute prediction errors.

### Findings

- **Test-set reuse during early stopping:** each held-out test set is passed to CatBoost for early stopping, then used for the reported score. Since test labels influence the selected iteration, those scores are not unbiased final estimates.
- **DOI interval under-coverage:** saved DOI-grouped conformal intervals cover 76.8% of test records, below the nominal 90%. The notebook's interpretation text incorrectly describes both splits as close to 90% coverage.
- **Inactive feature:** is_three_electrode appears to be zero for every record. Labels are normalized to three_electrode, then the feature is derived by searching for the character 3.
- **Reproducibility gap:** saved notebook outputs come from a Windows environment. One cell may install openpyxl automatically, so a clean rerun should first review that cell and record the environment.

### Recommended evaluation redesign

1. Reserve papers for a final DOI-level test set before model development.
2. Use only training and validation papers for early stopping and feature choices.
3. Keep conformal calibration papers disjoint from model fitting and final testing; assess coverage at paper level as well as row level.
4. Fix the electrode-configuration indicator and add checks that verify engineered flags vary as expected.
5. Rerun from a clean environment and report split DOI lists, package versions, metrics, and interval coverage together.

These are recommendations from the review, not changes already applied to the upstream model.

## Upstream project

The upstream project predicts gravimetric specific capacitance (F/g) from literature-derived electrode and material descriptors using CatBoost. Its notebook includes preprocessing, feature engineering, random and DOI-grouped splits, external validation, feature importance, SHAP, ablation analysis, and conformal intervals.

The training CSV contains 910 records from 52 DOIs; the notebook reports 549 records after filtering. The external validation workbook contains 31 rows.

## Files and use

- Final_Model_Catboost.ipynb — upstream model and analysis notebook.
- composites.csv — upstream literature-derived training data.
- Master_External_Validation_Dataset.xlsx — upstream external-validation data.

The upstream README recommends Python 3.8+ with catboost, scikit-learn, xgboost, shap, pandas, numpy, matplotlib, openpyxl, and scipy. Open the notebook with both datasets in the same directory. Some cells train several models and may take several minutes. Review the automatic openpyxl installation cell before running.

## Resume framing

A truthful description of this contribution is:

> Audited a literature-based CatBoost capacitance predictor's saved random-split, DOI-grouped, and external-validation results; identified test-set reuse in early stopping, DOI interval under-coverage, and an inactive electrode-configuration feature; proposed a paper-disjoint evaluation redesign.

List the upstream model as a forked project and distinguish your review from the original implementation.
