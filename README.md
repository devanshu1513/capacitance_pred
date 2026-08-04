# CatBoost Supercapacitor Specific Capacitance Predictor

A machine learning pipeline to predict **specific capacitance (F/g)** of biomass-derived
carbon composite supercapacitor electrodes using CatBoostRegressor.

---

## Repository Structure

```
├── Final_Model_Catboost.ipynb              # Complete pipeline notebook
├── composites.csv                           # Training dataset
├── Master_External_Validation_Dataset.xlsx # Independent external validation dataset
└── README.md
```

---

## Overview

This project builds a gradient-boosting regression model to predict specific capacitance
of electrode materials used in supercapacitors. The pipeline includes:

- Data filtering and unit standardization (GCD test mode, electrode measurements only)
- Biomass precursor grouping into semantic categories
- Physics-informed feature engineering (6 cumulative feature groups)
- Two evaluation strategies: random split and DOI-grouped (paper-wise) split
- External validation on a completely independent dataset from separate publications
- Permutation importance on the held-out DOI-grouped test set
- SHAP-based model interpretability
- Ablation study across 6 progressively richer feature sets
- Uncertainty Quantification via Split Conformal Prediction (90% prediction intervals)

---

## Dataset

**Training:** `composites.csv`

Sourced from experimental literature on carbon-based composite electrode materials.
After filtering:
- Measurement type: `electrode`
- Test mode: GCD (Galvanostatic Charge-Discharge)
- Current density unit: `mA/g` or `A/g` (standardized to A/g)
- Capacitance unit: `F/g` or `mF/g` (standardized to F/g)

**External Validation:** `Master_External_Validation_Dataset.xlsx`

An independent set of measurements sourced from publications not present in the training
set. Used to evaluate true cross-paper generalization performance of the DOI-grouped model.

---

## Features

Six progressively richer feature groups:

| Group | Description |
|---|---|
| **Base** | Surface area, pore volume, pore size, current density, potential window, electrolyte concentration, mass & metal loading |
| **Physics-Informed** | Voltage², total resistance, conductivity index, log/inverse current density, SSA–pore interaction |
| **Electrochemical/Structural** | Rct, Rs, log-resistances, ID/IG ratio, carbonization & activation temperatures, activation ratio |
| **Interaction** | SSA × pore, surface-to-pore ratio, biomass-modulated SSA/pore effects |
| **Regime/Material** | Binary flags (carbon, composite, metal oxide, metal sulfide, pseudocapacitive, 3-electrode), material–SSA interaction terms |
| **Categorical** | Biomass precursor, biomass group, activation agent, electrolyte, electrolyte state, material type, electrode configuration |

---

## Model

```python
CatBoostRegressor(
    iterations=3000,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=5,
    random_strength=1,
    bagging_temperature=1,
    border_count=128,
    loss_function="RMSE",
    eval_metric="R2",
    early_stopping_rounds=100,
    random_seed=42
)
```

Categorical features are passed natively to CatBoost via `Pool` — no one-hot encoding required.

---

## Evaluation Strategy

| Strategy | Description |
|---|---|
| **Random Split** | Standard 80/20 train-test split |
| **DOI Group Split** | All measurements from the same DOI kept in the same fold — tests true cross-paper generalization |
| **External Validation** | Completely independent dataset from separate publications |

Actual metric values (R², RMSE, MAE, Pearson r, Spearman ρ) are printed at runtime in the notebook.

---

## Ablation Study

Six cumulative feature sets evaluated with 30-repeat DOI-grouped cross-validation:

```
Stage 1: Base
Stage 2: Base + Physics-Informed
Stage 3: Base + Physics + Electrochemical/Structural
Stage 4: Base + Physics + Electrochemical + Interaction
Stage 5: Base + Physics + Electrochemical + Interaction + Regime/Material
Stage 6: Full Model (All Features incl. Categorical)
```

---

## Uncertainty Quantification

Split Conformal Prediction is used to produce **90% prediction intervals**:
- **Random Split** calibration
- **DOI Group Split** calibration (tests cross-paper interval reliability)

Coverage calibration tables and parity plots with error bars are generated automatically.

---

## Interpretability

- **Feature Importance** — CatBoost built-in importance (PredictionValuesChange), top-20 bar chart
- **Permutation Importance** — Held-out DOI-grouped test set, top-20 bar chart
- **SHAP** — TreeExplainer summary plot showing direction and magnitude per feature per sample

---

## Installation

```bash
pip install catboost scikit-learn xgboost shap pandas numpy matplotlib openpyxl scipy
```

Python 3.8+ recommended.

---

## Usage

1. Place `composites.csv` and `Master_External_Validation_Dataset.xlsx` in the same
   directory as the notebook.
2. Open and run `Final_Model_Catboost.ipynb` top-to-bottom.

---

## Dependencies

| Package | Purpose |
|---|---|
| `catboost` | Primary model |
| `scikit-learn` | Baselines, metrics, conformal prediction utilities |
| `xgboost` | Baseline model |
| `shap` | SHAP interpretability |
| `scipy` | Pearson/Spearman correlation, paired t-tests |
| `openpyxl` | Read/write Excel files |
| `pandas`, `numpy` | Data processing |
| `matplotlib` | Plotting |
