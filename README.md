# CatBoost Supercapacitor Specific Capacitance Predictor

A machine learning model to predict **specific capacitance (F/g)** of supercapacitor electrodes using CatBoostRegressor, trained on experimentally sourced composite material data.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Ablation Study](#ablation-study)
- [Interpretability](#interpretability)

---

## Overview

This project builds a gradient-boosting regression model to predict the specific capacitance of electrode materials used in supercapacitors. The pipeline includes:

- Data filtering and unit standardization (GCD test mode, electrode measurements)
- Biomass precursor grouping into semantic categories
- Physics-informed feature engineering
- Ablation study across 6 progressively richer feature sets
- SHAP-based model interpretability

---

## Repository Structure

```
├── Final_Model_Catboost.ipynb   # Main notebook: preprocessing → training → evaluation
├── composites.csv               # Raw dataset (CSV)
└── README.md
```

---

## Dataset

**File:** `composites.csv`

The dataset is sourced from experimental literature on carbon-based composite electrode materials. After filtering:

- **Measurement type:** `electrode`
- **Test mode:** GCD (Galvanostatic Charge-Discharge)
- **Current density unit:** `mA/g` or `A/g` (standardized to `A/g`)
- **Capacitance unit:** `F/g` or `mF/g` (standardized to `F/g`)

---

## Features

The model uses 6 progressively richer feature groups:

| Group | Description |
|---|---|
| **Base** | Surface area, pore volume, pore size, current density, potential window, electrolyte concentration, mass & metal loading |
| **Physics-Informed** | Voltage², total resistance, conductivity index, log/inverse current density, surface-current interaction |
| **Electrochemical/Structural** | Rct, Rs, log-transformed resistances, ID/IG ratio, carbonization & activation temperatures, activation ratio |
| **Interaction** | SSA × pore interactions, surface-to-pore ratio, biomass-modulated pore/SSA effects |
| **Regime/Material** | One-hot flags for carbon, composite, metal oxide, metal sulfide, pseudocapacitive, 3-electrode; material-SSA interaction terms |
| **Categorical** | Biomass precursor, biomass group, activation agent, electrolyte, electrolyte state, material type, electrode configuration |

**Biomass groups:** `shell_based`, `protein_mineral`, `agro_residue`, `woody_structural`, `soft_biomass`, `food_waste`, `other_biomass`, `non_biomass`

---

## Model Architecture

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

Two evaluation strategies are used:
- **Random split** — standard 80/20 train-test split
- **Group split (DOI-wise)** — ensures no paper's data leaks across train/test sets

---

## Results

| Model | R² | RMSE |
|---|---|---|
| Linear Regression | — | — |
| Random Forest | — | — |
| XGBoost | — | — |
| **CatBoost (Full)** | **Best** | **Lowest** |

> Actual metric values are printed at runtime in the notebook.

Comparison models (Linear, RF, XGB) are benchmarked against CatBoost under both random and DOI-based group splits.

---

## Installation

```bash
# Clone the private repo
git clone https://github.com/<your-org>/<repo-name>.git
cd <repo-name>

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# Install dependencies
pip install catboost scikit-learn xgboost shap pandas numpy matplotlib
```

> Python 3.8+ recommended.

---

## Usage

1. Place `composites.csv` in the same directory as the notebook.
2. Open `Final_Model_Catboost.ipynb` in Jupyter:

```bash
jupyter notebook Final_Model_Catboost.ipynb
```

3. Run all cells top-to-bottom. The notebook will:
   - Filter and clean the dataset
   - Engineer all feature groups
   - Train CatBoost and baseline models
   - Print R² / RMSE for both random and group splits
   - Plot feature importances and SHAP summary

---

## Ablation Study

Six cumulative feature sets are evaluated to quantify each group's marginal contribution:

```
Base
Base + Physics
Base + Physics + Electrochemical
Base + Physics + Electrochemical + Interaction
Base + Physics + Electrochemical + Interaction + Regime
Full Model (All Features)
```

Results are printed as a sorted DataFrame showing R² and RMSE for each configuration.

---

## Interpretability

- **Feature Importance** — CatBoost's built-in feature importance (top 20 plotted)
- **SHAP** — `shap.TreeExplainer` generates a summary plot showing the direction and magnitude of each feature's impact on predictions

---

## Dependencies

| Package | Purpose |
|---|---|
| `catboost` | Primary model |
| `scikit-learn` | Baselines, metrics, train/test split |
| `xgboost` | Baseline model |
| `shap` | Model interpretability |
| `pandas`, `numpy` | Data processing |
| `matplotlib` | Plotting |
