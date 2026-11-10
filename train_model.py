"""Independent DOI-disjoint baseline for literature-derived capacitance data.

The source dataset and reference notebook are from the upstream project. This
script is a separate implementation with held-out papers reserved from model
selection. It does not use the test set for early stopping.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

NUMERIC = [
    "specific_surface_area_m2g", "pore_volume_cm3g", "average_pore_size_nm",
    "current_density_value", "potential_window_V", "electrolyte_concentration_M",
    "Rct_ohms", "Rs_ohms", "ID_IG_ratio", "carbonization_temperature_C",
    "activation_temperature_C", "activation_ratio", "calcination_temperature_C",
    "metal_loading_wt_percent", "mass_loading_mgcm2",
]
CATEGORICAL = [
    "biomass_group", "material_type", "electrode_configuration", "electrolyte",
    "electrolyte_state", "activation_agent",
]
TARGET = "specific_capacitance"


def _as_number(value):
    """Convert a number or simple numeric range to a float."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    matches = re.findall(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    if len(matches) == 1:
        return float(matches[0])
    if len(matches) == 2:
        return (float(matches[0]) + float(matches[1])) / 2
    return np.nan


def _biomass_group(value):
    if pd.isna(value) or str(value).strip().lower() in {"", "nan", "none", "unknown"}:
        return "non_biomass"
    s = str(value).lower()
    for label, terms in [
        ("shell_based", ("shell", "septum")),
        ("protein_mineral", ("crab", "egg", "agaric", "membrane")),
        ("agro_residue", ("bagasse", "straw", "husk", "stalk")),
        ("woody_structural", ("wood", "stem", "cellulose", "fiber")),
        ("soft_biomass", ("leaf", "leaves", "catkin")),
        ("food_waste", ("peel", "fruit", "seed", "potato")),
    ]:
        if any(term in s for term in terms):
            return label
    return "other_biomass"


def prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    required = {"doi", "measurement_type", "test_mode", "current_density_unit",
                "capacitance_unit", "current_density_value", TARGET}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    for col in ("measurement_type", "test_mode", "current_density_unit", "capacitance_unit"):
        df[col] = df[col].fillna("").astype(str).str.strip().str.lower()
    keep = (
        df["measurement_type"].eq("electrode")
        & df["test_mode"].str.contains("gcd", na=False)
        & df["current_density_unit"].str.contains(r"^(ma/g|a/g)$", regex=True, na=False)
        & df["capacitance_unit"].str.contains(r"^(mf/g|f/g)$", regex=True, na=False)
    )
    df = df.loc[keep].copy()
    df["doi"] = df["doi"].fillna("").astype(str).str.strip()
    df = df.loc[df["doi"].ne("")].copy()

    for col in NUMERIC + [TARGET]:
        if col not in df:
            df[col] = np.nan
        df[col] = df[col].map(_as_number)
    df.loc[df["current_density_unit"].eq("ma/g"), "current_density_value"] /= 1000
    df.loc[df["capacitance_unit"].eq("mf/g"), TARGET] /= 1000
    df = df.dropna(subset=[TARGET])

    for col in CATEGORICAL:
        if col not in df:
            df[col] = "unknown"
    df["biomass_group"] = df.get("biomass_precursor", pd.Series(index=df.index, dtype=object)).map(_biomass_group)
    df["material_type"] = df["material_type"].fillna("unknown").astype(str).str.lower().str.strip()
    df["electrolyte"] = df["electrolyte"].fillna("unknown").astype(str).str.lower().str.strip()
    df["electrolyte_state"] = df["electrolyte_state"].fillna("unknown").astype(str).str.lower().str.strip()
    df["activation_agent"] = df["activation_agent"].fillna("unknown").astype(str).str.lower().str.strip()
    config = df.get("electrode_configuration", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str).str.lower()
    df["electrode_configuration"] = np.select(
        [config.str.contains(r"3|three"), config.str.contains(r"2|two|symmetric|cell")],
        ["three_electrode", "two_electrode"], default="unknown"
    )

    cd = df["current_density_value"].where(df["current_density_value"] >= 0)
    df["cd_log"] = np.log1p(cd)
    df["cd_inverse"] = 1 / cd.replace(0, np.nan)
    df["voltage_squared"] = df["potential_window_V"] ** 2
    df["total_resistance"] = df["Rs_ohms"] + df["Rct_ohms"]
    df["log_total_resistance"] = np.log1p(df["total_resistance"].where(df["total_resistance"] >= 0))
    df["ssa_pore_interaction"] = df["specific_surface_area_m2g"] * df["pore_volume_cm3g"]
    df["is_three_electrode"] = df["electrode_configuration"].eq("three_electrode").astype(int)
    return df


def split_by_paper(X, y, groups, seed=42):
    """Create disjoint train/validation/test groups (64/16/20 approx.)."""
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    trainval_idx, test_idx = next(outer.split(X, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + 1)
    train_rel, val_rel = next(inner.split(X.iloc[trainval_idx], y.iloc[trainval_idx], groups.iloc[trainval_idx]))
    train_idx, val_idx = trainval_idx[train_rel], trainval_idx[val_rel]
    assert set(groups.iloc[train_idx]).isdisjoint(groups.iloc[val_idx])
    assert set(groups.iloc[train_idx]).isdisjoint(groups.iloc[test_idx])
    assert set(groups.iloc[val_idx]).isdisjoint(groups.iloc[test_idx])
    return train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("composites.csv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    df = prepare(args.data)
    features = NUMERIC + ["cd_log", "cd_inverse", "voltage_squared", "total_resistance",
                          "log_total_resistance", "ssa_pore_interaction", "is_three_electrode"] + CATEGORICAL
    X = df[features].copy()
    for col in CATEGORICAL:
        X[col] = X[col].fillna("unknown").astype(str)
    y, groups = df[TARGET], df["doi"]
    train_i, val_i, test_i = split_by_paper(X, y, groups)
    cat_idx = [X.columns.get_loc(c) for c in CATEGORICAL]
    train_pool = Pool(X.iloc[train_i], y.iloc[train_i], cat_features=cat_idx)
    val_pool = Pool(X.iloc[val_i], y.iloc[val_i], cat_features=cat_idx)
    model = CatBoostRegressor(
        iterations=3000, depth=6, learning_rate=0.03, l2_leaf_reg=5,
        loss_function="RMSE", eval_metric="RMSE", random_seed=42,
        early_stopping_rounds=100, verbose=200,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    test_pool = Pool(X.iloc[test_i], y.iloc[test_i], cat_features=cat_idx)
    pred = model.predict(test_pool)
    metrics = {
        "n_train": int(len(train_i)), "n_validation": int(len(val_i)), "n_test": int(len(test_i)),
        "train_dois": int(groups.iloc[train_i].nunique()),
        "validation_dois": int(groups.iloc[val_i].nunique()),
        "test_dois": int(groups.iloc[test_i].nunique()),
        "r2": float(r2_score(y.iloc[test_i], pred)),
        "rmse_F_per_g": float(np.sqrt(mean_squared_error(y.iloc[test_i], pred))),
        "mae_F_per_g": float(mean_absolute_error(y.iloc[test_i], pred)),
        "best_iteration": int(model.get_best_iteration()),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_model(args.out / "catboost_doi_disjoint.cbm")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame({"doi": groups.iloc[test_i].values, "actual_F_per_g": y.iloc[test_i].values,
                  "predicted_F_per_g": pred}).to_csv(args.out / "test_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
