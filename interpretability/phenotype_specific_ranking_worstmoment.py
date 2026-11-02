import numpy as np
import pandas as pd

feature_names = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun",
    "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2",
    "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
    "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt",
    "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine", "wbc"
]

phenotype_map = pd.read_csv("cohort_icd9_phenotype_mapping_with_fallback.csv")


def worst_moment_attribution(attr_array, X_raw_array, ids_df, model_name, top_n=3, min_n=30):
    """For each patient/feature, take attribution at the hour where that
    feature's raw value was most extreme (furthest from 0 in Z-scored space)."""
    n_patients, n_features, n_hours = attr_array.shape
    worst_attr = np.zeros((n_patients, n_features))

    for f in range(n_features):
        # Index of the most extreme (highest |value|) hour, per patient, for this feature
        worst_hour_idx = np.argmax(np.abs(X_raw_array[:, f, :]), axis=1)
        worst_attr[:, f] = np.abs(attr_array[np.arange(n_patients), f, worst_hour_idx])

    df = pd.DataFrame(worst_attr, columns=feature_names)
    df["stay_id"] = ids_df["stay_id"].values
    df = df.merge(phenotype_map, on="stay_id", how="inner")

    rows = []
    for phenotype, group in df.groupby("phenotype"):
        n = len(group)
        if n < min_n:
            continue
        mean_by_feature = group[feature_names].mean().sort_values(ascending=False)
        top_features = mean_by_feature.head(top_n)
        rows.append({
            "phenotype": phenotype,
            "n_patients": n,
            "model": model_name,
            **{f"top{i+1}_feature": top_features.index[i] for i in range(top_n)},
            **{f"top{i+1}_value": round(top_features.values[i], 4) for i in range(top_n)},
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TPC
# ---------------------------------------------------------------------------
tpc_shap = np.load("tpc_shap_values_full_testset.npy").squeeze(-1)   # (10367, 48, 24)
tpc_ids = pd.read_csv("tpc_shap_full_testset_stay_ids.csv")
tpc_X = np.load("tpc_X_scaled.npy")           # (51835, 48, 24)
tpc_test_rows = np.load("tpc_test_rows.npy")
tpc_X_test = tpc_X[tpc_test_rows]             # (10367, 48, 24)

tpc_phenotype_top = worst_moment_attribution(tpc_shap, tpc_X_test, tpc_ids, "TPC")

# ---------------------------------------------------------------------------
# BAT
# ---------------------------------------------------------------------------
bat_ig = np.load("bat_ig_values_full_testset_healthybaseline.npy")   # (10367, 48, 24)
bat_ids = pd.read_csv("bat_ig_full_testset_stay_ids_healthybaseline.csv")
bat_X = np.load("bat_X_scaled.npy")           # (51835, 24, 48)
bat_test_rows = np.load("bat_test_rows.npy")
bat_X_test = np.transpose(bat_X[bat_test_rows], (0, 2, 1))  # -> (10367, 48, 24)

bat_phenotype_top = worst_moment_attribution(bat_ig, bat_X_test, bat_ids, "BAT")

combined = pd.concat([tpc_phenotype_top, bat_phenotype_top], ignore_index=True)
combined = combined.sort_values(["phenotype", "model"])

print("=== Top-3 features per phenotype, per model (WORST-MOMENT attribution) ===\n")
print(combined.to_string(index=False))

combined.to_csv("phenotype_specific_feature_ranking_worstmoment.csv", index=False)
print("\nSaved: phenotype_specific_feature_ranking_worstmoment.csv")

# ---------------------------------------------------------------------------
# Same face-validity check as before, for direct comparison
# ---------------------------------------------------------------------------
EXPECTED = {
    "Acute and unspecified renal failure": ["crea", "bun"],
    "Other liver diseases": ["bili", "bili_dir"],
    "Chronic obstructive pulmonary disease and bronchiectasis": ["fio2", "o2sat", "resp"],
    "Acute myocardial infarction": ["map", "sbp", "dbp", "hr"],
}

print("\n=== Face-validity check (worst-moment) ===")
for phenotype, expected_features in EXPECTED.items():
    for model_name, df in [("TPC", tpc_phenotype_top), ("BAT", bat_phenotype_top)]:
        row = df[df["phenotype"] == phenotype]
        if len(row) == 0:
            continue
        top1 = row.iloc[0]["top1_feature"]
        match = "YES" if top1 in expected_features else "no"
        print(f"{phenotype} ({model_name}): top feature = {top1}, "
              f"expected one of {expected_features} -> match: {match}")