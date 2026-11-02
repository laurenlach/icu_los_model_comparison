import numpy as np
import pandas as pd

feature_names = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun",
    "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2",
    "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
    "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt",
    "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine", "wbc"
]

# ---------------------------------------------------------------------------
# 1. Load attribution + stay IDs for each model, and the phenotype mapping
# ---------------------------------------------------------------------------
tpc_shap = np.load("tpc_shap_values_full_testset.npy").squeeze(-1)   # (10367, 48, 24)
tpc_ids = pd.read_csv("tpc_shap_full_testset_stay_ids.csv")

bat_ig = np.load("bat_ig_values_full_testset_healthybaseline.npy")   # (10367, 48, 24)
bat_ids = pd.read_csv("bat_ig_full_testset_stay_ids_healthybaseline.csv")

phenotype_map = pd.read_csv("cohort_icd9_phenotype_mapping_with_fallback.csv")

def top_features_per_phenotype(attr_array, ids_df, model_name, top_n=3, min_n=30):
    mean_attr = np.abs(attr_array).mean(axis=2)  # (n_patients, 48) - mean over window
    df = pd.DataFrame(mean_attr, columns=feature_names)
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

tpc_phenotype_top = top_features_per_phenotype(tpc_shap, tpc_ids, "TPC")
bat_phenotype_top = top_features_per_phenotype(bat_ig, bat_ids, "BAT")

combined = pd.concat([tpc_phenotype_top, bat_phenotype_top], ignore_index=True)
combined = combined.sort_values(["phenotype", "model"])

print("=== Top-3 features per phenotype, per model ===\n")
print(combined.to_string(index=False))

combined.to_csv("phenotype_specific_feature_ranking.csv", index=False)
print("\nSaved: phenotype_specific_feature_ranking.csv")

# ---------------------------------------------------------------------------
# 2. Quick face-validity check: does the #1 feature match clinical expectation
#    for a few obviously-relevant phenotypes?
# ---------------------------------------------------------------------------
EXPECTED = {
    "Acute and unspecified renal failure": ["crea", "bun"],
    "Other liver diseases": ["bili", "bili_dir"],
    "Chronic obstructive pulmonary disease and bronchiectasis": ["fio2", "o2sat", "resp"],
    "Acute myocardial infarction": ["map", "sbp", "dbp", "hr"],
}

print("\n=== Face-validity check: does the #1-ranked feature match clinical expectation? ===")
for phenotype, expected_features in EXPECTED.items():
    for model_name, df in [("TPC", tpc_phenotype_top), ("BAT", bat_phenotype_top)]:
        row = df[df["phenotype"] == phenotype]
        if len(row) == 0:
            continue
        top1 = row.iloc[0]["top1_feature"]
        match = "YES" if top1 in expected_features else "no"
        print(f"{phenotype} ({model_name}): top feature = {top1}, "
              f"expected one of {expected_features} -> match: {match}")