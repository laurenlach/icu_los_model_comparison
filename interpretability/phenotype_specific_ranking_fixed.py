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

tpc_shap = np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1)
tpc_ids = pd.read_csv("tpc_shap_full_testset_stay_ids_fixed.csv")

def top_features_per_phenotype(attr_array, ids_df, top_n=3, min_n=30):
    mean_attr = np.abs(attr_array).mean(axis=2)
    df = pd.DataFrame(mean_attr, columns=feature_names)
    df["stay_id"] = ids_df["stay_id"].values
    df = df.merge(phenotype_map, on="stay_id", how="inner")

    rows = []
    for phenotype, group in df.groupby("phenotype"):
        n = len(group)
        if n < min_n:
            continue
        top = group[feature_names].mean().sort_values(ascending=False).head(top_n)
        rows.append({
            "phenotype": phenotype, "n_patients": n,
            **{f"top{i+1}_feature": top.index[i] for i in range(top_n)},
        })
    return pd.DataFrame(rows)

tpc_phenotype_top = top_features_per_phenotype(tpc_shap, tpc_ids)
print(tpc_phenotype_top.to_string(index=False))

EXPECTED = {
    "Acute and unspecified renal failure": ["crea", "bun"],
    "Other liver diseases": ["bili", "bili_dir"],
    "Chronic obstructive pulmonary disease and bronchiectasis": ["fio2", "o2sat", "resp"],
    "Acute myocardial infarction": ["map", "sbp", "dbp", "hr"],
}
print("\n=== Face-validity check (fixed TPC) ===")
for phenotype, expected in EXPECTED.items():
    row = tpc_phenotype_top[tpc_phenotype_top["phenotype"] == phenotype]
    if len(row) == 0:
        continue
    top1 = row.iloc[0]["top1_feature"]
    print(f"{phenotype}: top = {top1}, expected {expected} -> match: {'YES' if top1 in expected else 'no'}")

tpc_phenotype_top.to_csv("phenotype_specific_feature_ranking_fixed.csv", index=False)
print("\nSaved: phenotype_specific_feature_ranking_fixed.csv")