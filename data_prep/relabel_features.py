import pandas as pd

# The real, ordered feature names — pulled directly from dyn.parquet via get_feature_names.py
feature_names = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun",
    "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2",
    "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
    "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt",
    "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine", "wbc"
]

# Load the results we saved earlier (currently labeled feature_0, feature_1, ...)
df = pd.read_csv("feature_importance_comparison.csv")

# Replace the placeholder names with the real ones, in the same order
df["feature"] = feature_names

df.to_csv("feature_importance_comparison.csv", index=False)
print("Re-labeled and saved feature_importance_comparison.csv with real feature names")
print(df.sort_values("tpc_rank").head(10))