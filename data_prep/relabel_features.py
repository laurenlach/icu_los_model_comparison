import pandas as pd


feature_names = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun",
    "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2",
    "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
    "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt",
    "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine", "wbc"
]


df = pd.read_csv("feature_importance_comparison.csv")


df["feature"] = feature_names

df.to_csv("feature_importance_comparison.csv", index=False)
print("Re-labeled and saved feature_importance_comparison.csv with real feature names")
print(df.sort_values("tpc_rank").head(10))