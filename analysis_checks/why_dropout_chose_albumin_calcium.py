import numpy as np
import pandas as pd
from scipy.stats import spearmanr

feature_names = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun",
    "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2",
    "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
    "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt",
    "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine", "wbc"
]

mask = np.load("bat_mask.npy")[np.load("bat_test_rows.npy")]   # (n,24,48)
observed_rate = mask.mean(axis=(0, 1))   # fraction of patient-hours genuinely measured, per feature

importance = pd.read_csv("permutation_importance_tpc_featuredropout.csv").set_index("feature")
importance = importance.reindex(feature_names)["mae_increase"].values

df = pd.DataFrame({"feature": feature_names, "observed_rate": observed_rate, "mae_increase": importance})
df = df.sort_values("mae_increase", ascending=False)
print(df.head(10).to_string(index=False))

rho, p = spearmanr(df["observed_rate"], df["mae_increase"])
print(f"\nrho(measurement frequency, dropout-model importance) = {rho:+.3f} (p={p:.3g})")
print(f"(compare: original fixed-TPC model had rho=+0.580)")