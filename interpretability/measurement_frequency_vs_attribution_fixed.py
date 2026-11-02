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

mask = np.load("bat_mask.npy")  # TPC and BAT share the same underlying patients/mask structure
test_rows = np.load("bat_test_rows.npy")
mask_test = mask[test_rows]
observed_pct = mask_test.mean(axis=(0, 1)) * 100

importance_df = pd.read_csv("feature_importance_comparison_fixed.csv")
importance_df = importance_df.set_index("feature").loc[feature_names].reset_index()
tpc_importance = importance_df["tpc_shap_importance_fixed"].values

rho, p = spearmanr(observed_pct, tpc_importance)
print(f"TPC (fixed): rho={rho:.4f}, p={p:.4g}")
print("(Old buggy TPC: rho=0.6516, p=5.3e-07; BAT: rho=0.7034, p=2.5e-08)")