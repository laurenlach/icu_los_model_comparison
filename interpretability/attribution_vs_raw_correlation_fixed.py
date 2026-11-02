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

X = np.load("tpc_X_scaled.npy")
y = np.load("tpc_y.npy")
test_rows = np.load("tpc_test_rows.npy")
X_test = X[test_rows].mean(axis=2)
y_test = y[test_rows]

raw_correlations = np.array([abs(spearmanr(X_test[:, i], y_test)[0]) for i in range(48)])
raw_rank = np.argsort(-raw_correlations)

importance_df = pd.read_csv("feature_importance_comparison_fixed.csv")
importance_df = importance_df.set_index("feature").loc[feature_names].reset_index()
tpc_importance = importance_df["tpc_shap_importance_fixed"].values
tpc_rank = np.argsort(-tpc_importance)

overlap = set(tpc_rank[:10]) & set(raw_rank[:10])
print(f"TPC (fixed) top-10 vs raw-correlation top-10 overlap: {len(overlap)}/10 - "
      f"{[feature_names[i] for i in overlap]}")

rho, p = spearmanr(tpc_importance, raw_correlations)
print(f"Full-ranking agreement: rho={rho:.4f} (p={p:.4g})")
print("(Old buggy TPC: rho=-0.1008, p=0.4952; BAT: rho=-0.3028, p=0.036)")

pd.DataFrame({
    "feature": feature_names, "raw_correlation": raw_correlations,
    "tpc_importance_fixed": tpc_importance,
}).to_csv("attribution_vs_raw_correlation_fixed.csv", index=False)