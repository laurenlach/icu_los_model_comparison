import numpy as np
import pandas as pd

importance_df = pd.read_csv("feature_importance_comparison_fixed.csv")
tpc_top10 = importance_df.sort_values("tpc_rank_fixed").head(10)["feature"].tolist()
print("TPC (fixed) top-10:", tpc_top10)

feature_names = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun",
    "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2",
    "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
    "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt",
    "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine", "wbc"
]

X = np.load("tpc_X_scaled.npy")
test_rows = np.load("tpc_test_rows.npy")
X_test = X[test_rows].mean(axis=2)  # (10367, 48)
df_features = pd.DataFrame(X_test, columns=feature_names)

corr = df_features[tpc_top10].corr(method="spearman")
print("\nHighly correlated pairs (|rho|>0.6):")
for i in range(len(tpc_top10)):
    for j in range(i+1, len(tpc_top10)):
        r = corr.iloc[i, j]
        if abs(r) > 0.6:
            print(f"  {tpc_top10[i]} <-> {tpc_top10[j]}: rho={r:.3f}")

def normalized_entropy(v):
    p = v / v.sum()
    p = p[p > 0]
    return -np.sum(p * np.log(p)) / np.log(len(v))

tpc_importance = importance_df["tpc_shap_importance_fixed"].values
print(f"\nTPC (fixed) entropy: {normalized_entropy(tpc_importance):.4f}")
print("(Old buggy TPC entropy was 0.8356; BAT is 0.7952)")