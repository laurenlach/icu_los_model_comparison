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

# 1. Load raw feature values and true LOS for test patients
X = np.load("bat_X_scaled.npy")            # (51835, 24, 48)
y = np.load("bat_y.npy")                   # (51835,)
test_rows = np.load("bat_test_rows.npy")

X_test = X[test_rows].mean(axis=1)         # (10367, 48) - one value per patient per feature
y_test = y[test_rows]                      # (10367,) - true LOS in days


# 2. Plain Spearman correlation of each feature with true LOS

raw_correlations = []
for i, name in enumerate(feature_names):
    rho, _ = spearmanr(X_test[:, i], y_test)
    raw_correlations.append(abs(rho))  # magnitude - direction doesn't matter for ranking

raw_correlations = np.array(raw_correlations)
raw_rank = np.argsort(-raw_correlations)  # descending, most correlated first

print("=== Top-10 features by plain correlation with true LOS (no model) ===")
for i, idx in enumerate(raw_rank[:10]):
    print(f"{i+1}. {feature_names[idx]}: |rho|={raw_correlations[idx]:.4f}")


# 3. Load the model-based rankings and compare

importance_df = pd.read_csv("feature_importance_comparison_healthybaseline.csv")
importance_df = importance_df.set_index("feature").loc[feature_names].reset_index()

tpc_importance = importance_df["tpc_shap_importance"].values
bat_importance = importance_df["bat_ig_importance_healthybaseline"].values

tpc_rank = np.argsort(-tpc_importance)
bat_rank = np.argsort(-bat_importance)

raw_top10 = set(raw_rank[:10])
tpc_top10 = set(tpc_rank[:10])
bat_top10 = set(bat_rank[:10])

print(f"\nTPC top-10 vs. raw-correlation top-10 overlap: "
      f"{len(tpc_top10 & raw_top10)}/10 - "
      f"{[feature_names[i] for i in (tpc_top10 & raw_top10)]}")
print(f"BAT top-10 vs. raw-correlation top-10 overlap: "
      f"{len(bat_top10 & raw_top10)}/10 - "
      f"{[feature_names[i] for i in (bat_top10 & raw_top10)]}")

rho_tpc_vs_raw, p_tpc = spearmanr(tpc_importance, raw_correlations)
rho_bat_vs_raw, p_bat = spearmanr(bat_importance, raw_correlations)

print(f"\nFull-ranking (all 48 features) agreement with raw correlation:")
print(f"  TPC vs. raw: rho={rho_tpc_vs_raw:.4f} (p={p_tpc:.4g})")
print(f"  BAT vs. raw: rho={rho_bat_vs_raw:.4f} (p={p_bat:.4g})")

# 4. Save results

results_df = pd.DataFrame({
    "feature": feature_names,
    "raw_correlation_with_los": raw_correlations,
    "raw_rank": np.argsort(raw_rank) + 1,
    "tpc_shap_importance": tpc_importance,
    "tpc_rank": np.argsort(tpc_rank) + 1,
    "bat_ig_importance_healthybaseline": bat_importance,
    "bat_rank_healthybaseline": np.argsort(bat_rank) + 1,
})
results_df.to_csv("attribution_vs_raw_correlation.csv", index=False)
print("\nSaved: attribution_vs_raw_correlation.csv")