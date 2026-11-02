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

tpc_shap = np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1)
bat_ig = np.load("bat_ig_values_full_testset_healthybaseline.npy")

tpc_importance = np.abs(tpc_shap).mean(axis=(0, 2))
bat_importance = np.abs(bat_ig).mean(axis=(0, 2))

tpc_rank = np.argsort(-tpc_importance)
bat_rank = np.argsort(-bat_importance)

print("=== TPC (FIXED) Top 10 ===")
for i, idx in enumerate(tpc_rank[:10]):
    print(f"{i+1}. {feature_names[idx]}: {tpc_importance[idx]:.4f}")

print("\n=== BAT Top 10 (unchanged) ===")
for i, idx in enumerate(bat_rank[:10]):
    print(f"{i+1}. {feature_names[idx]}: {bat_importance[idx]:.4f}")

overlap = set(tpc_rank[:10]) & set(bat_rank[:10])
print(f"\nTop-10 overlap: {len(overlap)}/10 - {[feature_names[i] for i in overlap]}")

rho, p = spearmanr(tpc_importance, bat_importance)
print(f"Spearman (all 48): rho={rho:.4f}, p={p:.4g}")

# Compare fixed TPC's ranking against the OLD buggy TPC's ranking
old_tpc_shap = np.load("tpc_shap_values_full_testset.npy").squeeze(-1)
old_tpc_importance = np.abs(old_tpc_shap).mean(axis=(0, 2))
rho_old_vs_new, _ = spearmanr(old_tpc_importance, tpc_importance)
print(f"\nTPC ranking stability - old (buggy) vs new (fixed): rho={rho_old_vs_new:.4f}")

results_df = pd.DataFrame({
    "feature": feature_names,
    "tpc_shap_importance_fixed": tpc_importance,
    "tpc_shap_importance_old": old_tpc_importance,
    "bat_ig_importance": bat_importance,
    "tpc_rank_fixed": np.argsort(tpc_rank) + 1,
    "bat_rank": np.argsort(bat_rank) + 1,
})
results_df.to_csv("feature_importance_comparison_fixed.csv", index=False)
print("\nSaved: feature_importance_comparison_fixed.csv")