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

# TPC's SHAP is unaffected by BAT's baseline change - reuse the original
tpc_shap = np.load("tpc_shap_values_full_testset.npy").squeeze(-1)

# BAT: NEW healthy-baseline IG values
bat_ig = np.load("bat_ig_values_full_testset_healthybaseline.npy")

tpc_feature_importance = np.abs(tpc_shap).mean(axis=(0, 2))
bat_feature_importance = np.abs(bat_ig).mean(axis=(0, 2))

tpc_rank = np.argsort(-tpc_feature_importance)
bat_rank = np.argsort(-bat_feature_importance)

print("=== TPC Top 10 (unchanged) ===")
for i, idx in enumerate(tpc_rank[:10]):
    print(f"{i+1}. {feature_names[idx]}: {tpc_feature_importance[idx]:.4f}")

print("\n=== BAT Top 10 (HEALTHY baseline) ===")
for i, idx in enumerate(bat_rank[:10]):
    print(f"{i+1}. {feature_names[idx]}: {bat_feature_importance[idx]:.4f}")

tpc_top10 = set(tpc_rank[:10])
bat_top10 = set(bat_rank[:10])
overlap = tpc_top10 & bat_top10
print(f"\nTop-10 overlap: {len(overlap)}/10 - {[feature_names[i] for i in overlap]}")

rho, pval = spearmanr(tpc_feature_importance, bat_feature_importance)
print(f"\nSpearman correlation (full 48 features): rho={rho:.4f}, p={pval:.4g}")

# Compare against the OLD baseline's ranking too, so you can see how much this changed
old_bat_ig = np.load("bat_ig_values_full_testset.npy")
old_bat_feature_importance = np.abs(old_bat_ig).mean(axis=(0, 2))
old_bat_rank = np.argsort(-old_bat_feature_importance)

rank_shift_rho, _ = spearmanr(old_bat_feature_importance, bat_feature_importance)
print(f"\nBAT ranking stability - old vs new baseline: rho={rank_shift_rho:.4f} "
      f"(1.0 = identical ranking, 0 = no relationship)")

results_df = pd.DataFrame({
    "feature": feature_names,
    "tpc_shap_importance": tpc_feature_importance,
    "bat_ig_importance_healthybaseline": bat_feature_importance,
    "bat_ig_importance_oldbaseline": old_bat_feature_importance,
    "tpc_rank": np.argsort(tpc_rank) + 1,
    "bat_rank_healthybaseline": np.argsort(bat_rank) + 1,
    "bat_rank_oldbaseline": np.argsort(old_bat_rank) + 1,
})
results_df.to_csv("feature_importance_comparison_healthybaseline.csv", index=False)
print("\nSaved: feature_importance_comparison_healthybaseline.csv")