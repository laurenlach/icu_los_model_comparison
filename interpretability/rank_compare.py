import numpy as np

tpc_shap = np.load("tpc_shap_values_full_testset.npy")
bat_ig = np.load("bat_ig_values_full_testset.npy")

# Squeeze TPC's trailing output-dim (10367, 48, 24, 1) -> (10367, 48, 24)
tpc_shap = tpc_shap.squeeze(-1)

print("TPC SHAP shape after squeeze:", tpc_shap.shape)
print("BAT IG shape:", bat_ig.shape)

# Mean |attribution| per feature, averaged over patients AND time
# axis=(0,2) collapses patients (axis 0) and time (axis 2), leaving one value per feature
tpc_feature_importance = np.abs(tpc_shap).mean(axis=(0, 2))  # shape: (48,)
bat_feature_importance = np.abs(bat_ig).mean(axis=(0, 2))    # shape: (48,)

# --- Fill this in once you've pulled the ordered list from build_bat_los_dataset.py ---
feature_names = [f"feature_{i}" for i in range(48)]  # PLACEHOLDER — replace with real names
# ----------------------------------------------------------------------------------

# Rank features by importance, per model
tpc_rank = np.argsort(-tpc_feature_importance)  # descending
bat_rank = np.argsort(-bat_feature_importance)

print("\n=== TPC Top 10 features (SHAP) ===")
for i, idx in enumerate(tpc_rank[:10]):
    print(f"{i+1}. {feature_names[idx]}: {tpc_feature_importance[idx]:.4f}")

print("\n=== BAT Top 10 features (IG) ===")
for i, idx in enumerate(bat_rank[:10]):
    print(f"{i+1}. {feature_names[idx]}: {bat_feature_importance[idx]:.4f}")

# Overlap between top-10 sets
tpc_top10 = set(tpc_rank[:10])
bat_top10 = set(bat_rank[:10])
overlap = tpc_top10 & bat_top10
print(f"\nTop-10 overlap: {len(overlap)}/10 features in common")
print("Shared features:", [feature_names[i] for i in overlap])

# Full-ranking agreement (Spearman rank correlation across all 48 features)
from scipy.stats import spearmanr
rho, pval = spearmanr(tpc_feature_importance, bat_feature_importance)
print(f"\nSpearman correlation between full TPC and BAT rankings: rho={rho:.4f}, p={pval:.4g}")

# Save results for later use in the write-up / reliability matrix update
import pandas as pd
results_df = pd.DataFrame({
    "feature": feature_names,
    "tpc_shap_importance": tpc_feature_importance,
    "bat_ig_importance": bat_feature_importance,
    "tpc_rank": np.argsort(tpc_rank) + 1,   # rank position (1 = most important)
    "bat_rank": np.argsort(bat_rank) + 1,
})
results_df.to_csv("feature_importance_comparison.csv", index=False)
print("\nSaved: feature_importance_comparison.csv")