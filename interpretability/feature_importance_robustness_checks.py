import numpy as np
import pandas as pd


# 1. Load the final feature-importance ranking 

importance_df = pd.read_csv("feature_importance_comparison_healthybaseline.csv")

tpc_top10 = importance_df.sort_values("tpc_rank").head(10)["feature"].tolist()
bat_top10 = importance_df.sort_values("bat_rank_healthybaseline").head(10)["feature"].tolist()

print("TPC top-10:", tpc_top10)
print("BAT top-10:", bat_top10)


# 2. Correlated-feature critique: how correlated are each model's top-10
#    features with each other OTHER 

feature_names = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun",
    "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2",
    "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
    "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt",
    "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine", "wbc"
]

X = np.load("bat_X_scaled.npy")           # (51835, 24, 48)
test_rows = np.load("bat_test_rows.npy")
X_test = X[test_rows]                     # (10367, 24, 48)

# Collapse to one value per patient per feature (mean across the 24h window),
# so each patient contributes one row 

X_test_flat = X_test.mean(axis=1)         # (10367, 48)

df_features = pd.DataFrame(X_test_flat, columns=feature_names)


def print_correlation_block(top10, model_name):
    print(f"\n=== {model_name} top-10 feature correlation matrix ===")
    corr = df_features[top10].corr(method="spearman")
    print(corr.round(2).to_string())

    # Flag pairs with |correlation| > 0.6 as "meaningfully redundant"
    print(f"\nHighly correlated pairs within {model_name}'s top-10 (|rho| > 0.6):")
    flagged = []
    for i in range(len(top10)):
        for j in range(i + 1, len(top10)):
            r = corr.iloc[i, j]
            if abs(r) > 0.6:
                flagged.append((top10[i], top10[j], round(r, 3)))
                print(f"  {top10[i]} <-> {top10[j]}: rho={r:.3f}")
    if not flagged:
        print("  None found.")
    return corr, flagged


tpc_corr, tpc_flagged = print_correlation_block(tpc_top10, "TPC")
bat_corr, bat_flagged = print_correlation_block(bat_top10, "BAT")


# 3. Attribution entropy: does each model spread importance across many
#    features, or concentrate on a few? 

def normalized_entropy(importance_values):
    p = importance_values / importance_values.sum()
    p = p[p > 0]  # avoid log(0)
    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(len(importance_values))
    return entropy / max_entropy


tpc_importance = importance_df["tpc_shap_importance"].values
bat_importance = importance_df["bat_ig_importance_healthybaseline"].values

tpc_entropy = normalized_entropy(tpc_importance)
bat_entropy = normalized_entropy(bat_importance)

print(f"\n=== Attribution concentration (normalized entropy, 0=concentrated, 1=uniform) ===")
print(f"TPC (SHAP): {tpc_entropy:.4f}")
print(f"BAT (IG, healthy baseline): {bat_entropy:.4f}")


# 4. Save results

results = {
    "tpc_top10_correlated_pairs": tpc_flagged,
    "bat_top10_correlated_pairs": bat_flagged,
    "tpc_attribution_entropy": tpc_entropy,
    "bat_attribution_entropy": bat_entropy,
}
pd.DataFrame([results]).to_csv("feature_importance_robustness_summary.csv", index=False)
tpc_corr.to_csv("tpc_top10_correlation_matrix.csv")
bat_corr.to_csv("bat_top10_correlation_matrix.csv")

print("\nSaved: feature_importance_robustness_summary.csv, "
      "tpc_top10_correlation_matrix.csv, bat_top10_correlation_matrix.csv")