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

# ---------------------------------------------------------------------------
# 1. Observed-rate per feature (already computed once, redo cleanly here so
#    this script stands alone)
# ---------------------------------------------------------------------------
mask = np.load("bat_mask.npy")
test_rows = np.load("bat_test_rows.npy")
mask_test = mask[test_rows]
observed_pct = mask_test.mean(axis=(0, 1)) * 100  # % observed, per feature

# ---------------------------------------------------------------------------
# 2. Load attribution importance (final, healthy-baseline version for BAT)
# ---------------------------------------------------------------------------
importance_df = pd.read_csv("feature_importance_comparison_healthybaseline.csv")
importance_df = importance_df.set_index("feature").loc[feature_names].reset_index()

tpc_importance = importance_df["tpc_shap_importance"].values
bat_importance = importance_df["bat_ig_importance_healthybaseline"].values

# ---------------------------------------------------------------------------
# 3. Correlate observed-rate against attribution, for each model
# ---------------------------------------------------------------------------
rho_tpc, p_tpc = spearmanr(observed_pct, tpc_importance)
rho_bat, p_bat = spearmanr(observed_pct, bat_importance)

print("=== Measurement-frequency vs. attribution importance (all 48 features) ===\n")
print(f"TPC (SHAP):  rho={rho_tpc:.4f}, p={p_tpc:.4g}")
print(f"BAT (IG):    rho={rho_bat:.4f}, p={p_bat:.4g}")

interpretation = (
    "positive: more-measured features tend to get MORE attribution" if rho_tpc > 0
    else "negative: more-measured features tend to get LESS attribution"
)
print(f"\nTPC direction: {interpretation}")
interpretation = (
    "positive: more-measured features tend to get MORE attribution" if rho_bat > 0
    else "negative: more-measured features tend to get LESS attribution"
)
print(f"BAT direction: {interpretation}")

# ---------------------------------------------------------------------------
# 4. Show the full table so specific features can be checked directly
# ---------------------------------------------------------------------------
results_df = pd.DataFrame({
    "feature": feature_names,
    "observed_pct": observed_pct,
    "tpc_shap_importance": tpc_importance,
    "bat_ig_importance_healthybaseline": bat_importance,
}).sort_values("observed_pct")

print("\n=== Sparsest 10 features: observed rate vs. attribution ===")
print(results_df.head(10).to_string(index=False))

print("\n=== Most-complete 10 features: observed rate vs. attribution ===")
print(results_df.tail(10).to_string(index=False))

results_df.to_csv("measurement_frequency_vs_attribution.csv", index=False)
print("\nSaved: measurement_frequency_vs_attribution.csv")