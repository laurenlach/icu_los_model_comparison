import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Load attribution arrays - already in (patients, features, hours) shape
# ---------------------------------------------------------------------------
tpc_shap = np.load("tpc_shap_values_full_testset.npy").squeeze(-1)   # (10367, 48, 24)
bat_ig = np.load("bat_ig_values_full_testset_healthybaseline.npy")   # (10367, 48, 24)

# ---------------------------------------------------------------------------
# 2. Mean |attribution| per HOUR, averaged across all patients and all 48
#    features - collapses the feature dimension instead of the time dimension,
#    the mirror image of the global feature ranking from Step 1
# ---------------------------------------------------------------------------
tpc_by_hour = np.abs(tpc_shap).mean(axis=(0, 1))   # (24,)
bat_by_hour = np.abs(bat_ig).mean(axis=(0, 1))     # (24,)

print("=== Mean |attribution| by hour (0 = first hour of admission, 23 = last) ===\n")
print(f"{'Hour':<6}{'TPC (SHAP)':<15}{'BAT (IG)':<15}")
for h in range(24):
    print(f"{h:<6}{tpc_by_hour[h]:<15.5f}{bat_by_hour[h]:<15.5f}")

# ---------------------------------------------------------------------------
# 3. Normalized entropy of the hourly distribution - same formula used for
#    Phase E's attention-entropy check and Step 1's feature-entropy check,
#    now applied to the TIME dimension of actual attribution
# ---------------------------------------------------------------------------
def normalized_entropy(values):
    p = values / values.sum()
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return entropy / np.log(len(values))

tpc_entropy = normalized_entropy(tpc_by_hour)
bat_entropy = normalized_entropy(bat_by_hour)

print(f"\n=== Temporal concentration (normalized entropy, 0=concentrated on one hour, 1=uniform) ===")
print(f"TPC (SHAP) attribution:  {tpc_entropy:.4f}")
print(f"BAT (IG) attribution:    {bat_entropy:.4f}")
print(f"(For comparison: BAT's raw ATTENTION temporal entropy from Phase E was 0.8755)")

# ---------------------------------------------------------------------------
# 4. Early (hours 0-7) vs. late (hours 16-23) window comparison - a simple,
#    interpretable summary of whether either model leans toward early warning
#    signal or just the most recent readings
# ---------------------------------------------------------------------------
early_tpc = tpc_by_hour[0:8].mean()
late_tpc = tpc_by_hour[16:24].mean()
early_bat = bat_by_hour[0:8].mean()
late_bat = bat_by_hour[16:24].mean()

print(f"\n=== Early (hours 0-7) vs. Late (hours 16-23) mean attribution ===")
print(f"TPC: early={early_tpc:.5f}, late={late_tpc:.5f}, "
      f"ratio late/early={late_tpc/early_tpc:.2f}")
print(f"BAT: early={early_bat:.5f}, late={late_bat:.5f}, "
      f"ratio late/early={late_bat/early_bat:.2f}")

results_df = pd.DataFrame({
    "hour": range(24),
    "tpc_mean_attribution": tpc_by_hour,
    "bat_mean_attribution": bat_by_hour,
})
results_df.to_csv("temporal_feature_importance.csv", index=False)
print("\nSaved: temporal_feature_importance.csv")