import numpy as np
import pandas as pd

tpc_shap = np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1)
bat_ig = np.load("bat_ig_values_full_testset_healthybaseline.npy")

tpc_by_hour = np.abs(tpc_shap).mean(axis=(0, 1))
bat_by_hour = np.abs(bat_ig).mean(axis=(0, 1))

def normalized_entropy(values):
    p = values / values.sum()
    p = p[p > 0]
    return -np.sum(p * np.log(p)) / np.log(len(values))

tpc_entropy = normalized_entropy(tpc_by_hour)
bat_entropy = normalized_entropy(bat_by_hour)

print(f"TPC (fixed) temporal entropy: {tpc_entropy:.4f}")
print(f"BAT temporal entropy: {bat_entropy:.4f}")
print("(Old buggy TPC temporal entropy was -0.0000 - all mass on hour 23)")

early_tpc = tpc_by_hour[0:8].mean()
late_tpc = tpc_by_hour[16:24].mean()
print(f"\nTPC (fixed) early(0-7) vs late(16-23): {early_tpc:.5f} vs {late_tpc:.5f}, "
      f"ratio={late_tpc/early_tpc if early_tpc>0 else float('inf'):.2f}")

pd.DataFrame({"hour": range(24), "tpc_fixed": tpc_by_hour, "bat": bat_by_hour}).to_csv(
    "temporal_feature_importance_fixed.csv", index=False
)