import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

SOFA_SENSOR_INDICES = [27, 9, 15, 6, 39, 19, 35]  # map, bun, crea, bili, plt, fio2, o2sat
SOFA_SENSOR_NAMES = ["map", "bun", "crea", "bili", "plt", "fio2", "o2sat"]

mask = np.load("bat_mask.npy")            # (51835, 24, 48)
test_rows = np.load("bat_test_rows.npy")
mask_test = mask[test_rows]               # (10367, 24, 48)
bat_dangerous = np.load("bat_dangerous_mask_threshold_3_0days.npy")

# % of hours observed, per patient, averaged across the 7 SOFA sensors
sofa_observed_pct = mask_test[:, :, SOFA_SENSOR_INDICES].mean(axis=(1, 2)) * 100

df = pd.DataFrame({
    "dangerous": bat_dangerous,
    "sofa_observed_pct": sofa_observed_pct,
})

accurate = df[~df["dangerous"]]
dangerous = df[df["dangerous"]]

print("=== HYPOTHESIS C: Is measurement sparser for SOFA sensors in dangerous cases? ===\n")
print(f"Accurate  (n={len(accurate)}): mean %% observed on SOFA sensors = {accurate['sofa_observed_pct'].mean():.2f}%%")
print(f"Dangerous (n={len(dangerous)}): mean %% observed on SOFA sensors = {dangerous['sofa_observed_pct'].mean():.2f}%%")

stat, p = mannwhitneyu(dangerous["sofa_observed_pct"], accurate["sofa_observed_pct"])
print(f"Mann-Whitney U p={p:.4g}")

# Per-sensor breakdown, since one sensor's sparsity could hide in the average
print("\n=== Per-sensor observed rate ===")
for i, name in enumerate(SOFA_SENSOR_NAMES):
    idx = SOFA_SENSOR_INDICES[i]
    acc_pct = mask_test[~bat_dangerous, :, idx].mean() * 100
    dan_pct = mask_test[bat_dangerous, :, idx].mean() * 100
    print(f"{name}: accurate={acc_pct:.2f}%, dangerous={dan_pct:.2f}%")

df.to_csv("hypothesis_c_measurement_comparison.csv", index=False)
print("\nSaved: hypothesis_c_measurement_comparison.csv")