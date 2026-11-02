import numpy as np
import pandas as pd
from scipy.stats import spearmanr

X = np.load("tpc_X_scaled.npy")
rows = np.load("tpc_test_rows.npy")
y = np.load("tpc_test_targets_fixed.npy")
X_test = X[rows]                                          # (n, 48, 24)
mask = np.load("bat_mask.npy")[np.load("bat_test_rows.npy")]   # (n, 24, 48), 1 = measured

total_meas = mask.sum(axis=(1, 2))                        # care-intensity proxy: all measurements in 24 h
print(f"Overall: rho(total measurements, true LOS) = {spearmanr(total_meas, y)[0]:+.3f}\n")

for name, idx in [("o2sat", 35), ("map", 27)]:
    v = X_test[:, idx, :]                                 # (n, 24)
    bins = pd.qcut(v.mean(axis=1), 5, labels=False, duplicates="drop") + 1
    df = pd.DataFrame({
        "bin": bins, "los": y, "pct_under_1day": (y < 1).astype(float) * 100,
        "total_measurements": total_meas,
        "hours_this_sensor_measured": mask[:, :, idx].sum(axis=1),
        "min_z": v.min(axis=1), "max_z": v.max(axis=1), "last_hour_z": v[:, -1],
    })
    out = df.groupby("bin").agg(
        n=("los", "size"), median_los=("los", "median"), pct_under_1day=("pct_under_1day", "mean"),
        median_total_meas=("total_measurements", "median"),
        median_hours_sensor=("hours_this_sensor_measured", "median"),
        median_min_z=("min_z", "median"), median_max_z=("max_z", "median"),
        median_last_hour_z=("last_hour_z", "median"),
    ).round(3)
    print(f"=== {name}: five groups by 24-hour mean reading (1 = lowest) ===")
    print(out.to_string(), "\n")