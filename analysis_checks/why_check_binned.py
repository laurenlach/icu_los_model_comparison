import numpy as np
import pandas as pd

X = np.load("tpc_X_scaled.npy")
rows = np.load("tpc_test_rows.npy")
y = np.load("tpc_test_targets_fixed.npy")
X_test = X[rows]
mask = np.load("bat_mask.npy")[np.load("bat_test_rows.npy")]     # (n, 24, 48), 1 = really measured

fio2_measured = mask[:, :, 19].sum(axis=1) >= 1
fio2_z = X_test[:, 19, :].mean(axis=1)
fio2_z_measured_only = np.where(fio2_measured, fio2_z, np.nan)   # imputed values excluded

for name, idx in [("o2sat", 35), ("map", 27)]:
    v = X_test[:, idx, :].mean(axis=1)
    df = pd.DataFrame({
        "bin": pd.qcut(v, 5, labels=False, duplicates="drop") + 1,   # 1 = lowest reading, 5 = highest
        "value_z": v, "los": y,
        "fio2_measured": fio2_measured.astype(float) * 100,
        "fio2_z_measured_only": fio2_z_measured_only,
    })
    out = df.groupby("bin").agg(
        n=("los", "size"),
        mean_reading_z=("value_z", "mean"),
        median_los=("los", "median"),
        mean_los=("los", "mean"),
        pct_fio2_measured=("fio2_measured", "mean"),
        mean_fio2_z_if_measured=("fio2_z_measured_only", "mean"),
    ).round(3)
    print(f"\n=== {name}: five equal groups by 24-hour mean reading (1 = lowest, 5 = highest) ===")
    print(out.to_string())