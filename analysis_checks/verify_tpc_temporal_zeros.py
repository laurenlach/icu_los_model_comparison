import numpy as np

tpc_shap = np.load("tpc_shap_values_full_testset.npy").squeeze(-1)  # (10367, 48, 24)

print("Shape:", tpc_shap.shape)
print("Overall min/max/mean (raw, signed, not abs):", tpc_shap.min(), tpc_shap.max(), tpc_shap.mean())

print("\n=== Patient 0, feature 'o2sat' (index 35), all 24 hours, full precision ===")
for h in range(24):
    print(f"  hour {h}: {tpc_shap[0, 35, h]:.10f}")

print("\n=== Are ALL values at hours 0-22 EXACTLY zero, across all patients/features? ===")
early_hours = tpc_shap[:, :, 0:23]
print("Count of exactly-zero values in hours 0-22:", np.sum(early_hours == 0.0),
      "out of", early_hours.size, "total")
print("Count of NONZERO values in hours 0-22:", np.sum(early_hours != 0.0))
print("Max absolute value in hours 0-22 (should be 0 if truly all-zero):", np.abs(early_hours).max())