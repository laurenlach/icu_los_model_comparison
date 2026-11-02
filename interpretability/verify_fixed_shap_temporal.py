import numpy as np

tpc_shap_fixed = np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1)  # (10367, 48, 24)

mean_by_hour = np.abs(tpc_shap_fixed).mean(axis=(0, 1))

print("=== Mean |SHAP| by hour, FIXED model ===")
for h in range(24):
    print(f"hour {h}: {mean_by_hour[h]:.6f}")

n_nonzero_hours = np.sum(mean_by_hour > 1e-8)
print(f"\nHours with meaningfully nonzero attribution: {n_nonzero_hours} / 24")