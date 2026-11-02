import numpy as np

X = np.load("bat_X.npy")
mask = np.load("bat_mask.npy")

n_features = X.shape[-1]
X_flat = X.reshape(-1, n_features)
mask_flat = mask.reshape(-1, n_features)

feature_means = np.zeros(n_features)
feature_stds = np.zeros(n_features)

for i in range(n_features):
    observed_values = X_flat[mask_flat[:, i] == 1, i]
    feature_means[i] = observed_values.mean()
    feature_stds[i] = observed_values.std()
    feature_stds[i] = feature_stds[i] if feature_stds[i] > 1e-6 else 1.0

np.save("bat_feature_means.npy", feature_means)
np.save("bat_feature_stds.npy", feature_stds)
print("Saved: bat_feature_means.npy, bat_feature_stds.npy (recomputed on FULL dataset, matching the original script)")

# --- Validation, same check as before ---
X_scaled_actual = np.load("bat_X_scaled.npy")
X_scaled_ours = (X - feature_means) / feature_stds

max_diff = np.abs(X_scaled_ours - X_scaled_actual).max()
print(f"\nValidation: max difference across ALL patients: {max_diff:.8f}")

if max_diff < 0.001:
    print("PASS — our recomputed mean/std now correctly reproduces the original scaling.")
else:
    print("Still mismatched — needs further investigation.")