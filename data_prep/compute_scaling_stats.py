import numpy as np

# --- Load raw (unscaled) data, mask, and training split ---
X = np.load("bat_X.npy")             # shape: (51835, 24, 48) - raw clinical values
mask = np.load("bat_mask.npy")       # shape: (51835, 24, 48) - 1=observed, 0=filled/padded
train_rows = np.load("bat_train_rows.npy")  # indices of training patients

print("X shape:", X.shape)
print("mask shape:", mask.shape)
print("Number of training patients:", len(train_rows))

# --- Subset to training patients only (scaling must never use val/test data) ---
X_train = X[train_rows]        # shape: (n_train, 24, 48)
mask_train = mask[train_rows]  # shape: (n_train, 24, 48)

# --- Compute mean/std per feature, using ONLY genuinely-observed values ---
# (i.e. excluding forward-filled/padded timesteps, per your original preprocessing logic)
n_features = X_train.shape[2]
feature_means = np.zeros(n_features)
feature_stds = np.zeros(n_features)

for f in range(n_features):
    observed_values = X_train[:, :, f][mask_train[:, :, f] == 1]
    feature_means[f] = observed_values.mean()
    feature_stds[f] = observed_values.std()
    if feature_stds[f] == 0:
        print(f"WARNING: feature {f} has zero std — check this feature before continuing")

print("\nComputed means (first 5):", feature_means[:5])
print("Computed stds (first 5):", feature_stds[:5])

np.save("bat_feature_means.npy", feature_means)
np.save("bat_feature_stds.npy", feature_stds)
print("\nSaved: bat_feature_means.npy, bat_feature_stds.npy")

# ---------------------------------------------------------------------------
# VALIDATION: apply our recomputed mean/std to a real patient's raw data,
# and confirm it matches bat_X_scaled.npy for that same patient.
# If this doesn't match closely, STOP — something about our recomputation
# doesn't match the original scaling logic, and the healthy baseline would
# be built on wrong numbers.
# ---------------------------------------------------------------------------
X_scaled_actual = np.load("bat_X_scaled.npy")

test_patient_idx = train_rows[0]  # a real training patient, known to be scaled originally
raw_patient = X[test_patient_idx]              # (24, 48)
our_scaled_patient = (raw_patient - feature_means) / feature_stds
actual_scaled_patient = X_scaled_actual[test_patient_idx]  # (24, 48)

max_diff = np.abs(our_scaled_patient - actual_scaled_patient).max()
print(f"\nValidation: max difference between our recomputed scaling and the "
      f"actual bat_X_scaled.npy, for one real patient: {max_diff:.6f}")

if max_diff < 0.01:
    print("PASS — our recomputed mean/std correctly reproduces the original scaling.")
else:
    print("FAIL — mismatch detected. Do not proceed to build the healthy baseline "
          "until this is resolved (likely a masking or train-split difference).")