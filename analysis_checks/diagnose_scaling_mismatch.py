import numpy as np

X = np.load("bat_X.npy")
mask = np.load("bat_mask.npy")
train_rows = np.load("bat_train_rows.npy")
X_scaled_actual = np.load("bat_X_scaled.npy")
feature_means = np.load("bat_feature_means.npy")
feature_stds = np.load("bat_feature_stds.npy")

feature_names = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun",
    "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2",
    "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
    "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt",
    "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine", "wbc"
]

# Check 1: does the mismatch spread evenly across all features, or concentrate in just a few? 
X_train_raw = X[train_rows]
our_scaled_train = (X_train_raw - feature_means) / feature_stds
actual_scaled_train = X_scaled_actual[train_rows]

per_feature_max_diff = np.abs(our_scaled_train - actual_scaled_train).max(axis=(0, 1))

print("=== Per-feature max mismatch (sorted, worst first) ===")
sorted_idx = np.argsort(-per_feature_max_diff)
for i in sorted_idx[:10]:
    print(f"{feature_names[i]}: max_diff={per_feature_max_diff[i]:.6f}  "
          f"(our_std={feature_stds[i]:.4f})")

print(f"\nOverall: {np.mean(per_feature_max_diff < 0.01)*100:.1f}% of features "
      f"have max_diff < 0.01")
print(f"Median max_diff across features: {np.median(per_feature_max_diff):.6f}")

# Check 2: does using sample std (ddof=1) instead of population std (ddof=0, numpy's default, what was used) close the gap?            ---
f = sorted_idx[0]  # worst-offending feature
observed_vals = X_train_raw[:, :, f][mask[train_rows][:, :, f] == 1]
std_pop = observed_vals.std(ddof=0)   # what we used
std_sample = observed_vals.std(ddof=1)  # alternative
print(f"\nWorst feature ({feature_names[f]}): "
      f"population std={std_pop:.4f}, sample std={std_sample:.4f}, "
      f"our saved std={feature_stds[f]:.4f}")

# Check 3: is this float32 precision noise? Check dtype 
print(f"\nbat_X.npy dtype: {X.dtype}")
print(f"bat_X_scaled.npy dtype: {X_scaled_actual.dtype}")