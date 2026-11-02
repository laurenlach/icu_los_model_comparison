
import pandas as pd
import numpy as np
# 1. Reuse the same underlying data BAT already built
X = np.load("bat_X.npy")           # (51835, 24, 48) - unscaled, will scale below
mask = np.load("bat_mask.npy")     # (51835, 24, 48) - real observed indicator
static = np.load("bat_static.npy") # (51835, 4) - unscaled, will scale below
y = np.load("bat_y.npy")           # (51835,)
stay_id_order = pd.read_csv("bat_stay_id_order.csv")

print("Loaded shapes:", X.shape, mask.shape, static.shape, y.shape)


# 2. Z-score normalize 
X_flat = X.reshape(-1, X.shape[-1])
mask_flat = mask.reshape(-1, mask.shape[-1])

means = np.zeros(X.shape[-1])
stds = np.zeros(X.shape[-1])

for i in range(X.shape[-1]):
    observed_values = X_flat[mask_flat[:, i] == 1, i]
    means[i] = observed_values.mean()
    stds[i] = observed_values.std()
    stds[i] = stds[i] if stds[i] > 1e-6 else 1.0

X_scaled = (X - means) / stds

observed_scaled_values = X_scaled[mask == 1]
print("\nScaling check (observed values only):")
print("Mean (~0 expected):", observed_scaled_values.mean())
print("Std (~1 expected):", observed_scaled_values.std())

static_means = static.mean(axis=0)
static_stds = static.std(axis=0)
static_stds = np.where(static_stds > 1e-6, static_stds, 1.0)
static_scaled = (static - static_means) / static_stds

# 3. Reuse BAT's split directly, rather than recomputing it 
train_rows = np.load("bat_train_rows.npy")
val_rows = np.load("bat_val_rows.npy")
test_rows = np.load("bat_test_rows.npy")

print("\nReusing BAT's exact split:")
print("Train:", len(train_rows), f"({len(train_rows)/len(y)*100:.1f}%)")
print("Val:", len(val_rows), f"({len(val_rows)/len(y)*100:.1f}%)")
print("Test:", len(test_rows), f"({len(test_rows)/len(y)*100:.1f}%)")

assert len(train_rows) + len(val_rows) + len(test_rows) == len(y), "Split sizes don't add up - stop and investigate."
print("\nSplit reused successfully - guaranteed identical to BAT's split.")

# 4. Save TPC's data
# X is currently (N, T, F) - transpose here to (N, F, T) for TPC's expected input
X_scaled_tpc = np.transpose(X_scaled, (0, 2, 1))
print("\nX_scaled_tpc shape (expect (51835, 48, 24)):", X_scaled_tpc.shape)

np.save("tpc_X_scaled.npy", X_scaled_tpc)
np.save("tpc_static_scaled.npy", static_scaled)
np.save("tpc_y.npy", y)
np.save("tpc_train_rows.npy", train_rows)
np.save("tpc_val_rows.npy", val_rows)
np.save("tpc_test_rows.npy", test_rows)

print("\nSaved: tpc_X_scaled.npy, tpc_static_scaled.npy, tpc_y.npy, tpc_train_rows.npy, tpc_val_rows.npy, tpc_test_rows.npy")
print("Done.")