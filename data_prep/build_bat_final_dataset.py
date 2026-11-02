import pandas as pd
import numpy as np
import polars as pl

from icu_benchmarks.data.split_process_data import make_single_split
from icu_benchmarks.data.constants import DataSplit as Split, DataSegment as Segment, VarType as Var
from icu_benchmarks.constants import RunMode

# 1. Load already-built data
X = np.load("bat_X.npy")           # (51835, 24, 48)
mask = np.load("bat_mask.npy")     # (51835, 24, 48)
static = np.load("bat_static.npy") # (51835, 4)
y = np.load("bat_y.npy")           # (51835,)
stay_id_order = pd.read_csv("bat_stay_id_order.csv")

print("Loaded shapes:", X.shape, mask.shape, static.shape, y.shape)

# 2. Z-score normalize X and static 
X_flat = X.reshape(-1, X.shape[-1])
mask_flat = mask.reshape(-1, mask.shape[-1])

means = np.zeros(X.shape[-1])
stds = np.zeros(X.shape[-1])

for i in range(X.shape[-1]):
    observed_values = X_flat[mask_flat[:, i] == 1, i]
    means[i] = observed_values.mean()
    stds[i] = observed_values.std()
    stds[i] = stds[i] if stds[i] > 1e-6 else 1.0  # avoid divide-by-zero for near-constant columns

X_scaled = (X - means) / stds

observed_scaled_values = X_scaled[mask == 1]
print("\nScaling check (observed values only):")
print("Mean (~0 expected):", observed_scaled_values.mean())
print("Std (~1 expected):", observed_scaled_values.std())

static_means = static.mean(axis=0)
static_stds = static.std(axis=0)
static_stds = np.where(static_stds > 1e-6, static_stds, 1.0)
static_scaled = (static - static_means) / static_stds

print("\nStatic scaling check:")
print("Mean per feature (~0 expected):", static_scaled.mean(axis=0))
print("Std per feature (~1 expected):", static_scaled.std(axis=0))

# 3. Build the data dictionary make_single_split expects
outcome_df = pl.DataFrame({
    "stay_id": stay_id_order["stay_id"].to_numpy(),
    "los": y,
})
id_only_df = pl.DataFrame({
    "stay_id": stay_id_order["stay_id"].to_numpy(),
})

data = {
    Segment.outcome: outcome_df,
    Segment.static: id_only_df,
    Segment.dynamic: id_only_df,
}

vars_dict = {
    Var.group: "stay_id",
    Var.label: "los",
}

# 4. Run YAIB's own split function 
split_result = make_single_split(
    data=data,
    vars=vars_dict,
    cv_repetitions=1,
    repetition_index=0,
    cv_folds=4,
    fold_index=0,
    train_size=int(0.8 * len(y)),
    seed=42,
    debug=False,
    runmode=RunMode.regression,
    polars=True,
)

train_ids = split_result[Split.train][Segment.outcome]["stay_id"].to_numpy()
val_ids = split_result[Split.val][Segment.outcome]["stay_id"].to_numpy()
test_ids = split_result[Split.test][Segment.outcome]["stay_id"].to_numpy()

print("\nSplit sizes:")
print("Train:", len(train_ids), f"({len(train_ids)/len(y)*100:.1f}%)")
print("Val:", len(val_ids), f"({len(val_ids)/len(y)*100:.1f}%)")
print("Test:", len(test_ids), f"({len(test_ids)/len(y)*100:.1f}%)")

assert len(train_ids) + len(val_ids) + len(test_ids) == len(y), "Split sizes don't add up to total - stop and investigate."
assert len(set(train_ids) & set(val_ids)) == 0, "Train/val overlap detected - stop and investigate."
assert len(set(train_ids) & set(test_ids)) == 0, "Train/test overlap detected - stop and investigate."
assert len(set(val_ids) & set(test_ids)) == 0, "Val/test overlap detected - stop and investigate."
print("\nNo overlap between splits - confirmed clean.")

# 5. Save everything needed for training
stay_id_to_row = {sid: i for i, sid in enumerate(stay_id_order["stay_id"])}
train_rows = [stay_id_to_row[sid] for sid in train_ids]
val_rows = [stay_id_to_row[sid] for sid in val_ids]
test_rows = [stay_id_to_row[sid] for sid in test_ids]

np.save("bat_X_scaled.npy", X_scaled)
np.save("bat_static_scaled.npy", static_scaled)
np.save("bat_train_rows.npy", np.array(train_rows))
np.save("bat_val_rows.npy", np.array(val_rows))
np.save("bat_test_rows.npy", np.array(test_rows))

print("\nSaved: bat_X_scaled.npy, bat_static_scaled.npy, bat_train_rows.npy, bat_val_rows.npy, bat_test_rows.npy")
print("Done.")