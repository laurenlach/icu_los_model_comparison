
import pandas as pd
import numpy as np

# 1. Load 
dyn = pd.read_parquet(r"..\YAIB-cohorts-main\YAIB-cohorts-main\data\los\mimic\dyn.parquet")
sta_los = pd.read_parquet("bat_sta_with_los.parquet")

print("Dynamic data shape:", dyn.shape)
print("Static+LOS data shape:", sta_los.shape)

# 2. Prep time column and sort (required before forward-filling)
dyn["time"] = pd.to_timedelta(dyn["time"])
dyn = dyn.sort_values(["stay_id", "time"]).reset_index(drop=True)

# Sensor columns = everything except the two ID/time columns
sensor_cols = [c for c in dyn.columns if c not in ("stay_id", "time")]
print("\nNumber of sensor columns:", len(sensor_cols))

# 3. Track which values are genuinely observed, BEFORE any filling happens 
mask_df = dyn[sensor_cols].notna().astype(np.float32)
mask_df["stay_id"] = dyn["stay_id"].values
mask_df["time"] = dyn["time"].values

# 4. Forward-fill missing values within each patient's own timeline
#    (carries the last known reading forward, never uses future info)
print("\nForward-filling missing values per patient...")
dyn[sensor_cols] = dyn.groupby("stay_id")[sensor_cols].ffill()

nan_after_ffill = dyn[sensor_cols].isna().sum().sum()
print("NaN count after forward-fill:", nan_after_ffill)


# 5. Any NaNs still remaining (start of a stay, before that variable was
#    ever measured) get filled using the overall population median.
medians = dyn[sensor_cols].median()
dyn[sensor_cols] = dyn[sensor_cols].fillna(medians)

nan_after_median = dyn[sensor_cols].isna().sum().sum()
print("NaN count after median fill (should be 0):", nan_after_median)


# 6. Keep only the first 24 hours per patient - for BOTH the data and the mask

first_24h = dyn[dyn["time"] < pd.Timedelta(hours=24)]
mask_24h = mask_df[mask_df["time"] < pd.Timedelta(hours=24)]

print("\nAfter filtering to first 24h:", first_24h.shape)
assert len(first_24h) == len(mask_24h), "Data and mask row counts diverged - stop and investigate."

hours_per_patient = first_24h.groupby("stay_id").size()
print("\nHours per patient - min/max/mean:")
print(hours_per_patient.min(), hours_per_patient.max(), hours_per_patient.mean())

n_unique = first_24h["stay_id"].nunique()
print("\nUnique patients in first_24h:", n_unique)
print("Total patients expected:", sta_los.shape[0])
assert n_unique == sta_los.shape[0], "Patient count mismatch - stop and investigate before continuing."


# 7. Build one padded (24, 48) array per patient - for data AND mask together,
#    using the SAME per-patient grouping so they can never fall out of sync.

def build_patient_tensor(group, cols):
    values = group[cols].to_numpy()
    actual_hours = values.shape[0]
    if actual_hours < 24:
        padding = np.zeros((24 - actual_hours, len(cols)))
        values = np.vstack([values, padding])
    return values[:24]


first_patient_id = first_24h["stay_id"].iloc[0]
test_group_data = first_24h[first_24h["stay_id"] == first_patient_id]
test_group_mask = mask_24h[mask_24h["stay_id"] == first_patient_id]
test_tensor = build_patient_tensor(test_group_data, sensor_cols)
test_mask = build_patient_tensor(test_group_mask, sensor_cols)
print("\nTest patient tensor shape (expect (24, 48)):", test_tensor.shape)
print("Test patient mask shape (expect (24, 48)):", test_mask.shape)
assert test_tensor.shape == (24, len(sensor_cols)), "Tensor shape check failed - stop and investigate."
assert test_mask.shape == (24, len(sensor_cols)), "Mask shape check failed - stop and investigate."


# 8. Build tensors for every patient 
print("\nBuilding tensors for all patients... (this will take a minute or two)")

all_tensors = []
all_masks = []
all_stay_ids = []

data_groups = dict(tuple(first_24h.groupby("stay_id")))
mask_groups = dict(tuple(mask_24h.groupby("stay_id")))

for stay_id in data_groups.keys():
    tensor = build_patient_tensor(data_groups[stay_id], sensor_cols)
    mask = build_patient_tensor(mask_groups[stay_id], sensor_cols)
    all_tensors.append(tensor)
    all_masks.append(mask)
    all_stay_ids.append(stay_id)

X = np.stack(all_tensors)      # shape: (51835, 24, 48)
MASK = np.stack(all_masks)     # shape: (51835, 24, 48)

print("Final X shape:", X.shape)
print("Final MASK shape:", MASK.shape)
print("Any NaN values present in X:", np.isnan(X).any())
print("Total NaN count in X:", np.isnan(X).sum(), "out of", X.size, "values")
print("Fraction of MASK that is 1 (observed):", MASK.mean())


# 9. Build the matching y (LOS target) array, in the SAME patient order as X
stay_id_order = pd.DataFrame({"stay_id": all_stay_ids})

y_df = stay_id_order.merge(sta_los[["stay_id", "los"]], on="stay_id", how="left")
assert y_df["los"].isna().sum() == 0, "Some patients are missing an LOS label - stop and investigate."

y = y_df["los"].to_numpy()
print("\nFinal y shape:", y.shape)
print("y sample values:", y[:5])

# 10. Save everything to disk for the next step (model training)
np.save("bat_X.npy", X)
np.save("bat_mask.npy", MASK)
np.save("bat_y.npy", y)
stay_id_order.to_csv("bat_stay_id_order.csv", index=False)

print("\nSaved: bat_X.npy, bat_mask.npy, bat_y.npy, bat_stay_id_order.csv")
print("Done.")