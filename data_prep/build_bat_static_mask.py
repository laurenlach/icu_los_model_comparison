import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 1. Load everything we need, in the SAME patient order as bat_X.npy/bat_y.npy
# ---------------------------------------------------------------------------
stay_id_order = pd.read_csv("bat_stay_id_order.csv")
sta_los = pd.read_parquet("bat_sta_with_los.parquet")

print("stay_id_order rows:", len(stay_id_order))
print("sta_los rows:", len(sta_los))

# Merge, preserving stay_id_order's row order exactly (how="left", no sorting)
static_ordered = stay_id_order.merge(sta_los, on="stay_id", how="left")
assert len(static_ordered) == len(stay_id_order), "Row count changed during merge - stop and check for duplicate stay_ids."
assert static_ordered["age"].isna().sum() == 0, "Missing age values after merge - stop and investigate."

print("\nMerged static data preview:")
print(static_ordered.head())

# ---------------------------------------------------------------------------
# 2. Encode sex as numeric (0/1), handle missing height/weight
# ---------------------------------------------------------------------------
static_ordered["sex_encoded"] = (static_ordered["sex"] == "Male").astype(float)

print("\nMissing values before fill:")
print("height:", static_ordered["height"].isna().sum())
print("weight:", static_ordered["weight"].isna().sum())

# Fill missing height/weight with population median (same approach as sensor data)
static_ordered["height"] = static_ordered["height"].fillna(static_ordered["height"].median())
static_ordered["weight"] = static_ordered["weight"].fillna(static_ordered["weight"].median())

# ---------------------------------------------------------------------------
# 3. Build the final (N, 4) static array: age, sex, height, weight
# ---------------------------------------------------------------------------
static_array = static_ordered[["age", "sex_encoded", "height", "weight"]].to_numpy(dtype=np.float32)
print("\nFinal static array shape:", static_array.shape)
print("Any NaN present:", np.isnan(static_array).any())

np.save("bat_static.npy", static_array)
print("\nSaved bat_static.npy")