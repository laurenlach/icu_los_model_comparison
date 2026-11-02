# import pandas as pd
# import numpy as np
# import polars as pl

# from icu_benchmarks.data.split_process_data import make_single_split
# from icu_benchmarks.data.constants import DataSplit as Split, DataSegment as Segment, VarType as Var
# from icu_benchmarks.constants import RunMode

# # ---------------------------------------------------------------------------
# # 1. Reuse the same underlying data BAT already built - same features,
# #    same cohort, ensuring a genuinely fair, identical comparison.
# # ---------------------------------------------------------------------------
# X = np.load("bat_X.npy")           # (51835, 24, 48) - unscaled, will scale below
# mask = np.load("bat_mask.npy")     # (51835, 24, 48) - real observed indicator
# static = np.load("bat_static.npy") # (51835, 4) - unscaled, will scale below
# y = np.load("bat_y.npy")           # (51835,)
# stay_id_order = pd.read_csv("bat_stay_id_order.csv")

# print("Loaded shapes:", X.shape, mask.shape, static.shape, y.shape)

# # ---------------------------------------------------------------------------
# # 2. Z-score normalize, fit on OBSERVED values only - same approach as BAT
# # ---------------------------------------------------------------------------
# X_flat = X.reshape(-1, X.shape[-1])
# mask_flat = mask.reshape(-1, mask.shape[-1])

# means = np.zeros(X.shape[-1])
# stds = np.zeros(X.shape[-1])

# for i in range(X.shape[-1]):
#     observed_values = X_flat[mask_flat[:, i] == 1, i]
#     means[i] = observed_values.mean()
#     stds[i] = observed_values.std()
#     stds[i] = stds[i] if stds[i] > 1e-6 else 1.0

# X_scaled = (X - means) / stds

# observed_scaled_values = X_scaled[mask == 1]
# print("\nScaling check (observed values only):")
# print("Mean (~0 expected):", observed_scaled_values.mean())
# print("Std (~1 expected):", observed_scaled_values.std())

# static_means = static.mean(axis=0)
# static_stds = static.std(axis=0)
# static_stds = np.where(static_stds > 1e-6, static_stds, 1.0)
# static_scaled = (static - static_means) / static_stds

# # ---------------------------------------------------------------------------
# # 3. Build the SAME split as BAT used (same seed, same fold structure) -
# #    critical for a fair comparison: both models see the same patients
# #    in train/val/test.
# # ---------------------------------------------------------------------------
# outcome_df = pl.DataFrame({
#     "stay_id": stay_id_order["stay_id"].to_numpy(),
#     "los": y,
# })
# id_only_df = pl.DataFrame({
#     "stay_id": stay_id_order["stay_id"].to_numpy(),
# })

# data = {
#     Segment.outcome: outcome_df,
#     Segment.static: id_only_df,
#     Segment.dynamic: id_only_df,
# }

# vars_dict = {
#     Var.group: "stay_id",
#     Var.label: "los",
# }

# # split_result = make_single_split(
# #     data=data,
# #     vars=vars_dict,
# #     cv_repetitions=1,
# #     repetition_index=0,
# #     cv_folds=4,
# #     fold_index=0,
# #     train_size=int(0.8 * len(y)),
# #     seed=42,  # SAME seed as BAT - ensures identical split
# #     debug=False,
# #     runmode=RunMode.regression,
# #     polars=True,
# # )

# # train_ids = split_result[Split.train][Segment.outcome]["stay_id"].to_numpy()
# # val_ids = split_result[Split.val][Segment.outcome]["stay_id"].to_numpy()
# # test_ids = split_result[Split.test][Segment.outcome]["stay_id"].to_numpy()

# # print("\nSplit sizes:")
# # print("Train:", len(train_ids), f"({len(train_ids)/len(y)*100:.1f}%)")
# # print("Val:", len(val_ids), f"({len(val_ids)/len(y)*100:.1f}%)")
# # print("Test:", len(test_ids), f"({len(test_ids)/len(y)*100:.1f}%)")




# # ---------------------------------------------------------------------------
# # 3. Reuse BAT's EXACT split directly, rather than recomputing - this
# #    guarantees identical patients in each set, avoiding a real bug in
# #    YAIB's make_single_split where random_state isn't applied when
# #    train_size is used (confirmed by inspecting the function's source).
# # ---------------------------------------------------------------------------
# train_rows = np.load("bat_train_rows.npy")
# val_rows = np.load("bat_val_rows.npy")
# test_rows = np.load("bat_test_rows.npy")

# print("\nReusing BAT's exact split:")
# print("Train:", len(train_rows), f"({len(train_rows)/len(y)*100:.1f}%)")
# print("Val:", len(val_rows), f"({len(val_rows)/len(y)*100:.1f}%)")
# print("Test:", len(test_rows), f"({len(test_rows)/len(y)*100:.1f}%)")

# # ---------------------------------------------------------------------------
# # 4. Verify this split is IDENTICAL to BAT's saved split - not just the
# #    same size, but the exact same patients in each set.
# # ---------------------------------------------------------------------------
# bat_train_rows = np.load("bat_train_rows.npy")
# stay_id_to_row = {sid: i for i, sid in enumerate(stay_id_order["stay_id"])}
# bat_train_ids = set(stay_id_order["stay_id"].iloc[bat_train_rows])
# tpc_train_ids = set(train_ids)

# print("\nSplit identity check vs BAT's split:")
# print("Train sets identical:", bat_train_ids == tpc_train_ids)

# # ---------------------------------------------------------------------------
# # 5. Save TPC's data - note: NO transpose needed (TPC's forward() already
# #    expects channels-first (B, F, T), unlike BAT which needed (B, F, T)
# #    via an explicit transpose step from our original (B, T, F) build).
# #    NO mask needed either - not part of TPC's architecture.
# # ---------------------------------------------------------------------------
# train_rows = np.array([stay_id_to_row[sid] for sid in train_ids])
# val_rows = np.array([stay_id_to_row[sid] for sid in val_ids])
# test_rows = np.array([stay_id_to_row[sid] for sid in test_ids])

# # X is currently (N, T, F) - transpose ONCE here to (N, F, T) for TPC's expected input
# X_scaled_tpc = np.transpose(X_scaled, (0, 2, 1))
# print("\nX_scaled_tpc shape (expect (51835, 48, 24)):", X_scaled_tpc.shape)

# np.save("tpc_X_scaled.npy", X_scaled_tpc)
# np.save("tpc_static_scaled.npy", static_scaled)
# np.save("tpc_y.npy", y)
# np.save("tpc_train_rows.npy", train_rows)
# np.save("tpc_val_rows.npy", val_rows)
# np.save("tpc_test_rows.npy", test_rows)

# print("\nSaved: tpc_X_scaled.npy, tpc_static_scaled.npy, tpc_y.npy, tpc_train_rows.npy, tpc_val_rows.npy, tpc_test_rows.npy")
# print("Done.")







import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 1. Reuse the same underlying data BAT already built - same features,
#    same cohort, ensuring a genuinely fair, identical comparison.
# ---------------------------------------------------------------------------
X = np.load("bat_X.npy")           # (51835, 24, 48) - unscaled, will scale below
mask = np.load("bat_mask.npy")     # (51835, 24, 48) - real observed indicator
static = np.load("bat_static.npy") # (51835, 4) - unscaled, will scale below
y = np.load("bat_y.npy")           # (51835,)
stay_id_order = pd.read_csv("bat_stay_id_order.csv")

print("Loaded shapes:", X.shape, mask.shape, static.shape, y.shape)

# ---------------------------------------------------------------------------
# 2. Z-score normalize, fit on OBSERVED values only - same approach as BAT
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 3. Reuse BAT's EXACT split directly, rather than recomputing it.
#    This guarantees identical patients in train/val/test for both models -
#    critical for a fair comparison. Recomputing via make_single_split
#    again was tried and found NOT to reproduce the same split even with
#    the same seed: YAIB's make_single_split has a real bug where, when
#    train_size is provided, it calls ShuffleSplit(cv_repetitions,
#    train_size=train_size) WITHOUT passing random_state=seed - so the
#    seed argument is silently ignored on that code path. Reusing BAT's
#    already-saved split avoids this entirely.
# ---------------------------------------------------------------------------
train_rows = np.load("bat_train_rows.npy")
val_rows = np.load("bat_val_rows.npy")
test_rows = np.load("bat_test_rows.npy")

print("\nReusing BAT's exact split:")
print("Train:", len(train_rows), f"({len(train_rows)/len(y)*100:.1f}%)")
print("Val:", len(val_rows), f"({len(val_rows)/len(y)*100:.1f}%)")
print("Test:", len(test_rows), f"({len(test_rows)/len(y)*100:.1f}%)")

assert len(train_rows) + len(val_rows) + len(test_rows) == len(y), "Split sizes don't add up - stop and investigate."
print("\nSplit reused successfully - guaranteed identical to BAT's split.")

# ---------------------------------------------------------------------------
# 4. Save TPC's data - note: NO transpose needed the way BAT needed it
#    (TPC's forward() already expects channels-first (B, F, T)).
#    NO mask needed either - not part of TPC's architecture.
# ---------------------------------------------------------------------------
# X is currently (N, T, F) - transpose ONCE here to (N, F, T) for TPC's expected input
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