import numpy as np
import pandas as pd

# --- Load the full cohort's patient ID order and the test-split indices ---
stay_id_order = pd.read_csv("bat_stay_id_order.csv")  # full 51,835-patient order
test_rows = np.load("bat_test_rows.npy")               # indices into that order for the test split

test_stay_ids = stay_id_order.iloc[test_rows].reset_index(drop=True)
id_col = test_stay_ids.columns[0]  # whatever the single ID column is actually called
test_stay_ids = test_stay_ids.rename(columns={id_col: "stay_id"})

print(f"Test split size from indices: {len(test_rows)}")
print(f"Test stay_id table size: {len(test_stay_ids)}")

# --- Load predictions/targets for both models ---
bat_pred = np.load("bat_test_predictions.npy")
bat_true = np.load("bat_test_targets.npy")

tpc_pred = np.load("tpc_test_predictions_200epoch.npy")
tpc_true = np.load("tpc_test_targets_200epoch.npy")

# Sanity checks before trusting any of this
assert len(bat_pred) == len(test_stay_ids), \
    f"BAT prediction count ({len(bat_pred)}) != test stay_id count ({len(test_stay_ids)})"
assert len(tpc_pred) == len(test_stay_ids), \
    f"TPC prediction count ({len(tpc_pred)}) != test stay_id count ({len(test_stay_ids)})"
assert np.allclose(bat_true, tpc_true), \
    "BAT and TPC targets differ — they should be identical since both reuse the same test split"

phenotype_map = pd.read_csv("cohort_icd9_phenotype_mapping_with_fallback.csv")

DANGEROUS_THRESHOLD = -3.0  # matches Phase D's dangerous-case definition

# --- Build one long table: stay_id, model, error, phenotype ---
records = []
for model_name, pred, true in [("BAT", bat_pred, bat_true), ("TPC", tpc_pred, tpc_true)]:
    error = pred.flatten() - true.flatten()  # predicted - true, matches Phase D's sign convention
    df = pd.DataFrame({
        "stay_id": test_stay_ids["stay_id"].values,
        "model": model_name,
        "error": error,
    })
    records.append(df)

perf = pd.concat(records, ignore_index=True)
perf = perf.merge(phenotype_map, on="stay_id", how="inner")

print(f"\nMatched {len(perf)} of {2 * len(test_stay_ids)} expected rows to a phenotype "
      f"({perf['stay_id'].nunique()} unique patients)")

perf.to_csv("per_patient_error_with_phenotype.csv", index=False)

# --- Aggregate to per-phenotype, per-model performance ---
performance_summary = (
    perf.groupby(["phenotype", "model"])
    .agg(
        n_patients=("error", "size"),
        mae=("error", lambda x: x.abs().mean()),
        mean_error=("error", "mean"),
        dangerous_case_pct=("error", lambda x: (x <= DANGEROUS_THRESHOLD).mean() * 100),
    )
    .reset_index()
)

performance_summary.to_csv("phenotype_performance_summary.csv", index=False)

print("\nPerformance summary preview:")
print(performance_summary.sort_values(["phenotype", "model"]).to_string(index=False))