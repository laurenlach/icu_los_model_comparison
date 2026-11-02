import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

tpc_error = np.load("tpc_error.npy")
bat_error = np.load("bat_error.npy")
tpc_test_rows = np.load("tpc_test_rows.npy")
bat_test_rows = np.load("bat_test_rows.npy")

assert len(tpc_error) == len(bat_error), "Error array lengths differ - stop and investigate."
assert np.array_equal(tpc_test_rows, bat_test_rows), (
    "TPC and BAT test_rows are NOT identical - they are not directly comparable "
    "row-by-row. Stop and merge via stay_id instead before continuing."
)
print(f"Confirmed: TPC and BAT test_rows are identical ({len(tpc_test_rows)} patients) - "
      f"safe to compare tpc_error.npy and bat_error.npy by row position.")


stay_id_order = pd.read_csv("bat_stay_id_order.csv")
row_stay_ids = stay_id_order["stay_id"].iloc[bat_test_rows].values

assert len(row_stay_ids) == len(tpc_error), "Stay ID count mismatch - stop and investigate."


phenotype_map = pd.read_csv("cohort_icd9_phenotype_mapping_with_fallback.csv")

df = pd.DataFrame({
    "stay_id": row_stay_ids,
    "tpc_abs_error": np.abs(tpc_error),
    "bat_abs_error": np.abs(bat_error),
})
df = df.merge(phenotype_map, on="stay_id", how="left")

n_unmatched = df["phenotype"].isna().sum()
print(f"\nPatients matched to a phenotype: {len(df) - n_unmatched} / {len(df)} "
      f"({n_unmatched} unmatched, consistent with the ~94.3% fallback coverage)")

df = df.dropna(subset=["phenotype"])


results = []
for phenotype, group in df.groupby("phenotype"):
    n = len(group)
    tpc_mae = group["tpc_abs_error"].mean()
    bat_mae = group["bat_abs_error"].mean()

    if n < 10:
        stat, pval = np.nan, np.nan
        note = "n<10, test not run"
    else:
        try:
            stat, pval = wilcoxon(group["tpc_abs_error"], group["bat_abs_error"])
            note = "significant" if pval < 0.05 else "not significant"
        except ValueError:
            # Wilcoxon fails if all differences are exactly zero - rare, but handle it
            stat, pval, note = np.nan, np.nan, "all differences zero"

    results.append({
        "phenotype": phenotype,
        "n_patients": n,
        "TPC_MAE": round(tpc_mae, 4),
        "BAT_MAE": round(bat_mae, 4),
        "BAT_better": bat_mae < tpc_mae,
        "wilcoxon_stat": stat,
        "p_value": pval,
        "result": note,
    })

results_df = pd.DataFrame(results).sort_values("n_patients", ascending=False)
results_df.to_csv("significance_test_by_phenotype.csv", index=False)

print("\n=== Per-Phenotype Paired Wilcoxon Test: TPC vs BAT Absolute Error ===")
print(results_df.to_string(index=False))

n_significant = (results_df["p_value"] < 0.05).sum()
n_tested = results_df["p_value"].notna().sum()
print(f"\n{n_significant} / {n_tested} phenotypes show a statistically significant "
      f"difference (p < 0.05) between TPC and BAT's error.")

print("\nSaved: significance_test_by_phenotype.csv")