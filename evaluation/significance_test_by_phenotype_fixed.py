import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

tpc_error = np.load("tpc_error_fixed.npy")
bat_error = np.load("bat_error.npy")
tpc_test_rows = np.load("tpc_test_rows.npy")
bat_test_rows = np.load("bat_test_rows.npy")

assert np.array_equal(tpc_test_rows, bat_test_rows), "Row order mismatch - investigate before continuing."

stay_id_order = pd.read_csv("bat_stay_id_order.csv")
row_stay_ids = stay_id_order["stay_id"].iloc[bat_test_rows].values

phenotype_map = pd.read_csv("cohort_icd9_phenotype_mapping_with_fallback.csv")

df = pd.DataFrame({
    "stay_id": row_stay_ids,
    "tpc_abs_error": np.abs(tpc_error),
    "bat_abs_error": np.abs(bat_error),
})
df = df.merge(phenotype_map, on="stay_id", how="left").dropna(subset=["phenotype"])

results = []
for phenotype, group in df.groupby("phenotype"):
    n = len(group)
    tpc_mae = group["tpc_abs_error"].mean()
    bat_mae = group["bat_abs_error"].mean()
    if n < 10:
        stat, pval, note = np.nan, np.nan, "n<10"
    else:
        stat, pval = wilcoxon(group["tpc_abs_error"], group["bat_abs_error"])
        note = "significant" if pval < 0.05 else "not significant"
    results.append({
        "phenotype": phenotype, "n_patients": n,
        "TPC_MAE_fixed": round(tpc_mae, 4), "BAT_MAE": round(bat_mae, 4),
        "BAT_better": bat_mae < tpc_mae, "p_value": pval, "result": note,
    })

results_df = pd.DataFrame(results).sort_values("n_patients", ascending=False)
print(results_df.to_string(index=False))

n_sig = (results_df["p_value"] < 0.05).sum()
n_bat_better = results_df["BAT_better"].sum()
print(f"\n{n_sig}/{results_df['p_value'].notna().sum()} significant, "
      f"BAT better in {n_bat_better}/{len(results_df)} phenotypes")

results_df.to_csv("significance_test_by_phenotype_fixed.csv", index=False)