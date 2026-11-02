import pandas as pd

performance = pd.read_csv("phenotype_performance_summary.csv")
alignment = pd.read_csv("phenotype_alignment_aggregate_v2.csv")
significance = pd.read_csv("significance_test_by_phenotype.csv")

matrix = performance.merge(
    alignment,
    on=["phenotype", "model"],
    how="inner",
    suffixes=("_perf", "_align"),
)

assert len(matrix) == len(performance), \
    f"Row count changed after merge ({len(performance)} -> {len(matrix)}) — investigate"

# Significance test was run once per phenotype (TPC vs BAT together, not per-model),
# so merge it in on phenotype only, then it'll apply to both models' rows
sig_slim = significance[["phenotype", "n_patients", "p_value", "result"]].rename(
    columns={"n_patients": "n_patients_significance_test"}
)
matrix = matrix.merge(sig_slim, on="phenotype", how="left")

matrix = matrix[[
    "phenotype", "model", "n_patients", "mae", "mean_error", "dangerous_case_pct",
    "mean_alignment_score", "organs_with_data", "min_n_across_organs",
    "p_value", "result",
]]

matrix = matrix.sort_values(["phenotype", "model"])
matrix.to_csv("reliability_matrix_v2.csv", index=False)

print(matrix.to_string(index=False))