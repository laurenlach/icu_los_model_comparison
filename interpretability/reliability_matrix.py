import pandas as pd

performance = pd.read_csv("phenotype_performance_summary.csv")
alignment = pd.read_csv("phenotype_alignment_aggregate.csv")

matrix = performance.merge(
    alignment,
    on=["phenotype", "model"],
    how="inner",
    suffixes=("_perf", "_align"),
)

assert len(matrix) == len(performance), \
    f"Row count changed after merge ({len(performance)} -> {len(matrix)}) — a phenotype/model pair didn't match"

# Keep the columns that actually belong in the reliability matrix, in a sensible order
matrix = matrix[[
    "phenotype", "model", "n_patients", "mae", "mean_error", "dangerous_case_pct",
    "mean_alignment_score", "organs_with_data", "min_n_across_organs",
]]

matrix = matrix.sort_values(["phenotype", "model"])
matrix.to_csv("reliability_matrix.csv", index=False)

print(matrix.to_string(index=False))