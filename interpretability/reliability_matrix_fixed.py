import pandas as pd

performance = pd.read_csv("significance_test_by_phenotype_fixed.csv")
alignment = pd.read_csv("phenotype_alignment_aggregate_fixed.csv")

# Build a performance table in long format (one row per phenotype/model)
perf_long = []
for _, row in performance.iterrows():
    perf_long.append({"phenotype": row["phenotype"], "model": "TPC", "mae": row["TPC_MAE_fixed"]})
    perf_long.append({"phenotype": row["phenotype"], "model": "BAT", "mae": row["BAT_MAE"]})
perf_df = pd.DataFrame(perf_long)

matrix = perf_df.merge(alignment, on=["phenotype", "model"], how="inner")

sig_slim = performance[["phenotype", "n_patients", "p_value", "result"]]
matrix = matrix.merge(sig_slim, on="phenotype", how="left")

matrix = matrix.sort_values(["phenotype", "model"])
print(matrix.to_string(index=False))

# Summary counts
mae_pivot = matrix.pivot(index="phenotype", columns="model", values="mae")
align_pivot = matrix.pivot(index="phenotype", columns="model", values="mean_alignment_score")

bat_wins_mae = (mae_pivot["BAT"] < mae_pivot["TPC"]).sum()
bat_wins_align = (align_pivot["BAT"] > align_pivot["TPC"]).sum()

print(f"\n=== SUMMARY (fixed TPC vs BAT, 25 phenotypes) ===")
print(f"BAT wins MAE: {bat_wins_mae}/25")
print(f"BAT wins SOFA-alignment: {bat_wins_align}/25")

matrix.to_csv("reliability_matrix_fixed.csv", index=False)
print("\nSaved: reliability_matrix_fixed.csv")