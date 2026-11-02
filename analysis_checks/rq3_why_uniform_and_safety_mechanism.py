import numpy as np
import pandas as pd
from scipy.stats import spearmanr

tpc_pred = np.load("tpc_test_predictions.npy")
tpc_true = np.load("tpc_test_targets.npy")
bat_pred = np.load("bat_test_predictions.npy")
bat_true = np.load("bat_test_targets.npy")

pheno = pd.read_csv("cohort_icd9_phenotype_mapping_with_fallback.csv")
# Align to test-set order via stay_id -- adjust column name if different
stay_ids = pd.read_csv("tpc_shap_full_testset_stay_ids.csv")
id_col = stay_ids.columns[0]
test_pheno = stay_ids.merge(pheno, left_on=id_col, right_on=pheno.columns[0], how="left")["phenotype"]

df = pd.DataFrame({
    "phenotype": test_pheno.values,
    "tpc_pred": tpc_pred, "tpc_true": tpc_true,
    "bat_pred": bat_pred, "bat_true": bat_true,
})
df["tpc_abs_error"] = np.abs(df.tpc_pred - df.tpc_true)
df["bat_abs_error"] = np.abs(df.bat_pred - df.bat_true)
df["tpc_error"] = df.tpc_pred - df.tpc_true
df["bat_error"] = df.bat_pred - df.bat_true
df["bat_dangerous"] = df.bat_error <= -3.0
df["tpc_dangerous"] = df.tpc_error <= -3.0

# Check 1: does the BAT-vs-TPC MAE gap size vary with phenotype
# characteristics (n patients, mean true LOS)?
print("=== Check 1: does gap size vary systematically by phenotype? ===")
summary = df.dropna(subset=["phenotype"]).groupby("phenotype").agg(
    n=("tpc_true", "size"),
    mean_true_los=("tpc_true", "mean"),
    tpc_mae=("tpc_abs_error", "mean"),
    bat_mae=("bat_abs_error", "mean"),
).reset_index()
summary["mae_gap"] = summary["tpc_mae"] - summary["bat_mae"]   # positive = BAT better

rho_n, p_n = spearmanr(summary["n"], summary["mae_gap"])
rho_los, p_los = spearmanr(summary["mean_true_los"], summary["mae_gap"])
print(f"Correlation(phenotype size, BAT's MAE advantage):     rho={rho_n:+.3f} (p={p_n:.3g})")
print(f"Correlation(phenotype mean LOS, BAT's MAE advantage): rho={rho_los:+.3f} (p={p_los:.3g})")
print("\nTop 5 phenotypes where BAT's advantage is LARGEST:")
print(summary.sort_values("mae_gap", ascending=False).head(5)[["phenotype","n","mean_true_los","mae_gap"]].to_string(index=False))
print("\nTop 5 phenotypes where BAT's advantage is SMALLEST:")
print(summary.sort_values("mae_gap").head(5)[["phenotype","n","mean_true_los","mae_gap"]].to_string(index=False))

# Check 2: within each phenotype, is TPC's dangerous-case advantage
# concentrated in that phenotype's own long-stay (top-quintile) patients?
print("\n=== Check 2: is the safety pattern the same long-stay mechanism, per phenotype? ===")
results = []
for name, group in df.dropna(subset=["phenotype"]).groupby("phenotype"):
    if len(group) < 50:
        continue
    cutoff = group["tpc_true"].quantile(0.8)   # phenotype's OWN top-quintile cutoff
    long_stay = group["tpc_true"] >= cutoff
    bat_dang_long = group.loc[long_stay, "bat_dangerous"].mean()
    bat_dang_short = group.loc[~long_stay, "bat_dangerous"].mean()
    tpc_dang_long = group.loc[long_stay, "tpc_dangerous"].mean()
    tpc_dang_short = group.loc[~long_stay, "tpc_dangerous"].mean()
    results.append({
        "phenotype": name, "n": len(group),
        "bat_dangerous_overall": group["bat_dangerous"].mean(),
        "tpc_dangerous_overall": group["tpc_dangerous"].mean(),
        "bat_dang_in_own_top20pct": bat_dang_long,
        "tpc_dang_in_own_top20pct": tpc_dang_long,
        "bat_dang_outside_top20pct": bat_dang_short,
    })
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
print(f"\nMean dangerous-rate inside each phenotype's own top-20% LOS group (BAT): {res_df['bat_dang_in_own_top20pct'].mean():.3f}")
print(f"Mean dangerous-rate OUTSIDE that group (BAT):                             {res_df['bat_dang_outside_top20pct'].mean():.3f}")

df.to_csv("rq3_per_patient_phenotype_errors.csv", index=False)
res_df.to_csv("rq3_dangerous_by_phenotype_longstay.csv", index=False)
print("\nSaved: rq3_per_patient_phenotype_errors.csv, rq3_dangerous_by_phenotype_longstay.csv")