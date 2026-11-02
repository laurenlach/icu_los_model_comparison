import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sofa = pd.read_csv("sofa_24h_scores.csv").rename(columns={"icustay_id": "stay_id"})
sofa_cols = ["sofa_cardio_comp", "sofa_renal_comp", "sofa_liver_comp", "sofa_coag_comp", "sofa_resp_comp", "sofa_cns_comp"]
sofa["sofa_total"] = sofa[sofa_cols].sum(axis=1, skipna=True)

bat_error = np.load("bat_error.npy")
bat_dangerous = np.load("bat_dangerous_mask_threshold_3_0days.npy")
bat_y = np.load("bat_y.npy")
bat_test_rows = np.load("bat_test_rows.npy")
stay_id_order = pd.read_csv("bat_stay_id_order.csv")

true_los = bat_y[bat_test_rows]
stay_ids = stay_id_order["stay_id"].iloc[bat_test_rows].values

df = pd.DataFrame({
    "stay_id": stay_ids,
    "dangerous": bat_dangerous,
    "true_los": true_los,
    "error": bat_error,
})
df = df.merge(sofa[["stay_id", "sofa_total"] + sofa_cols], on="stay_id", how="left")

accurate = df[~df["dangerous"]]
dangerous = df[df["dangerous"]]

print("=== HYPOTHESIS B: Do dangerous cases look mild during the first 24h? ===\n")
print(f"Accurate group  (n={len(accurate)}): mean true LOS = {accurate['true_los'].mean():.2f} days, "
      f"mean SOFA total = {accurate['sofa_total'].mean():.2f}")
print(f"Dangerous group (n={len(dangerous)}): mean true LOS = {dangerous['true_los'].mean():.2f} days, "
      f"mean SOFA total = {dangerous['sofa_total'].mean():.2f}")

stat, p = mannwhitneyu(dangerous["sofa_total"].dropna(), accurate["sofa_total"].dropna())
print(f"\nMann-Whitney U test, SOFA total (dangerous vs accurate): p={p:.4g}")

print("\n=== Per-organ SOFA sub-score comparison ===")
for col in sofa_cols:
    acc_mean = accurate[col].mean()
    dan_mean = dangerous[col].mean()
    print(f"{col}: accurate={acc_mean:.3f}, dangerous={dan_mean:.3f}, "
          f"{'HIGHER in dangerous' if dan_mean > acc_mean else 'LOWER in dangerous'}")

df.to_csv("hypothesis_b_severity_comparison.csv", index=False)
print("\nSaved: hypothesis_b_severity_comparison.csv")