import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

SOFA_SENSOR_INDICES = [27, 9, 15, 6, 39, 19, 35]  # map, bun, crea, bili, plt, fio2, o2sat

sofa = pd.read_csv("sofa_24h_scores.csv").rename(columns={"icustay_id": "stay_id"})
sofa_cols = ["sofa_cardio_comp", "sofa_renal_comp", "sofa_liver_comp", "sofa_coag_comp", "sofa_resp_comp", "sofa_cns_comp"]
sofa["sofa_total"] = sofa[sofa_cols].sum(axis=1, skipna=True)

bat_ig = np.load("bat_ig_values_full_testset_healthybaseline.npy")
bat_ids = pd.read_csv("bat_ig_full_testset_stay_ids_healthybaseline.csv")
bat_dangerous = np.load("bat_dangerous_mask_threshold_3_0days.npy")

# Total attribution mass on the 7 SOFA-relevant sensors, per patient
sofa_attr_mass = np.abs(bat_ig[:, SOFA_SENSOR_INDICES, :]).sum(axis=(1, 2))

df = pd.DataFrame({
    "stay_id": bat_ids["stay_id"],
    "dangerous": bat_dangerous,
    "sofa_attr_mass": sofa_attr_mass,
})
df = df.merge(sofa[["stay_id", "sofa_total"]], on="stay_id", how="left").dropna(subset=["sofa_total"])

# Avoid division by zero for patients with sofa_total = 0
df = df[df["sofa_total"] > 0].copy()
df["attribution_per_severity"] = df["sofa_attr_mass"] / df["sofa_total"]

accurate = df[~df["dangerous"]]
dangerous = df[df["dangerous"]]

print("=== HYPOTHESIS A: Does BAT under-weight SOFA sensors relative to severity in dangerous cases? ===\n")

print(f"Raw SOFA-sensor attribution mass:")
print(f"  Accurate  (n={len(accurate)}): mean = {accurate['sofa_attr_mass'].mean():.4f}")
print(f"  Dangerous (n={len(dangerous)}): mean = {dangerous['sofa_attr_mass'].mean():.4f}")
stat, p = mannwhitneyu(dangerous["sofa_attr_mass"], accurate["sofa_attr_mass"])
print(f"  Mann-Whitney U p={p:.4g}")

print(f"\nAttribution PER UNIT of SOFA severity (attribution_mass / sofa_total):")
print(f"  Accurate  : mean = {accurate['attribution_per_severity'].mean():.4f}")
print(f"  Dangerous : mean = {dangerous['attribution_per_severity'].mean():.4f}")
stat2, p2 = mannwhitneyu(dangerous["attribution_per_severity"], accurate["attribution_per_severity"])
print(f"  Mann-Whitney U p={p2:.4g}")

ratio = dangerous['attribution_per_severity'].mean() / accurate['attribution_per_severity'].mean()
print(f"\nDangerous-group attribution-per-severity is {ratio:.2f}x the accurate group's "
      f"({'LOWER = under-weighting evidence' if ratio < 1 else 'HIGHER or similar = no under-weighting'})")

df.to_csv("hypothesis_a_underweighting_comparison.csv", index=False)
print("\nSaved: hypothesis_a_underweighting_comparison.csv")