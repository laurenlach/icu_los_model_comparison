import numpy as np
import pandas as pd

# The 7 SOFA-relevant sensor indices
SOFA_SENSOR_INDICES = [27, 9, 15, 6, 39, 19, 35]  # map, bun, crea, bili, plt, fio2, o2sat
SOFA_SENSOR_NAMES = ["map", "bun", "crea", "bili", "plt", "fio2", "o2sat"]

# TPC (SHAP) 
tpc_shap = np.load("tpc_shap_values_full_testset.npy").squeeze(-1)  # (10367, 48, 24)

tpc_total_mass = np.abs(tpc_shap).sum()
tpc_sofa_mass = np.abs(tpc_shap[:, SOFA_SENSOR_INDICES, :]).sum()
tpc_sofa_pct = 100 * tpc_sofa_mass / tpc_total_mass

# BAT (IG, healthy baseline)
bat_ig = np.load("bat_ig_values_full_testset_healthybaseline.npy")  # (10367, 48, 24)

bat_total_mass = np.abs(bat_ig).sum()
bat_sofa_mass = np.abs(bat_ig[:, SOFA_SENSOR_INDICES, :]).sum()
bat_sofa_pct = 100 * bat_sofa_mass / bat_total_mass

# compute per-patient
def per_patient_sofa_pct(attr_array, sofa_indices):
    total_per_patient = np.abs(attr_array).sum(axis=(1, 2))
    sofa_per_patient = np.abs(attr_array[:, sofa_indices, :]).sum(axis=(1, 2))
    return 100 * sofa_per_patient / total_per_patient

tpc_per_patient = per_patient_sofa_pct(tpc_shap, SOFA_SENSOR_INDICES)
bat_per_patient = per_patient_sofa_pct(bat_ig, SOFA_SENSOR_INDICES)

# Random-chance baseline
chance_pct = 100 * len(SOFA_SENSOR_INDICES) / 48

print("=== SOFA-Relevant Attribution Mass ===")
print(f"(Chance baseline if attribution were random/uniform: {chance_pct:.2f}%)\n")

print(f"TPC (SHAP):")
print(f"  Pooled: {tpc_sofa_pct:.2f}% of total attribution mass on the 7 SOFA sensors")
print(f"  Per-patient mean: {tpc_per_patient.mean():.2f}% (std: {tpc_per_patient.std():.2f})")

print(f"\nBAT (IG, healthy baseline):")
print(f"  Pooled: {bat_sofa_pct:.2f}% of total attribution mass on the 7 SOFA sensors")
print(f"  Per-patient mean: {bat_per_patient.mean():.2f}% (std: {bat_per_patient.std():.2f})")

results_df = pd.DataFrame({
    "model": ["TPC", "BAT"],
    "pooled_sofa_mass_pct": [tpc_sofa_pct, bat_sofa_pct],
    "per_patient_mean_pct": [tpc_per_patient.mean(), bat_per_patient.mean()],
    "per_patient_std_pct": [tpc_per_patient.std(), bat_per_patient.std()],
    "chance_baseline_pct": [chance_pct, chance_pct],
})
results_df.to_csv("sofa_mass_proportion_results.csv", index=False)
print("\nSaved: sofa_mass_proportion_results.csv")