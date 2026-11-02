import numpy as np

SOFA_SENSOR_INDICES = [27, 9, 15, 6, 39, 19, 35]  # map, bun, crea, bili, plt, fio2, o2sat

tpc_shap = np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1)

tpc_total_mass = np.abs(tpc_shap).sum()
tpc_sofa_mass = np.abs(tpc_shap[:, SOFA_SENSOR_INDICES, :]).sum()
tpc_sofa_pct = 100 * tpc_sofa_mass / tpc_total_mass

def per_patient_sofa_pct(attr_array, sofa_indices):
    total_per_patient = np.abs(attr_array).sum(axis=(1, 2))
    sofa_per_patient = np.abs(attr_array[:, sofa_indices, :]).sum(axis=(1, 2))
    return 100 * sofa_per_patient / total_per_patient

tpc_per_patient = per_patient_sofa_pct(tpc_shap, SOFA_SENSOR_INDICES)

chance_pct = 100 * len(SOFA_SENSOR_INDICES) / 48

print(f"Chance baseline: {chance_pct:.2f}%")
print(f"TPC (fixed) pooled: {tpc_sofa_pct:.2f}%")
print(f"TPC (fixed) per-patient mean: {tpc_per_patient.mean():.2f}% (std {tpc_per_patient.std():.2f})")
print(f"\n(For comparison - old buggy TPC was 23.71% pooled, BAT is 35.89% pooled)")