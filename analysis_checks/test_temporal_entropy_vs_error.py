import numpy as np
from scipy.stats import spearmanr

def normalized_entropy(values):
    values = np.clip(values, 1e-12, None)
    p = values / values.sum()
    entropy = -np.sum(p * np.log(p))
    return entropy / np.log(len(values))

# BAT's IG attribution, healthy baseline (10367, 48, 24)
bat_ig = np.load("bat_ig_values_full_testset_healthybaseline.npy")
bat_error = np.load("bat_error.npy")  # predicted - true, per patient

# Collapse to per-patient, per-hour attribution (sum |attribution| across the 48 sensors)
attr_by_hour = np.abs(bat_ig).sum(axis=1)  # (10367, 24)

entropy_per_patient = np.array([normalized_entropy(row) for row in attr_by_hour])

print(f"Entropy range: {entropy_per_patient.min():.4f} to {entropy_per_patient.max():.4f}")
print(f"Mean: {entropy_per_patient.mean():.4f} (cohort-level check gave 0.9587 earlier)\n")

rho, p = spearmanr(entropy_per_patient, bat_error)
print(f"Correlation (entropy vs signed error): rho={rho:+.4f} (p={p:.3g})")
print("Negative rho = more uniform attention goes with MORE under-prediction (supports H1)")
print("Near-zero/positive rho = no support for H1\n")

# Also check against |error| in case direction isn't the right lens
rho_abs, p_abs = spearmanr(entropy_per_patient, np.abs(bat_error))
print(f"Correlation (entropy vs |error|): rho={rho_abs:+.4f} (p={p_abs:.3g})")

np.save("bat_temporal_entropy_per_patient.npy", entropy_per_patient)
print("\nSaved: bat_temporal_entropy_per_patient.npy")