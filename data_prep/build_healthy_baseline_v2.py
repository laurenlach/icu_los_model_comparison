import numpy as np

feature_names = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun",
    "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2",
    "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
    "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt",
    "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine", "wbc"
]

# Healthy/normal reference values, midpoint of normal range.
# Lab values: ABIM (2026), "Laboratory Reference Ranges" (freely accessible).
# Vital signs (hr, dbp, sbp, resp, temp, map, fio2, urine): standard clinical
# vital-sign norms, not covered by the ABIM lab-values document.
healthy_values = {
    "alb": 4.5, "alp": 75, "alt": 25, "ast": 25, "be": 0, "bicar": 25.5,
    "bili": 0.65, "bili_dir": 0.2, "bnd": 2.5, "bun": 14, "ca": 9.4, "cai": 1.24,
    "ck": 100, "ckmb": 2.5, "cl": 102, "crea": 0.9, "crp": 0.4, "dbp": 70,
    "fgn": 300, "fio2": 0.21, "glu": 84.5, "hgb": 15, "hr": 75, "inr_pt": 1.0,
    "k": 4.25, "lact": 1.4, "lymph": 37.5, "map": 85, "mch": 30, "mchc": 34.5,
    "mcv": 89, "methb": 1.75, "mg": 2.1, "na": 140.5, "neut": 60, "o2sat": 98,
    "pco2": 40, "ph": 7.41, "phos": 3.75, "plt": 300, "po2": 87.5, "ptt": 30,
    "resp": 16, "sbp": 110, "temp": 37.0, "tnt": 0.005, "urine": 60, "wbc": 7.5,
}

healthy_raw = np.array([healthy_values[f] for f in feature_names])

# Load the correct means/stds 
feature_means = np.load("bat_feature_means.npy")
feature_stds = np.load("bat_feature_stds.npy")

# Convert to Z-scored space 
healthy_zscored = (healthy_raw - feature_means) / feature_stds

np.save("healthy_baseline_raw.npy", healthy_raw)
np.save("healthy_baseline_zscored.npy", healthy_zscored)

print("=== Healthy baseline, raw vs Z-scored ===")
for i, f in enumerate(feature_names):
    print(f"{f:10s}  raw={healthy_raw[i]:8.3f}  z-scored={healthy_zscored[i]:8.3f}")

print("\nSaved: healthy_baseline_raw.npy, healthy_baseline_zscored.npy")