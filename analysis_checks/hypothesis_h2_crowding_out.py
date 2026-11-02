import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SOFA_SENSORS = {
    "map":   {"idx": 27, "worse": "low"},
    "bun":   {"idx": 9,  "worse": "high"},
    "crea":  {"idx": 15, "worse": "high"},
    "bili":  {"idx": 6,  "worse": "high"},
    "plt":   {"idx": 39, "worse": "low"},
    "fio2":  {"idx": 19, "worse": "high"},
    "o2sat": {"idx": 35, "worse": "low"},
}
DOMINANT_FEATURES = {"temp": 44, "ph": 37}  # TPC's top-2 causal features (o2sat is both dominant AND a SOFA sensor)

X = np.load("tpc_X_scaled.npy")
test_rows = np.load("tpc_test_rows.npy")
X_test = X[test_rows]

tpc_shap = np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1)
tpc_attr_mean = np.abs(tpc_shap).mean(axis=2)

# How abnormal is temp+ph, combined, per patient (average of both, higher = more extreme)
temp_abnormal = np.abs(X_test[:, DOMINANT_FEATURES["temp"], :].mean(axis=1))
ph_abnormal = np.abs(X_test[:, DOMINANT_FEATURES["ph"], :].mean(axis=1))
dominant_abnormality = (temp_abnormal + ph_abnormal) / 2

# Split patients into "dominant features calm" vs "dominant features extreme" (median split)
median_dom = np.median(dominant_abnormality)
calm_mask = dominant_abnormality <= median_dom
extreme_mask = dominant_abnormality > median_dom

print("=== H2 Test: Does TPC's dose-response improve when temp/ph AREN'T also extreme (less crowding-out)? ===\n")
print(f"{'Sensor':<8}{'rho (temp/ph calm)':<22}{'rho (temp/ph extreme)':<22}{'Direction'}")

results = []
for name, info in SOFA_SENSORS.items():
    idx = info["idx"]
    raw_full = X_test[:, idx, :].mean(axis=1)
    abnormality = -raw_full if info["worse"] == "low" else raw_full
    attr = tpc_attr_mean[:, idx]

    rho_calm, _ = spearmanr(abnormality[calm_mask], attr[calm_mask])
    rho_extreme, _ = spearmanr(abnormality[extreme_mask], attr[extreme_mask])

    direction = "Supports H2 (better when temp/ph calm)" if rho_calm > rho_extreme else "Does not support H2"
    print(f"{name:<8}{rho_calm:<22.4f}{rho_extreme:<22.4f}{direction}")
    results.append({"sensor": name, "rho_temp_ph_calm": rho_calm, "rho_temp_ph_extreme": rho_extreme, "supports_h2": rho_calm > rho_extreme})

pd.DataFrame(results).to_csv("hypothesis_h2_crowding_out.csv", index=False)
print("\nSaved: hypothesis_h2_crowding_out.csv")