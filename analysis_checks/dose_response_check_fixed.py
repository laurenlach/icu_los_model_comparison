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

X = np.load("tpc_X_scaled.npy")
test_rows = np.load("tpc_test_rows.npy")
X_test = X[test_rows].mean(axis=2)  # (10367, 48) - TPC's shape is (patients, features, time)

tpc_shap = np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1)
tpc_attr_mean = np.abs(tpc_shap).mean(axis=2)  # (10367, 48)

results = []
for name, info in SOFA_SENSORS.items():
    idx = info["idx"]
    raw_values = X_test[:, idx]
    abnormality = -raw_values if info["worse"] == "low" else raw_values
    rho, p = spearmanr(abnormality, tpc_attr_mean[:, idx])
    results.append({"sensor": name, "tpc_fixed_rho": rho, "p": p})

print("=== TPC (FIXED) Dose-response ===")
for r in results:
    print(f"{r['sensor']}: rho={r['tpc_fixed_rho']:.4f} (p={r['p']:.4g})")

print("\n(For comparison, OLD buggy TPC rhos were: "
      "map=-0.157, bun=-0.015, crea=-0.037, bili=-0.076, plt=-0.093, fio2=-0.004, o2sat=0.086)")
print("(BAT rhos were: map=0.042, bun=0.696, crea=0.392, bili=0.053, plt=0.377, fio2=0.536, o2sat=0.167)")

pd.DataFrame(results).to_csv("dose_response_check_fixed.csv", index=False)