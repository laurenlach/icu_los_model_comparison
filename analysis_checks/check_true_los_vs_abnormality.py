import numpy as np
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
y_test = np.load("tpc_test_targets_fixed.npy")
X_test = X[test_rows]

print("sensor | corr(abnormality, TRUE LOS) | TPC test-set mean (scaled)")
for name, info in SOFA_SENSORS.items():
    idx = info["idx"]
    raw = X_test[:, idx, :].mean(axis=1)
    abnormality = -raw if info["worse"] == "low" else raw
    rho, p = spearmanr(abnormality, y_test)
    print(f"{name:6s} rho={rho:+.4f} (p={p:.3g}) | mean={X_test[:, idx, :].mean():+.3f}")