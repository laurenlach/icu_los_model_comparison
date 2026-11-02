import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

SOFA_SENSORS = {
    "map": {"idx": 27, "worse": "low"}, "bun": {"idx": 9, "worse": "high"},
    "crea": {"idx": 15, "worse": "high"}, "bili": {"idx": 6, "worse": "high"},
    "plt": {"idx": 39, "worse": "low"}, "fio2": {"idx": 19, "worse": "high"},
    "o2sat": {"idx": 35, "worse": "low"},
}
TEMP_IDX = 44
PH_IDX = 37

X = np.load("tpc_X_scaled.npy")
test_rows = np.load("tpc_test_rows.npy")
X_test = X[test_rows]

tpc_shap = np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1)
tpc_attr_mean = np.abs(tpc_shap).mean(axis=2)

temp_abnormal = X_test[:, TEMP_IDX, :].mean(axis=1)
ph_abnormal = X_test[:, PH_IDX, :].mean(axis=1)
controls = np.column_stack([temp_abnormal, ph_abnormal])

def partial_correlation(x, y, controls):
    reg_x = LinearRegression().fit(controls, x)
    resid_x = x - reg_x.predict(controls)
    reg_y = LinearRegression().fit(controls, y)
    resid_y = y - reg_y.predict(controls)
    rho, p = spearmanr(resid_x, resid_y)
    return rho, p

print("=== H4 Test: Does controlling for temp/ph change the backwards relationship? ===\n")
print(f"{'Sensor':<8}{'Raw rho':<12}{'Partial rho (controlling temp+ph)':<36}{'Direction'}")

results = []
for name, info in SOFA_SENSORS.items():
    idx = info["idx"]
    raw_full = X_test[:, idx, :].mean(axis=1)
    abnormality = -raw_full if info["worse"] == "low" else raw_full
    attr = tpc_attr_mean[:, idx]

    raw_rho, _ = spearmanr(abnormality, attr)
    partial_rho, partial_p = partial_correlation(abnormality, attr, controls)

    direction = "SUPPORTS H4 (moved toward positive)" if partial_rho > raw_rho else "does not support H4"
    print(f"{name:<8}{raw_rho:<12.4f}{partial_rho:<36.4f}{direction}")
    results.append({"sensor": name, "raw_rho": raw_rho, "partial_rho": partial_rho, "partial_p": partial_p})

pd.DataFrame(results).to_csv("hypothesis_h4_partial_correlation.csv", index=False)
print("\nSaved: hypothesis_h4_partial_correlation.csv")