import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

X = np.load("tpc_X_scaled.npy")
test_rows = np.load("tpc_test_rows.npy")
X_test = X[test_rows]

tpc_shap = np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1)
tpc_attr_mean = np.abs(tpc_shap).mean(axis=2)

MAP_IDX = 27
SBP_IDX = 43
DBP_IDX = 17

map_abnormal = -X_test[:, MAP_IDX, :].mean(axis=1)  # low MAP = worse
map_attr = tpc_attr_mean[:, MAP_IDX]

sbp_vals = X_test[:, SBP_IDX, :].mean(axis=1)
dbp_vals = X_test[:, DBP_IDX, :].mean(axis=1)

def partial_correlation(x, y, controls):
    reg_x = LinearRegression().fit(controls, x)
    resid_x = x - reg_x.predict(controls)
    reg_y = LinearRegression().fit(controls, y)
    resid_y = y - reg_y.predict(controls)
    rho, p = spearmanr(resid_x, resid_y)
    return rho, p

raw_rho, raw_p = spearmanr(map_abnormal, map_attr)
print(f"Raw MAP rho: {raw_rho:.4f} (p={raw_p:.4g})")

# H5 test: control for SBP+DBP instead of temp/ph
controls_bp = np.column_stack([sbp_vals, dbp_vals])
partial_rho_bp, partial_p_bp = partial_correlation(map_abnormal, map_attr, controls_bp)
print(f"MAP rho controlling for SBP+DBP: {partial_rho_bp:.4f} (p={partial_p_bp:.4g})")

# For comparison: control for ALL of SBP, DBP, temp, ph together
TEMP_IDX, PH_IDX = 44, 37
temp_vals = X_test[:, TEMP_IDX, :].mean(axis=1)
ph_vals = X_test[:, PH_IDX, :].mean(axis=1)
controls_all = np.column_stack([sbp_vals, dbp_vals, temp_vals, ph_vals])
partial_rho_all, partial_p_all = partial_correlation(map_abnormal, map_attr, controls_all)
print(f"MAP rho controlling for SBP+DBP+temp+ph: {partial_rho_all:.4f} (p={partial_p_all:.4g})")

verdict = "SUPPORTS H5" if partial_rho_bp > raw_rho else "does not support H5"
print(f"\nVerdict: {verdict}")

pd.DataFrame([{
    "raw_rho": raw_rho, "partial_rho_bp_only": partial_rho_bp, "partial_rho_all_four": partial_rho_all
}]).to_csv("hypothesis_h5_bp_crowding.csv", index=False)
print("Saved: hypothesis_h5_bp_crowding.csv")