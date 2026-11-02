import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

X = np.load("tpc_X_scaled.npy")
rows = np.load("tpc_test_rows.npy")
y = np.load("tpc_test_targets_fixed.npy")
X_test = X[rows]
mask = np.load("bat_mask.npy")[np.load("bat_test_rows.npy")]   # (n, 24, 48), 1=real measurement

def partial_corr(x, y, controls):
    rx = x - LinearRegression().fit(controls, x).predict(controls)
    ry = y - LinearRegression().fit(controls, y).predict(controls)
    return spearmanr(rx, ry)[0]

# --- Check 1: platelets, monitoring-intensity test ---
plt_idx = 39
raw = X_test[:, plt_idx, :].mean(axis=1)
abn = -raw   # low platelets = worse
hours_measured = mask[:, :, plt_idx].sum(axis=1)

rho_raw = spearmanr(abn, y)[0]
rho_ctrl = partial_corr(abn, y, hours_measured.reshape(-1, 1).astype(float))

print("=== Platelets: monitoring-intensity check ===")
print(f"Raw correlation (abnormality vs true LOS):            rho={rho_raw:+.3f}")
print(f"...controlling for how often platelets were measured: rho={rho_ctrl:+.3f}")
reduction = 100 * (1 - abs(rho_ctrl) / abs(rho_raw)) if rho_raw != 0 else float("nan")
print(f"Reduction in magnitude: {reduction:.1f}%")

# --- Check 2: O2 sat, monitoring-intensity AND FiO2 together ---
o2_idx, fio2_idx = 35, 19
raw_o2 = X_test[:, o2_idx, :].mean(axis=1)
abn_o2 = -raw_o2   # low o2sat = worse
hours_measured_o2 = mask[:, :, o2_idx].sum(axis=1)
fio2_level = X_test[:, fio2_idx, :].mean(axis=1)

rho_raw_o2 = spearmanr(abn_o2, y)[0]
rho_ctrl_freq_only = partial_corr(abn_o2, y, hours_measured_o2.reshape(-1, 1).astype(float))
rho_ctrl_fio2_only = partial_corr(abn_o2, y, fio2_level.reshape(-1, 1))
controls_both = np.column_stack([hours_measured_o2, fio2_level])
rho_ctrl_both = partial_corr(abn_o2, y, controls_both)

print("\n=== O2 sat: monitoring-intensity + FiO2, combined ===")
print(f"Raw correlation:                          rho={rho_raw_o2:+.3f}")
print(f"...controlling for measurement frequency only: rho={rho_ctrl_freq_only:+.3f}")
print(f"...controlling for FiO2 only:                  rho={rho_ctrl_fio2_only:+.3f}")
print(f"...controlling for BOTH together:              rho={rho_ctrl_both:+.3f}")
reduction_both = 100 * (1 - abs(rho_ctrl_both) / abs(rho_raw_o2))
print(f"Reduction in magnitude (both combined): {reduction_both:.1f}%")