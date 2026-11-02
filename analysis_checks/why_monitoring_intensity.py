import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

feature_names = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun",
    "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2",
    "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
    "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt",
    "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine", "wbc"
]

X = np.load("tpc_X_scaled.npy")
rows = np.load("tpc_test_rows.npy")
y = np.load("tpc_test_targets_fixed.npy")
X_test = X[rows]
mask = np.load("bat_mask.npy")[np.load("bat_test_rows.npy")]   # (n, 24, 48), 1=real measurement

def partial_corr(x, y, controls):
    rx = x - LinearRegression().fit(controls, x).predict(controls)
    ry = y - LinearRegression().fit(controls, y).predict(controls)
    return spearmanr(rx, ry)[0]

for name, idx, worse in [("map", 27, "low"), ("o2sat", 35, "low")]:
    raw = X_test[:, idx, :].mean(axis=1)
    abn = -raw if worse == "low" else raw
    hours_measured = mask[:, :, idx].sum(axis=1)   # how often THIS sensor was measured

    # Overall sickness proxy: total measurements across all 48 sensors (care-intensity proxy)
    total_meas = mask.sum(axis=(1, 2))

    rho_freq_abn = spearmanr(hours_measured, abn)[0]
    rho_freq_los = spearmanr(hours_measured, y)[0]
    rho_abn_los = spearmanr(abn, y)[0]
    rho_abn_los_ctrl_freq = partial_corr(abn, y, hours_measured.reshape(-1, 1).astype(float))
    rho_abn_los_ctrl_totalmeas = partial_corr(abn, y, total_meas.reshape(-1, 1).astype(float))

    print(f"\n=== {name} ===")
    print(f"How often {name} itself was measured vs how abnormal it is:  rho={rho_freq_abn:+.3f}")
    print(f"How often {name} was measured vs true LOS:                   rho={rho_freq_los:+.3f}")
    print(f"Abnormality vs true LOS (raw):                                rho={rho_abn_los:+.3f}")
    print(f"...controlling for how often {name} itself was measured:     rho={rho_abn_los_ctrl_freq:+.3f}")
    print(f"...controlling for TOTAL measurements (overall care intensity): rho={rho_abn_los_ctrl_totalmeas:+.3f}")