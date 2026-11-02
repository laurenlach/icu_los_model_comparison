import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

SOFA_SENSORS = {
    "map":   {"idx": 27, "worse": "low"},
    "bun":   {"idx": 9,  "worse": "high"},
    "crea":  {"idx": 15, "worse": "high"},
    "bili":  {"idx": 6,  "worse": "high"},
    "plt":   {"idx": 39, "worse": "low"},
    "fio2":  {"idx": 19, "worse": "high"},
    "o2sat": {"idx": 35, "worse": "low"},
}

# --- Load everything (all files already exist from earlier steps) ---
X = np.load("tpc_X_scaled.npy")
tpc_rows = np.load("tpc_test_rows.npy")
y_test = np.load("tpc_test_targets_fixed.npy")
X_test = X[tpc_rows]                               # (n, 48, 24)

mask = np.load("bat_mask.npy")
bat_rows = np.load("bat_test_rows.npy")
mask_test = mask[bat_rows]                         # (n, 24, 48), 1 = really measured

y_bat = np.load("bat_y.npy")[bat_rows]
aligned = (y_bat.shape == y_test.shape) and np.allclose(y_bat, y_test)
print(f"TPC and BAT test patients aligned (same targets): {aligned}")
if not aligned:
    print("WARNING: targets differ, so the mask may not line up with TPC's patients. Tell me before trusting section A.")

tpc_attr = np.abs(np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1)).mean(axis=2)
bat_attr = np.abs(np.load("bat_ig_values_full_testset_healthybaseline.npy")).mean(axis=2)

def partial_correlation(x, y, controls):
    rx = x - LinearRegression().fit(controls, x).predict(controls)
    ry = y - LinearRegression().fit(controls, y).predict(controls)
    return spearmanr(rx, ry)[0]

# ---------------------------------------------------------------------
# A. Does restricting to REALLY MEASURED patients change the picture?
# ---------------------------------------------------------------------
print("\n=== A. Dose-response by how much the sensor was really measured ===")
rows = []
for name, info in SOFA_SENSORS.items():
    idx = info["idx"]
    raw = X_test[:, idx, :].mean(axis=1)
    abn = -raw if info["worse"] == "low" else raw
    hours_measured = mask_test[:, :, idx].sum(axis=1)

    for label, keep in [("all patients", hours_measured >= 0),
                        ("measured >=1 h", hours_measured >= 1),
                        ("measured >=12 h", hours_measured >= 12)]:
        n = int(keep.sum())
        if n < 200:
            print(f"{name:6s} {label:16s} n={n} (too few, skipped)")
            continue
        rho_los = spearmanr(abn[keep], y_test[keep])[0]
        rho_shap = spearmanr(abn[keep], tpc_attr[keep, idx])[0]
        rho_ig = spearmanr(abn[keep], bat_attr[keep, idx])[0]
        rows.append({"sensor": name, "subset": label, "n": n, "pct_of_patients": 100 * n / len(y_test),
                     "corr_with_true_LOS": rho_los, "TPC_SHAP_rho": rho_shap, "BAT_IG_rho": rho_ig})
        print(f"{name:6s} {label:16s} n={n:5d} ({100*n/len(y_test):5.1f}%) | true LOS rho={rho_los:+.3f} "
              f"| SHAP rho={rho_shap:+.3f} | IG rho={rho_ig:+.3f}")
    print()
pd.DataFrame(rows).to_csv("why_check_measured_only.csv", index=False)

# ---------------------------------------------------------------------
# B. Organ-support masking: is o2sat still "backwards" within FiO2 groups?
# ---------------------------------------------------------------------
print("=== B. o2sat vs true LOS, within FiO2 (respiratory support) tertiles ===")
fio2 = X_test[:, 19, :].mean(axis=1)
o2_abn = -X_test[:, 35, :].mean(axis=1)
cuts = np.quantile(fio2, [1/3, 2/3])
groups = {
    "low FiO2":  fio2 <= cuts[0],
    "mid FiO2":  (fio2 > cuts[0]) & (fio2 <= cuts[1]),
    "high FiO2": fio2 > cuts[1],
}
print(f"overall: rho={spearmanr(o2_abn, y_test)[0]:+.3f}")
for g, keep in groups.items():
    print(f"{g:10s} n={int(keep.sum()):5d} | rho={spearmanr(o2_abn[keep], y_test[keep])[0]:+.3f}")
print(f"controlling for FiO2 (partial): rho={partial_correlation(o2_abn, y_test, fio2.reshape(-1, 1)):+.3f}")