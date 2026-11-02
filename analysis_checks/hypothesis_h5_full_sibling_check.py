import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

feature_names = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun",
    "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2",
    "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
    "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt",
    "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine", "wbc"
]
SOFA_SENSORS = {
    "bun": {"idx": 9, "worse": "high"}, "crea": {"idx": 15, "worse": "high"},
    "plt": {"idx": 39, "worse": "low"}, "o2sat": {"idx": 35, "worse": "low"},
}

def partial_correlation(x, y, controls):
    reg_x = LinearRegression().fit(controls, x)
    resid_x = x - reg_x.predict(controls)
    reg_y = LinearRegression().fit(controls, y)
    resid_y = y - reg_y.predict(controls)
    return spearmanr(resid_x, resid_y)

def run_sibling_check(X_test, attr_mean, model_name):
    print(f"\n{'='*70}\n{model_name}\n{'='*70}")
    results = []
    for name, info in SOFA_SENSORS.items():
        idx = info["idx"]

        # Find this sensor's top-2 most correlated OTHER features
        this_vals = X_test[:, idx, :].mean(axis=1)
        correlations = []
        for j in range(48):
            if j == idx:
                continue
            other_vals = X_test[:, j, :].mean(axis=1)
            r, _ = spearmanr(this_vals, other_vals)
            correlations.append((feature_names[j], abs(r)))
        correlations.sort(key=lambda x: -x[1])
        top_siblings = correlations[:2]
        sibling_names = [s[0] for s in top_siblings]
        sibling_idx = [feature_names.index(s) for s in sibling_names]

        raw_full = X_test[:, idx, :].mean(axis=1)
        abnormality = -raw_full if info["worse"] == "low" else raw_full
        attr = attr_mean[:, idx]

        raw_rho, _ = spearmanr(abnormality, attr)

        controls = np.column_stack([X_test[:, si, :].mean(axis=1) for si in sibling_idx])
        partial_rho, partial_p = partial_correlation(abnormality, attr, controls)

        verdict = "MOVED TOWARD POSITIVE (supports sibling-crowding)" if partial_rho > raw_rho else "no improvement"
        print(f"{name} (siblings: {sibling_names}): raw={raw_rho:.4f}, partial={partial_rho:.4f} -> {verdict}")
        results.append({"sensor": name, "model": model_name, "siblings": sibling_names,
                         "raw_rho": raw_rho, "partial_rho": partial_rho})
    return results

# --- TPC ---
tpc_X = np.load("tpc_X_scaled.npy")
tpc_test_rows = np.load("tpc_test_rows.npy")
tpc_X_test = tpc_X[tpc_test_rows]
tpc_shap = np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1)
tpc_attr = np.abs(tpc_shap).mean(axis=2)
tpc_results = run_sibling_check(tpc_X_test, tpc_attr, "TPC (fixed)")

# --- BAT (control check) ---
bat_X = np.load("bat_X_scaled.npy")
bat_test_rows = np.load("bat_test_rows.npy")
bat_X_test = np.transpose(bat_X[bat_test_rows], (0, 2, 1))  # -> (n, 48, 24) to match TPC's layout
bat_ig = np.load("bat_ig_values_full_testset_healthybaseline.npy")
bat_attr = np.abs(bat_ig).mean(axis=2)
bat_results = run_sibling_check(bat_X_test, bat_attr, "BAT")

pd.DataFrame(tpc_results + bat_results).to_csv("hypothesis_h5_full_sibling_check.csv", index=False)
print("\nSaved: hypothesis_h5_full_sibling_check.csv")