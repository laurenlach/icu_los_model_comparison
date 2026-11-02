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

# Same 7 sensors and directions as dose_response_check_fixed.py
SOFA_SENSORS = {
    "map":   {"idx": 27, "worse": "low"},
    "bun":   {"idx": 9,  "worse": "high"},
    "crea":  {"idx": 15, "worse": "high"},
    "bili":  {"idx": 6,  "worse": "high"},
    "plt":   {"idx": 39, "worse": "low"},
    "fio2":  {"idx": 19, "worse": "high"},
    "o2sat": {"idx": 35, "worse": "low"},
}

# Same siblings that were confirmed/tested for TPC in the investigation
DOMAIN_SIBLINGS = {
    "map":   ["sbp", "dbp"],
    "bili":  ["temp", "ph"],
    "bun":   ["temp", "ph"],
    "fio2":  ["temp", "ph"],
    "crea":  ["bun", "phos"],
    "o2sat": ["fio2", "pco2", "po2", "resp"],
    "plt":   ["ptt", "inr_pt", "fgn"],
}

def partial_correlation(x, y, controls):
    reg_x = LinearRegression().fit(controls, x)
    resid_x = x - reg_x.predict(controls)
    reg_y = LinearRegression().fit(controls, y)
    resid_y = y - reg_y.predict(controls)
    return spearmanr(resid_x, resid_y)[0]

def verdict(raw, partial):
    if raw < 0 and partial > 0:
        return "FLIPPED TO POSITIVE"
    if partial > raw:
        return "moved toward positive"
    return "no improvement"

def run_check(X_test, attr_mean, model_name, results):
    print(f"\n{'='*78}\n{model_name}\n{'='*78}")

    # Model's own two most-attributed features overall (its "dominant" features)
    mean_attr_per_feature = attr_mean.mean(axis=0)
    ranked = np.argsort(-mean_attr_per_feature)

    for name, info in SOFA_SENSORS.items():
        idx = info["idx"]
        raw_full = X_test[:, idx, :].mean(axis=1)
        abnormality = -raw_full if info["worse"] == "low" else raw_full
        attr = attr_mean[:, idx]
        raw_rho = spearmanr(abnormality, attr)[0]

        # Test A: the physiological siblings used in the TPC investigation
        sib_names = DOMAIN_SIBLINGS[name]
        sib_idx = [feature_names.index(s) for s in sib_names]
        controls = np.column_stack([X_test[:, si, :].mean(axis=1) for si in sib_idx])
        partial_domain = partial_correlation(abnormality, attr, controls)

        # Test B: this model's own top-2 dominant features (excluding the sensor itself)
        own_top = [int(j) for j in ranked if j != idx][:2]
        own_names = [feature_names[j] for j in own_top]
        controls_own = np.column_stack([X_test[:, j, :].mean(axis=1) for j in own_top])
        partial_own = partial_correlation(abnormality, attr, controls_own)

        print(f"{name:6s} raw={raw_rho:+.4f} | domain {sib_names}: {partial_domain:+.4f} "
              f"({verdict(raw_rho, partial_domain)}) | own-dominant {own_names}: {partial_own:+.4f} "
              f"({verdict(raw_rho, partial_own)})")

        results.append({
            "model": model_name, "sensor": name, "raw_rho": raw_rho,
            "domain_siblings": " ".join(sib_names), "partial_domain": partial_domain,
            "domain_verdict": verdict(raw_rho, partial_domain),
            "own_dominant": " ".join(own_names), "partial_own_dominant": partial_own,
            "own_verdict": verdict(raw_rho, partial_own),
        })

results = []

# --- TPC (fixed) ---
tpc_X = np.load("tpc_X_scaled.npy")
tpc_test_rows = np.load("tpc_test_rows.npy")
tpc_X_test = tpc_X[tpc_test_rows]
tpc_shap = np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1)
tpc_attr = np.abs(tpc_shap).mean(axis=2)
run_check(tpc_X_test, tpc_attr, "TPC (fixed)", results)

# --- BAT (healthy-baseline IG) ---
bat_X = np.load("bat_X_scaled.npy")
bat_test_rows = np.load("bat_test_rows.npy")
bat_X_test = np.transpose(bat_X[bat_test_rows], (0, 2, 1))  # -> (n, 48, 24)
bat_ig = np.load("bat_ig_values_full_testset_healthybaseline.npy")
bat_attr = np.abs(bat_ig).mean(axis=2)
run_check(bat_X_test, bat_attr, "BAT", results)

pd.DataFrame(results).to_csv("hypothesis_h5_all_sensors_both_models.csv", index=False)
print("\nSaved: hypothesis_h5_all_sensors_both_models.csv")
