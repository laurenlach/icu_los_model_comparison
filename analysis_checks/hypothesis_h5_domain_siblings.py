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

mask = np.load("bat_mask.npy")
bat_test_rows = np.load("bat_test_rows.npy")
mask_test = mask[bat_test_rows]
observed_pct = mask_test.mean(axis=(0, 1)) * 100

# Check the observed-rate of the previously-selected "siblings"
print("=== Checking whether previously-chosen siblings are actually meaningful (well-measured) ===")
for f in ["methb", "crp", "wbc", "fgn"]:
    idx = feature_names.index(f)
    print(f"{f}: {observed_pct[idx]:.2f}% observed")

def partial_correlation(x, y, controls):
    reg_x = LinearRegression().fit(controls, x)
    resid_x = x - reg_x.predict(controls)
    reg_y = LinearRegression().fit(controls, y)
    resid_y = y - reg_y.predict(controls)
    return spearmanr(resid_x, resid_y)

tpc_X = np.load("tpc_X_scaled.npy")
tpc_test_rows = np.load("tpc_test_rows.npy")
tpc_X_test = tpc_X[tpc_test_rows]
tpc_shap = np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1)
tpc_attr = np.abs(tpc_shap).mean(axis=2)

DOMAIN_SIBLINGS = {
    "o2sat": {"idx": 35, "worse": "low", "siblings": ["fio2", "pco2", "po2", "resp"]},
    "plt":   {"idx": 39, "worse": "low", "siblings": ["ptt", "inr_pt", "fgn"]},
}

print("\n=== Re-test with physiologically-sensible siblings instead of correlation-selected ones ===")
results = []
for name, info in DOMAIN_SIBLINGS.items():
    idx = info["idx"]
    raw_full = tpc_X_test[:, idx, :].mean(axis=1)
    abnormality = -raw_full if info["worse"] == "low" else raw_full
    attr = tpc_attr[:, idx]

    raw_rho, _ = spearmanr(abnormality, attr)

    sibling_idx = [feature_names.index(s) for s in info["siblings"]]
    controls = np.column_stack([tpc_X_test[:, si, :].mean(axis=1) for si in sibling_idx])
    partial_rho, partial_p = partial_correlation(abnormality, attr, controls)

    print(f"{name} (domain siblings: {info['siblings']}): raw={raw_rho:.4f}, partial={partial_rho:.4f}, "
          f"{'IMPROVED' if partial_rho > raw_rho else 'no improvement'}")
    results.append({"sensor": name, "raw_rho": raw_rho, "partial_rho_domain": partial_rho})

pd.DataFrame(results).to_csv("hypothesis_h5_domain_siblings.csv", index=False)
print("\nSaved: hypothesis_h5_domain_siblings.csv")