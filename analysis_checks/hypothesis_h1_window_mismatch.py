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

X = np.load("tpc_X_scaled.npy")           # (51835, 48, 24)
test_rows = np.load("tpc_test_rows.npy")
X_test = X[test_rows]                     # (10367, 48, 24)

tpc_shap = np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1)  # (10367, 48, 24)
tpc_attr_mean_full = np.abs(tpc_shap).mean(axis=2)  # mean over all 24h (original method)

results = []
for name, info in SOFA_SENSORS.items():
    idx = info["idx"]

    # ORIGINAL: abnormality averaged over ALL 24 hours
    raw_full = X_test[:, idx, :].mean(axis=1)
    abnormality_full = -raw_full if info["worse"] == "low" else raw_full
    rho_full, p_full = spearmanr(abnormality_full, tpc_attr_mean_full[:, idx])

    # NEW (H1 test): abnormality averaged over ONLY hours 9-23 (TPC's real receptive field)
    raw_window = X_test[:, idx, 9:24].mean(axis=1)
    abnormality_window = -raw_window if info["worse"] == "low" else raw_window
    attr_window = np.abs(tpc_shap[:, idx, 9:24]).mean(axis=1)
    rho_window, p_window = spearmanr(abnormality_window, attr_window)

    results.append({
        "sensor": name,
        "rho_full_24h (original)": round(rho_full, 4),
        "rho_hours_9_23_only (H1 test)": round(rho_window, 4),
        "change": "IMPROVED (less negative/more positive)" if rho_window > rho_full else "WORSE or same",
    })

results_df = pd.DataFrame(results)
print("=== H1 Test: Does restricting to TPC's real receptive field (hours 9-23) fix the backwards pattern? ===\n")
print(results_df.to_string(index=False))

results_df.to_csv("hypothesis_h1_window_mismatch.csv", index=False)
print("\nSaved: hypothesis_h1_window_mismatch.csv")