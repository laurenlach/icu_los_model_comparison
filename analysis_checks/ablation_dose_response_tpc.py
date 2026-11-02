import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.dl_models.tpc import TPC

SOFA_SENSORS = {
    "map":   {"idx": 27, "worse": "low"},
    "bun":   {"idx": 9,  "worse": "high"},
    "crea":  {"idx": 15, "worse": "high"},
    "bili":  {"idx": 6,  "worse": "high"},
    "plt":   {"idx": 39, "worse": "low"},
    "fio2":  {"idx": 19, "worse": "high"},
    "o2sat": {"idx": 35, "worse": "low"},
}

# --- Load fixed TPC (same settings as permutation_importance_tpc_fixed.py) ---
model = TPC(
    input_size=(64, 48, 24), static_size=4, n_layers=3, kernel_size=3,
    temp_kernels=4, point_size=8, dropout=0.1, run_mode=RunMode.regression,
)
model.load_state_dict(torch.load("tpc_best_model_fixed.pt", map_location="cpu"))
model.eval()
print("Fixed TPC model loaded successfully.")

X = np.load("tpc_X_scaled.npy")
static = np.load("tpc_static_scaled.npy")
test_rows = np.load("tpc_test_rows.npy")
X_test = X[test_rows]
static_test = static[test_rows]

def predict(X_arr, static_arr, batch_size=1024):
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_arr), batch_size):
            xb = torch.tensor(X_arr[i:i+batch_size], dtype=torch.float32)
            sb = torch.tensor(static_arr[i:i+batch_size], dtype=torch.float32)
            preds.append(model(xb, sb).squeeze(-1).numpy())
    return np.concatenate(preds)

baseline_preds = predict(X_test, static_test)

# SHAP attribution (for side-by-side comparison with the ablation result)
tpc_shap = np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1)
tpc_attr = np.abs(tpc_shap).mean(axis=2)

results = []
print("\nRunning per-patient ablation (sensor replaced by cohort average, i.e. z-score 0)...")
for name, info in SOFA_SENSORS.items():
    idx = info["idx"]

    # Same abnormality definition as dose_response_check_fixed.py
    raw_full = X_test[:, idx, :].mean(axis=1)
    abnormality = -raw_full if info["worse"] == "low" else raw_full

    # Ablate this one sensor for every patient
    X_abl = X_test.copy()
    X_abl[:, idx, :] = 0.0
    abl_preds = predict(X_abl, static_test)

    signed_effect = baseline_preds - abl_preds   # + means the real readings pushed LOS up
    abs_effect = np.abs(signed_effect)

    rho_shap, _ = spearmanr(abnormality, tpc_attr[:, idx])
    rho_abs, p_abs = spearmanr(abnormality, abs_effect)
    rho_signed, p_signed = spearmanr(abnormality, signed_effect)

    results.append({
        "sensor": name,
        "shap_dose_response": rho_shap,
        "ablation_abs_dose_response": rho_abs,
        "ablation_signed_dose_response": rho_signed,
        "mean_abs_effect_days": abs_effect.mean(),
    })
    print(f"{name:6s} SHAP rho={rho_shap:+.4f} | ablation |effect| rho={rho_abs:+.4f} "
          f"| ablation signed rho={rho_signed:+.4f} | mean |effect|={abs_effect.mean():.4f} days")

pd.DataFrame(results).to_csv("ablation_dose_response_tpc.csv", index=False)
print("\nSaved: ablation_dose_response_tpc.csv")