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

model = TPC(
    input_size=(64, 48, 24), static_size=4, n_layers=3, kernel_size=3,
    temp_kernels=4, point_size=8, dropout=0.1, run_mode=RunMode.regression,
)
model.load_state_dict(torch.load("tpc_best_model_fixed.pt", map_location="cpu"))
model.eval()

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

results = []
for name, info in SOFA_SENSORS.items():
    idx = info["idx"]
    raw_full = X_test[:, idx, :].mean(axis=1)
    abnormality = -raw_full if info["worse"] == "low" else raw_full

    cohort_mean = float(X_test[:, idx, :].mean())   # same rule as the BAT ablation
    X_abl = X_test.copy()
    X_abl[:, idx, :] = cohort_mean
    abl_preds = predict(X_abl, static_test)

    signed_effect = baseline_preds - abl_preds
    abs_effect = np.abs(signed_effect)
    rho_abs = spearmanr(abnormality, abs_effect)[0]
    rho_signed = spearmanr(abnormality, signed_effect)[0]

    results.append({"sensor": name, "cohort_mean_used": cohort_mean,
                    "ablation_abs_dose_response": rho_abs,
                    "ablation_signed_dose_response": rho_signed,
                    "mean_abs_effect_days": abs_effect.mean()})
    print(f"{name:6s} (mean used={cohort_mean:+.3f}) ablation |effect| rho={rho_abs:+.4f} "
          f"| signed rho={rho_signed:+.4f} | mean |effect|={abs_effect.mean():.4f} days")

pd.DataFrame(results).to_csv("ablation_dose_response_tpc_cohortmean.csv", index=False)
print("\nSaved: ablation_dose_response_tpc_cohortmean.csv")