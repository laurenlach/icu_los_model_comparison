import torch
import numpy as np
from scipy.stats import spearmanr
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.dl_models.tpc import TPC

feature_names = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun",
    "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2",
    "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
    "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt",
    "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine", "wbc"
]
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
model.load_state_dict(torch.load("tpc_best_model_alignmentpenalty.pt", map_location="cpu"))
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

# Part 1: dose-response via direct ablation 
print("=== Dose-response (direct ablation): alignment-penalty TPC vs original TPC ===")
ORIGINAL_SIGNED_RHO = { 
    "map": -0.890, "bun": +0.878, "crea": -0.747, "bili": +0.311,
    "plt": +0.130, "fio2": +0.520, "o2sat": -0.723,
}
for name, info in SOFA_SENSORS.items():
    idx = info["idx"]
    raw_full = X_test[:, idx, :].mean(axis=1)
    abnormality = -raw_full if info["worse"] == "low" else raw_full

    cohort_mean = float(X_test[:, idx, :].mean())
    X_abl = X_test.copy()
    X_abl[:, idx, :] = cohort_mean
    abl_preds = predict(X_abl, static_test)

    signed_effect = baseline_preds - abl_preds
    rho_signed = spearmanr(abnormality, signed_effect)[0]
    mean_abs_effect = np.abs(signed_effect).mean()

    orig = ORIGINAL_SIGNED_RHO[name]
    moved = "IMPROVED" if (orig < 0 and rho_signed > orig) or (orig > 0 and rho_signed > orig) else \
            "WORSE" if (orig < 0 and rho_signed < orig) or (orig > 0 and rho_signed < orig) else "same"
    print(f"{name:6s} alignment-penalty rho={rho_signed:+.3f} (mean|effect|={mean_abs_effect:.3f}d) "
          f"| original TPC rho={orig:+.3f} | {moved}")


# Part 2: dangerous-case rate
y_test = np.load("tpc_test_targets_fixed.npy")
error = baseline_preds - y_test
dangerous = error <= -3.0
bias = error.mean()

print(f"\n=== Safety: alignment-penalty TPC vs original TPC ===")
print(f"Alignment-penalty TPC -- bias: {bias:+.4f} days | dangerous cases: {dangerous.sum()} ({100*dangerous.mean():.2f}%)")
print(f"Original TPC (fixed)  -- bias: +0.0378 days | dangerous cases: 1125 (10.85%)")