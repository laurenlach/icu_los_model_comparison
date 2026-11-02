import torch
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
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

mask = np.load("bat_mask.npy")
mask_test = mask[np.load("bat_test_rows.npy")]      # (n, 24, 48), 1 = really measured

def predict(X_arr, static_arr, batch_size=1024):
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_arr), batch_size):
            xb = torch.tensor(X_arr[i:i+batch_size], dtype=torch.float32)
            sb = torch.tensor(static_arr[i:i+batch_size], dtype=torch.float32)
            preds.append(model(xb, sb).squeeze(-1).numpy())
    return np.concatenate(preds)

def partial_correlation(x, y, controls):
    rx = x - LinearRegression().fit(controls, x).predict(controls)
    ry = y - LinearRegression().fit(controls, y).predict(controls)
    return spearmanr(rx, ry)[0]

baseline = predict(X_test, static_test)
effects = {}
for name, info in SOFA_SENSORS.items():
    idx = info["idx"]
    X_abl = X_test.copy()
    X_abl[:, idx, :] = float(X_test[:, idx, :].mean())   # cohort-mean rule, same as before
    effects[name] = baseline - predict(X_abl, static_test)

np.save("tpc_ablation_signed_effects.npy", np.column_stack([effects[n] for n in SOFA_SENSORS]))

# ---------------- Part 1: all patients vs really-measured patients ----------------
print("=== 1. TPC ablation dose-response: all patients vs measured-only ===")
for name, info in SOFA_SENSORS.items():
    idx = info["idx"]
    raw = X_test[:, idx, :].mean(axis=1)
    abn = -raw if info["worse"] == "low" else raw
    measured = mask_test[:, :, idx].sum(axis=1) >= 1
    eff = effects[name]
    print(f"{name:6s} all: signed rho={spearmanr(abn, eff)[0]:+.3f}, mean|effect|={np.abs(eff).mean():.3f} d "
          f"| measured (n={int(measured.sum())}): signed rho={spearmanr(abn[measured], eff[measured])[0]:+.3f}, "
          f"mean|effect|={np.abs(eff[measured]).mean():.3f} d")

# ---------------- Part 2: is TPC's creatinine reversal really a BUN effect? ----------------
print("\n=== 2. TPC creatinine effect vs BUN ===")
crea = X_test[:, 15, :].mean(axis=1)
bun = X_test[:, 9, :].mean(axis=1)
eff = effects["crea"]
print(f"creatinine effect vs creatinine level:                   rho={spearmanr(crea, eff)[0]:+.3f}")
print(f"creatinine effect vs BUN level:                          rho={spearmanr(bun, eff)[0]:+.3f}")
print(f"creatinine effect vs creatinine, controlling for BUN:    rho={partial_correlation(crea, eff, bun.reshape(-1, 1)):+.3f}")
cuts = np.quantile(bun, [1/3, 2/3])
groups = {"low BUN": bun <= cuts[0], "mid BUN": (bun > cuts[0]) & (bun <= cuts[1]), "high BUN": bun > cuts[1]}
for g, keep in groups.items():
    if keep.sum() > 200:
        print(f"within {g:8s} (n={int(keep.sum()):5d}): creatinine effect vs creatinine rho={spearmanr(crea[keep], eff[keep])[0]:+.3f}")
print("\nSaved: tpc_ablation_signed_effects.npy")