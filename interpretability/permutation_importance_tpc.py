import numpy as np
import torch
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.dl_models.tpc import TPC

feature_names = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun",
    "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2",
    "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
    "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt",
    "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine", "wbc"
]

model = TPC(
    input_size=(64, 48, 24), static_size=4, n_layers=3, kernel_size=3,
    temp_kernels=4, point_size=8, dropout=0.1, run_mode=RunMode.regression,
)
model.load_state_dict(torch.load("tpc_best_model_200epoch.pt", map_location="cpu"))
model.eval()
print("Trained TPC model loaded successfully.")

X = np.load("tpc_X_scaled.npy")              # (51835, 48, 24) - features axis=1
static = np.load("tpc_static_scaled.npy")
test_rows = np.load("tpc_test_rows.npy")
y_test = np.load("tpc_test_targets_200epoch.npy")  # true LOS, already in test-row order

X_test = X[test_rows]
static_test = static[test_rows]
n_test = len(test_rows)

def predict(X_arr, static_arr, batch_size=1024):
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_arr), batch_size):
            xb = torch.tensor(X_arr[i:i+batch_size], dtype=torch.float32)
            sb = torch.tensor(static_arr[i:i+batch_size], dtype=torch.float32)
            out = model(xb, sb).squeeze(-1)
            preds.append(out.numpy())
    return np.concatenate(preds)

baseline_preds = predict(X_test, static_test)
baseline_mae = np.abs(baseline_preds - y_test).mean()
print(f"Baseline MAE (unperturbed): {baseline_mae:.4f} days\n")

N_REPEATS = 5
results = []
np.random.seed(42)

for f_idx, f_name in enumerate(feature_names):
    mae_increases = []
    for rep in range(N_REPEATS):
        X_perm = X_test.copy()
        shuffle_idx = np.random.permutation(n_test)
        X_perm[:, f_idx, :] = X_test[shuffle_idx, f_idx, :]  # swap whole 24h trajectory across patients

        perm_preds = predict(X_perm, static_test)
        perm_mae = np.abs(perm_preds - y_test).mean()
        mae_increases.append(perm_mae - baseline_mae)

    mean_increase = np.mean(mae_increases)
    std_increase = np.std(mae_increases)
    results.append((f_name, mean_increase, std_increase))
    print(f"{f_name}: MAE increase = {mean_increase:.4f} (+/- {std_increase:.4f})")

results.sort(key=lambda x: -x[1])
print("\n=== TPC Permutation Importance - Top 10 (largest MAE increase when scrambled) ===")
for name, inc, std in results[:10]:
    print(f"{name}: +{inc:.4f} days (+/- {std:.4f})")

import pandas as pd
pd.DataFrame(results, columns=["feature", "mae_increase", "mae_increase_std"]).to_csv(
    "permutation_importance_tpc.csv", index=False
)
print("\nSaved: permutation_importance_tpc.csv")