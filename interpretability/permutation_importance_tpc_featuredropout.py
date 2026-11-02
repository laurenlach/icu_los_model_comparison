import torch
import numpy as np
import pandas as pd
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
model.load_state_dict(torch.load("tpc_best_model_featuredropout.pt", map_location="cpu"))
model.eval()

X = np.load("tpc_X_scaled.npy")
static = np.load("tpc_static_scaled.npy")
test_rows = np.load("tpc_test_rows.npy")
y_test = np.load("tpc_featuredropout_test_targets.npy")
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

base_preds = predict(X_test, static_test)
base_mae = np.mean(np.abs(base_preds - y_test))
print(f"Baseline MAE (feature-dropout model): {base_mae:.4f} days\n")

rng = np.random.default_rng(42)
results = []
for i, name in enumerate(feature_names):
    X_perm = X_test.copy()
    perm_idx = rng.permutation(len(X_perm))
    X_perm[:, i, :] = X_test[perm_idx, i, :]   # shuffle this sensor's 24h series across patients
    perm_preds = predict(X_perm, static_test)
    perm_mae = np.mean(np.abs(perm_preds - y_test))
    increase = perm_mae - base_mae
    results.append({"feature": name, "mae_increase": increase})
    print(f"{name:10s} mae_increase={increase:+.4f}")

df = pd.DataFrame(results).sort_values("mae_increase", ascending=False)
df.to_csv("permutation_importance_tpc_featuredropout.csv", index=False)
print("\nTop 10 features:")
print(df.head(10).to_string(index=False))
print("\nSaved: permutation_importance_tpc_featuredropout.csv")
