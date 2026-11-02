import numpy as np
import torch
from icu_benchmarks.models.dl_models.bat import BAT, RegressionHead

feature_names = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun",
    "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2",
    "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
    "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt",
    "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine", "wbc"
]

model = BAT(
    input_size=(64, 48, 24), value_embed_size=128, layers=2, heads=1,
    dropout=0.36427, attn_dropout=0.20659, use_mask=False,
    prediction_head=RegressionHead, prediction_head_kwargs={"output_dim": 1},
    static_count=4,
)
model.load_state_dict(torch.load("bat_best_model.pt", map_location="cpu"))
model.eval()
print("Trained BAT model loaded successfully.")

X = np.load("bat_X_scaled.npy")              # (51835, 24, 48) - features axis=2
static = np.load("bat_static_scaled.npy")
mask = np.load("bat_mask.npy")
test_rows = np.load("bat_test_rows.npy")
y = np.load("bat_y.npy")

X_test_raw = X[test_rows]        # (n, 24, 48)
static_test = static[test_rows]
mask_test_raw = mask[test_rows]
y_test = y[test_rows]
n_test = len(test_rows)
time_arr = np.arange(24, dtype=np.float32)

def predict(X_raw, static_arr, mask_raw, batch_size=256):
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_raw), batch_size):
            xb_raw = X_raw[i:i+batch_size]
            mb_raw = mask_raw[i:i+batch_size]
            sb = static_arr[i:i+batch_size]
            bsz = len(xb_raw)

            xb = torch.tensor(np.transpose(xb_raw, (0, 2, 1)), dtype=torch.float32)
            mb = torch.tensor(np.transpose(mb_raw, (0, 2, 1)), dtype=torch.float32)
            sb_t = torch.tensor(sb, dtype=torch.float32)
            tb = torch.tensor(np.tile(time_arr, (bsz, 1)), dtype=torch.float32)

            out = model(xb, sb_t, tb, mb).squeeze(-1)
            preds.append(out.numpy())
    return np.concatenate(preds)

baseline_preds = predict(X_test_raw, static_test, mask_test_raw)
baseline_mae = np.abs(baseline_preds - y_test).mean()
print(f"Baseline MAE (unperturbed): {baseline_mae:.4f} days\n")

N_REPEATS = 5
results = []
np.random.seed(42)

for f_idx, f_name in enumerate(feature_names):
    mae_increases = []
    for rep in range(N_REPEATS):
        X_perm = X_test_raw.copy()
        shuffle_idx = np.random.permutation(n_test)
        X_perm[:, :, f_idx] = X_test_raw[shuffle_idx, :, f_idx]  # swap whole 24h trajectory

        perm_preds = predict(X_perm, static_test, mask_test_raw)
        perm_mae = np.abs(perm_preds - y_test).mean()
        mae_increases.append(perm_mae - baseline_mae)

    mean_increase = np.mean(mae_increases)
    std_increase = np.std(mae_increases)
    results.append((f_name, mean_increase, std_increase))
    print(f"{f_name}: MAE increase = {mean_increase:.4f} (+/- {std_increase:.4f})")

results.sort(key=lambda x: -x[1])
print("\n=== BAT Permutation Importance - Top 10 (largest MAE increase when scrambled) ===")
for name, inc, std in results[:10]:
    print(f"{name}: +{inc:.4f} days (+/- {std:.4f})")

import pandas as pd
pd.DataFrame(results, columns=["feature", "mae_increase", "mae_increase_std"]).to_csv(
    "permutation_importance_bat.csv", index=False
)
print("\nSaved: permutation_importance_bat.csv")