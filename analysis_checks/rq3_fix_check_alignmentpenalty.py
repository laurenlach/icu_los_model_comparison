import torch
import numpy as np
import pandas as pd
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.dl_models.tpc import TPC

model = TPC(
    input_size=(64, 48, 24), static_size=4, n_layers=3, kernel_size=3,
    temp_kernels=4, point_size=8, dropout=0.1, run_mode=RunMode.regression,
)
model.load_state_dict(torch.load("tpc_best_model_alignmentpenalty.pt", map_location="cpu"))
model.eval()

X = np.load("tpc_X_scaled.npy")
static = np.load("tpc_static_scaled.npy")
test_rows = np.load("tpc_test_rows.npy")
y_test = np.load("tpc_test_targets_fixed.npy")
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

ap_pred = predict(X_test, static_test)

# Reuse the same per-patient table structure as before
df = pd.read_csv("rq3_per_patient_phenotype_errors.csv")
assert len(df) == len(ap_pred), "Row count mismatch -- check test set alignment"
df["ap_pred"] = ap_pred
df["ap_abs_error"] = np.abs(df.ap_pred - df.tpc_true)
df["ap_error"] = df.ap_pred - df.tpc_true
df["ap_dangerous"] = df.ap_error <= -3.0

summary = df.dropna(subset=["phenotype"]).groupby("phenotype").agg(
    n=("tpc_true", "size"),
    tpc_mae=("tpc_abs_error", "mean"),
    ap_mae=("ap_abs_error", "mean"),
    tpc_dangerous_rate=("tpc_dangerous", "mean"),
    ap_dangerous_rate=("ap_dangerous", "mean"),
).reset_index()
summary["mae_change"] = summary["ap_mae"] - summary["tpc_mae"]        # negative = improved
summary["dangerous_change"] = summary["ap_dangerous_rate"] - summary["tpc_dangerous_rate"]  # negative = improved

print("=== Alignment-penalty TPC vs original TPC, per phenotype ===")
print(summary.to_string(index=False))

n_mae_improved = (summary["mae_change"] < 0).sum()
n_dangerous_improved = (summary["dangerous_change"] < 0).sum()
print(f"\nMAE improved in {n_mae_improved}/25 phenotypes")
print(f"Dangerous-case rate improved in {n_dangerous_improved}/25 phenotypes")
print(f"\nOverall: AP MAE={summary['ap_mae'].mean():.3f} vs original TPC MAE={summary['tpc_mae'].mean():.3f}")
print(f"Overall: AP dangerous rate={df['ap_dangerous'].mean():.3f} vs original TPC rate={df['tpc_dangerous'].mean():.3f}")

summary.to_csv("rq3_alignmentpenalty_vs_original_by_phenotype.csv", index=False)
print("\nSaved: rq3_alignmentpenalty_vs_original_by_phenotype.csv")