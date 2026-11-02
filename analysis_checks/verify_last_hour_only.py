import numpy as np
import torch
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.dl_models.tpc import TPC

model = TPC(
    input_size=(64, 48, 24), static_size=4, n_layers=3, kernel_size=3,
    temp_kernels=4, point_size=8, dropout=0.1, run_mode=RunMode.regression,
)
model.load_state_dict(torch.load("tpc_best_model_200epoch.pt", map_location="cpu"))
model.eval()

X = np.load("tpc_X_scaled.npy")
static = np.load("tpc_static_scaled.npy")
test_rows = np.load("tpc_test_rows.npy")

X_test = X[test_rows][:500]        # small sample, this is just a sanity check
static_test = static[test_rows][:500]

def predict(X_arr, static_arr):
    with torch.no_grad():
        xb = torch.tensor(X_arr, dtype=torch.float32)
        sb = torch.tensor(static_arr, dtype=torch.float32)
        return model(xb, sb).squeeze(-1).numpy()

baseline = predict(X_test, static_test)

# Test A: ZERO OUT hours 0-22, keep hour 23 (last) untouched
X_only_last_hour = X_test.copy()
X_only_last_hour[:, :, 0:23] = 0.0
pred_only_last = predict(X_only_last_hour, static_test)

# Test B: ZERO OUT hour 23 (last) only, keep hours 0-22 untouched
X_missing_last_hour = X_test.copy()
X_missing_last_hour[:, :, 23] = 0.0
pred_missing_last = predict(X_missing_last_hour, static_test)

print("Baseline vs zeroing hours 0-22 (keep only last hour):")
print("  Max abs difference from baseline:", np.abs(pred_only_last - baseline).max())
print("  Mean abs difference from baseline:", np.abs(pred_only_last - baseline).mean())

print("\nBaseline vs zeroing ONLY hour 23 (keep hours 0-22):")
print("  Max abs difference from baseline:", np.abs(pred_missing_last - baseline).max())
print("  Mean abs difference from baseline:", np.abs(pred_missing_last - baseline).mean())