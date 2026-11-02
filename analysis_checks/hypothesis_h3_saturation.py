import numpy as np
import pandas as pd
import torch
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.dl_models.tpc import TPC

SOFA_SENSORS = {
    "map": {"idx": 27, "worse": "low"}, "bun": {"idx": 9, "worse": "high"},
    "crea": {"idx": 15, "worse": "high"}, "bili": {"idx": 6, "worse": "high"},
    "plt": {"idx": 39, "worse": "low"}, "fio2": {"idx": 19, "worse": "high"},
    "o2sat": {"idx": 35, "worse": "low"},
}

model = TPC(input_size=(64,48,24), static_size=4, n_layers=3, kernel_size=3,
            temp_kernels=4, point_size=8, dropout=0.1, run_mode=RunMode.regression)
model.load_state_dict(torch.load("tpc_best_model_fixed.pt", map_location="cpu"))
model.eval()

X = np.load("tpc_X_scaled.npy")
static = np.load("tpc_static_scaled.npy")
test_rows = np.load("tpc_test_rows.npy")

np.random.seed(42)
sample_idx = np.random.choice(test_rows, size=300, replace=False)
X_sample = X[sample_idx].copy()
static_sample = static[sample_idx]

sweep_values = np.linspace(-3, 3, 25)  # Z-scored range, covers "very low" to "very high"

print("=== H3 Test: Does TPC's prediction SATURATE (flatten) at extreme sensor values? ===\n")
results = []
for name, info in SOFA_SENSORS.items():
    idx = info["idx"]
    mean_preds = []
    for val in sweep_values:
        X_mod = X_sample.copy()
        X_mod[:, idx, :] = val  # force this sensor to a fixed value across all 24h, for all 300 patients
        with torch.no_grad():
            xb = torch.tensor(X_mod, dtype=torch.float32)
            sb = torch.tensor(static_sample, dtype=torch.float32)
            preds = model(xb, sb).squeeze(-1).numpy()
        mean_preds.append(preds.mean())

    mean_preds = np.array(mean_preds)
    # Local slope at the extreme end vs the middle - saturation = slope shrinks toward the extreme
    slope_middle = abs(mean_preds[13] - mean_preds[11]) / (sweep_values[13] - sweep_values[11])
    slope_extreme = abs(mean_preds[24] - mean_preds[22]) / (sweep_values[24] - sweep_values[22])

    print(f"{name}: slope near middle={slope_middle:.4f}, slope at extreme={slope_extreme:.4f}, "
          f"{'SATURATES (supports H3)' if slope_extreme < slope_middle * 0.5 else 'does not clearly saturate'}")
    results.append({"sensor": name, "sweep_values": sweep_values.tolist(), "mean_preds": mean_preds.tolist(),
                     "slope_middle": slope_middle, "slope_extreme": slope_extreme})

pd.DataFrame(results).to_csv("hypothesis_h3_saturation.csv", index=False)
print("\nSaved: hypothesis_h3_saturation.csv (full curves for plotting)")