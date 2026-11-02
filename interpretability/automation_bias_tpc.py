import torch
import numpy as np
import pandas as pd
import shap
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.dl_models.tpc import TPC


# 1. Load the trained TPC model 
model = TPC(
    input_size=(64, 48, 24),
    static_size=4,
    n_layers=3,
    kernel_size=3,
    temp_kernels=4,
    point_size=8,
    dropout=0.1,
    run_mode=RunMode.regression,
)
model.load_state_dict(torch.load("tpc_best_model_200epoch.pt", map_location="cpu"))
model.eval()
print("Trained TPC model loaded successfully.")

# 2. Load data

X = np.load("tpc_X_scaled.npy")
static = np.load("tpc_static_scaled.npy")
train_rows = np.load("tpc_train_rows.npy")
test_rows = np.load("tpc_test_rows.npy")

np.random.seed(42)
background_idx = np.random.choice(train_rows, size=50, replace=False)
X_background = torch.tensor(X[background_idx], dtype=torch.float32)

class TPCWrapperSingleStatic(torch.nn.Module):
    def __init__(self, model, static_value):
        super().__init__()
        self.model = model
        self.static_value = static_value

    def forward(self, x):
        batch_size = x.shape[0]
        static_repeated = self.static_value.repeat(batch_size, 1)
        return self.model(x, static_repeated)


# 3. Subsample 200 test patients

n_sample = 200
sample_rows = np.random.choice(test_rows, size=n_sample, replace=False)

NOISE_STD = 0.03  # same noise level as BAT's check

pred_diffs = []
attr_shift_fractions = []
failed_rows = []

for i, row in enumerate(sample_rows):
    try:
        patient_static = torch.tensor(static[row:row+1], dtype=torch.float32)
        patient_X_np = X[row:row+1]  # (1, 48, 24) or (1, 24, 48) - matches training shape
        patient_X = torch.tensor(patient_X_np, dtype=torch.float32)

        wrapper = TPCWrapperSingleStatic(model, patient_static)

        # --- Original prediction and SHAP attribution ---
        with torch.no_grad():
            pred_original = wrapper(patient_X).item()

        explainer = shap.GradientExplainer(wrapper, X_background)
        shap_original = np.array(explainer.shap_values(patient_X))[0]

        # --- Add noise (whole array - no mask available for TPC) ---
        noise = np.random.normal(0, NOISE_STD, size=patient_X_np.shape)
        X_noisy_np = patient_X_np + noise
        X_noisy = torch.tensor(X_noisy_np, dtype=torch.float32)

        with torch.no_grad():
            pred_noisy = wrapper(X_noisy).item()

        shap_noisy = np.array(explainer.shap_values(X_noisy))[0]

        # --- Measure the shift, same metric as BAT's check ---
        pred_diffs.append(abs(pred_noisy - pred_original))

        attr_orig_mag = np.abs(shap_original).sum()
        attr_diff_mag = np.abs(shap_noisy - shap_original).sum()
        attr_shift_fractions.append(attr_diff_mag / (attr_orig_mag + 1e-8))

    except Exception as e:
        print(f"  WARNING: patient row {row} failed - {e}")
        failed_rows.append(row)

    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{n_sample} patients...")

pred_diffs = np.array(pred_diffs)
attr_shift_fractions = np.array(attr_shift_fractions)

print("\n=== TPC Automation-Bias / Stability Check ===")
print(f"Successfully processed: {len(pred_diffs)}/{n_sample} "
      f"({len(failed_rows)} failed)")
print(f"Mean prediction shift under noise: {pred_diffs.mean():.4f} days "
      f"(median: {np.median(pred_diffs):.4f})")
print(f"Mean attribution shift (fraction of original magnitude): "
      f"{attr_shift_fractions.mean():.4f} (median: {np.median(attr_shift_fractions):.4f})")

np.save("tpc_automation_bias_pred_diffs.npy", pred_diffs)
np.save("tpc_automation_bias_attr_shifts.npy", attr_shift_fractions)
print("\nSaved: tpc_automation_bias_pred_diffs.npy, tpc_automation_bias_attr_shifts.npy")