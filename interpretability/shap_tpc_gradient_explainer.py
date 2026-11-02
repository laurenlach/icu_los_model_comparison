import torch
import numpy as np
import shap
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.dl_models.tpc import TPC

# ---------------------------------------------------------------------------
# 1. Load the trained TPC model
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 2. Load data - same background/explain setup as before
# ---------------------------------------------------------------------------
X = np.load("tpc_X_scaled.npy")
static = np.load("tpc_static_scaled.npy")
train_rows = np.load("tpc_train_rows.npy")
test_rows = np.load("tpc_test_rows.npy")

np.random.seed(42)
background_idx = np.random.choice(train_rows, size=50, replace=False)
X_background = torch.tensor(X[background_idx], dtype=torch.float32)

explain_idx = test_rows[:5]
X_explain = torch.tensor(X[explain_idx], dtype=torch.float32)
static_explain = torch.tensor(static[explain_idx], dtype=torch.float32)

# ---------------------------------------------------------------------------
# 3. Same wrapper as before
# ---------------------------------------------------------------------------
class TPCWrapperSingleStatic(torch.nn.Module):
    def __init__(self, model, static_value):
        super().__init__()
        self.model = model
        self.static_value = static_value

    def forward(self, x):
        batch_size = x.shape[0]
        static_repeated = self.static_value.repeat(batch_size, 1)
        return self.model(x, static_repeated)

# ---------------------------------------------------------------------------
# 4. Try GradientExplainer instead of DeepExplainer - same single-patient
#    test as before, to see if this method handles TPC's architecture.
# ---------------------------------------------------------------------------
patient_static = static_explain[0:1]
wrapper = TPCWrapperSingleStatic(model, patient_static)

print("\nBuilding SHAP GradientExplainer for patient 0...")
explainer = shap.GradientExplainer(wrapper, X_background)

print("Computing SHAP values for patient 0 (this may take a moment)...")
shap_values = explainer.shap_values(X_explain[0:1])

print("\nSHAP values computed successfully - no additivity error this time.")
print("SHAP values shape:", np.array(shap_values).shape)
print("(Expect something like (1, 48, 24, 1) - one value per sensor per timestep)")

# Quick sanity check: sum of shap values should roughly match
# (prediction - average background prediction)
with torch.no_grad():
    prediction = wrapper(X_explain[0:1]).item()
    background_avg_pred = wrapper(X_background).mean().item()

shap_sum = np.array(shap_values).sum()
print(f"\nSanity check:")
print(f"Model prediction for this patient: {prediction:.4f}")
print(f"Average background prediction: {background_avg_pred:.4f}")
print(f"Difference (what SHAP values should sum to): {prediction - background_avg_pred:.4f}")
print(f"Actual sum of SHAP values: {shap_sum:.4f}")