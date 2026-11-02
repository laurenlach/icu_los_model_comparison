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
# 2. Load data - a small "background" set (SHAP needs reference examples
#    to compare against) and a small set of patients to actually explain
# ---------------------------------------------------------------------------
X = np.load("tpc_X_scaled.npy")
static = np.load("tpc_static_scaled.npy")
train_rows = np.load("tpc_train_rows.npy")
test_rows = np.load("tpc_test_rows.npy")

# Background: a modest random sample from TRAINING data (SHAP's reference
# distribution - represents "typical" patients to compare against)
np.random.seed(42)
background_idx = np.random.choice(train_rows, size=50, replace=False)
X_background = torch.tensor(X[background_idx], dtype=torch.float32)

# Patients we actually want explanations for: 5 from the TEST set
explain_idx = test_rows[:5]
X_explain = torch.tensor(X[explain_idx], dtype=torch.float32)
static_explain = torch.tensor(static[explain_idx], dtype=torch.float32)

print("Background shape:", X_background.shape)
print("Explain shape:", X_explain.shape)

# ---------------------------------------------------------------------------
# 3. Same wrapper as before - but now we need static fixed for the
#    background too. Since patients being explained have DIFFERENT static
#    values, we run SHAP once PER PATIENT, using that patient's own
#    static values fixed inside the wrapper for that run.
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
# 4. Run SHAP for just the FIRST patient, as a test, before looping
#    over all 5 - want to see this work correctly once before scaling up.
# ---------------------------------------------------------------------------
patient_static = static_explain[0:1]
wrapper = TPCWrapperSingleStatic(model, patient_static)

print("\nBuilding SHAP DeepExplainer for patient 0...")
explainer = shap.DeepExplainer(wrapper, X_background)

print("Computing SHAP values for patient 0 (this may take a moment)...")
shap_values = explainer.shap_values(X_explain[0:1])

print("\nSHAP values shape:", np.array(shap_values).shape)
print("(Expect something like (1, 48, 24, 1) - one value per sensor per timestep)")