import torch
import numpy as np
import shap
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.dl_models.tpc import TPC


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


X = np.load("tpc_X_scaled.npy")
static = np.load("tpc_static_scaled.npy")
train_rows = np.load("tpc_train_rows.npy")
test_rows = np.load("tpc_test_rows.npy")


np.random.seed(42)
background_idx = np.random.choice(train_rows, size=50, replace=False)
X_background = torch.tensor(X[background_idx], dtype=torch.float32)

# Patients we actually want explanations for: 5 from the TEST set
explain_idx = test_rows[:5]
X_explain = torch.tensor(X[explain_idx], dtype=torch.float32)
static_explain = torch.tensor(static[explain_idx], dtype=torch.float32)

print("Background shape:", X_background.shape)
print("Explain shape:", X_explain.shape)


class TPCWrapperSingleStatic(torch.nn.Module):
    def __init__(self, model, static_value):
        super().__init__()
        self.model = model
        self.static_value = static_value

    def forward(self, x):
        batch_size = x.shape[0]
        static_repeated = self.static_value.repeat(batch_size, 1)
        return self.model(x, static_repeated)
    
patient_static = static_explain[0:1]
wrapper = TPCWrapperSingleStatic(model, patient_static)

print("\nBuilding SHAP DeepExplainer for patient 0...")
explainer = shap.DeepExplainer(wrapper, X_background)

print("Computing SHAP values for patient 0 (this may take a moment)...")
shap_values = explainer.shap_values(X_explain[0:1])

print("\nSHAP values shape:", np.array(shap_values).shape)
print("(Expect something like (1, 48, 24, 1) - one value per sensor per timestep)")