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
test_rows = np.load("tpc_test_rows.npy")

# Just 5 patients for this first test
sample_idx = test_rows[:5]
X_sample = torch.tensor(X[sample_idx], dtype=torch.float32)
static_sample = torch.tensor(static[sample_idx], dtype=torch.float32)

print("X_sample shape:", X_sample.shape)
print("static_sample shape:", static_sample.shape)


class TPCWrapperSingleStatic(torch.nn.Module):
    def __init__(self, model, static_value):
        super().__init__()
        self.model = model
        self.static_value = static_value  # shape: (1, 4), fixed for this wrapper

    def forward(self, x):
        batch_size = x.shape[0]
        static_repeated = self.static_value.repeat(batch_size, 1)
        return self.model(x, static_repeated)


test_wrapper = TPCWrapperSingleStatic(model, static_sample[0:1])
test_output = test_wrapper(X_sample[0:1])
print("\nWrapper test output:", test_output)

direct_output = model(X_sample[0:1], static_sample[0:1])
print("Direct model output (should match):", direct_output)