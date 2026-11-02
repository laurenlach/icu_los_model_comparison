import torch
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.dl_models.tpc import TPC

model = TPC(
    input_size=(64, 48, 24), static_size=4, n_layers=3, kernel_size=3,
    temp_kernels=4, point_size=8, dropout=0.1, run_mode=RunMode.regression,
)
model.load_state_dict(torch.load("tpc_best_model_200epoch.pt", map_location="cpu"))
model.eval()

# Print the model architecture so we can see exactly how the final layer
# connects to the convolutional stack - specifically looking for anything
# that selects only the LAST timestep before the final prediction
print(model)