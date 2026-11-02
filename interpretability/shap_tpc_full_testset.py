import torch
import numpy as np
import pandas as pd
import shap
import time
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
# 2. Load data
# ---------------------------------------------------------------------------
X = np.load("tpc_X_scaled.npy")
static = np.load("tpc_static_scaled.npy")
train_rows = np.load("tpc_train_rows.npy")
test_rows = np.load("tpc_test_rows.npy")
stay_id_order = pd.read_csv("bat_stay_id_order.csv")

np.random.seed(42)
background_idx = np.random.choice(train_rows, size=50, replace=False)
X_background = torch.tensor(X[background_idx], dtype=torch.float32)

print(f"\nRunning SHAP for ALL {len(test_rows)} test patients...")
print("This should take roughly 15-20 minutes based on earlier timing.")

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
# 3. Run SHAP for every test patient, with periodic progress updates and
#    resilience - if one patient somehow fails, log it and continue rather
#    than losing the whole run.
# ---------------------------------------------------------------------------
all_shap_values = []
all_patient_stay_ids = []
failed_rows = []

start_time = time.time()
progress_interval = 500  # print progress every 500 patients, not every one

for i, patient_row in enumerate(test_rows):
    try:
        patient_static = torch.tensor(static[patient_row:patient_row+1], dtype=torch.float32)
        patient_X = torch.tensor(X[patient_row:patient_row+1], dtype=torch.float32)

        wrapper = TPCWrapperSingleStatic(model, patient_static)
        explainer = shap.GradientExplainer(wrapper, X_background)
        shap_values = explainer.shap_values(patient_X)

        all_shap_values.append(np.array(shap_values)[0])
        all_patient_stay_ids.append(stay_id_order["stay_id"].iloc[patient_row])
    except Exception as e:
        print(f"  WARNING: patient row {patient_row} failed - {e}")
        failed_rows.append(patient_row)

    if (i + 1) % progress_interval == 0 or (i + 1) == len(test_rows):
        elapsed = time.time() - start_time
        avg_per_patient = elapsed / (i + 1)
        remaining = (len(test_rows) - (i + 1)) * avg_per_patient
        print(f"  {i+1}/{len(test_rows)} done - elapsed: {elapsed/60:.1f} min, "
              f"est. remaining: {remaining/60:.1f} min")

total_elapsed = time.time() - start_time
print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
print(f"Successfully explained: {len(all_shap_values)} / {len(test_rows)} patients")
if failed_rows:
    print(f"Failed patients (rows): {failed_rows}")

# ---------------------------------------------------------------------------
# 4. Stack and save results
# ---------------------------------------------------------------------------
all_shap_values = np.stack(all_shap_values)
print("\nFinal SHAP values array shape:", all_shap_values.shape)

np.save("tpc_shap_values_full_testset.npy", all_shap_values)
pd.DataFrame({"stay_id": all_patient_stay_ids}).to_csv("tpc_shap_full_testset_stay_ids.csv", index=False)

print("\nSaved: tpc_shap_values_full_testset.npy, tpc_shap_full_testset_stay_ids.csv")
print("Done.")