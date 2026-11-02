import torch
import numpy as np
import pandas as pd
import shap
import time
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
stay_id_order = pd.read_csv("bat_stay_id_order.csv")

np.random.seed(42)
background_idx = np.random.choice(train_rows, size=50, replace=False)
X_background = torch.tensor(X[background_idx], dtype=torch.float32)


# 3. Sample real set of test patients to explain

n_patients_to_explain = 20
np.random.seed(123)  # different seed from background, avoid overlap bias
explain_sample_idx = np.random.choice(test_rows, size=n_patients_to_explain, replace=False)

print(f"\nRunning SHAP for {n_patients_to_explain} test patients...")

class TPCWrapperSingleStatic(torch.nn.Module):
    def __init__(self, model, static_value):
        super().__init__()
        self.model = model
        self.static_value = static_value

    def forward(self, x):
        batch_size = x.shape[0]
        static_repeated = self.static_value.repeat(batch_size, 1)
        return self.model(x, static_repeated)

all_shap_values = []
all_patient_stay_ids = []
start_time = time.time()

for i, patient_row in enumerate(explain_sample_idx):
    patient_static = torch.tensor(static[patient_row:patient_row+1], dtype=torch.float32)
    patient_X = torch.tensor(X[patient_row:patient_row+1], dtype=torch.float32)

    wrapper = TPCWrapperSingleStatic(model, patient_static)
    explainer = shap.GradientExplainer(wrapper, X_background)
    shap_values = explainer.shap_values(patient_X)

    all_shap_values.append(np.array(shap_values)[0])  # drop the leading batch dim of 1
    all_patient_stay_ids.append(stay_id_order["stay_id"].iloc[patient_row])

    elapsed = time.time() - start_time
    avg_per_patient = elapsed / (i + 1)
    print(f"  Patient {i+1}/{n_patients_to_explain} done - "
          f"elapsed: {elapsed:.1f}s, avg: {avg_per_patient:.1f}s/patient")

total_elapsed = time.time() - start_time
print(f"\nTotal time for {n_patients_to_explain} patients: {total_elapsed:.1f}s "
      f"({total_elapsed/n_patients_to_explain:.1f}s/patient average)")


# 4. Stack and save results

all_shap_values = np.stack(all_shap_values)  # shape: (n_patients, 48, 24, 1)
print("\nFinal SHAP values array shape:", all_shap_values.shape)

np.save("tpc_shap_values_sample.npy", all_shap_values)
pd.DataFrame({"stay_id": all_patient_stay_ids}).to_csv("tpc_shap_sample_stay_ids.csv", index=False)

print("\nSaved: tpc_shap_values_sample.npy, tpc_shap_sample_stay_ids.csv")
print("Done.")
print(f"\nBased on this timing, explaining ALL {len(test_rows)} test patients "
      f"would take approximately {total_elapsed/n_patients_to_explain*len(test_rows)/60:.1f} minutes.")