import torch
import numpy as np
import shap
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.dl_models.tpc import TPC

feature_names = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun",
    "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2",
    "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
    "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt",
    "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine", "wbc"
]
SOFA_IDX = [feature_names.index(f) for f in ["map", "bun", "crea", "bili", "plt", "fio2", "o2sat"]]

model = TPC(
    input_size=(64, 48, 24), static_size=4, n_layers=3, kernel_size=3,
    temp_kernels=4, point_size=8, dropout=0.1, run_mode=RunMode.regression,
)
model.load_state_dict(torch.load("tpc_best_model_alignmentpenalty.pt", map_location="cpu"))
model.eval()

X = np.load("tpc_X_scaled.npy")
static = np.load("tpc_static_scaled.npy")
train_rows = np.load("tpc_train_rows.npy")
test_rows = np.load("tpc_test_rows.npy")


class TPCWrapperSingleStatic(torch.nn.Module):
    def __init__(self, model, static_vec):
        super().__init__()
        self.model = model
        self.static_vec = static_vec
    def forward(self, x):
        static_b = self.static_vec.repeat(x.shape[0], 1)
        return self.model(x, static_b)

rng = np.random.default_rng(0)
background_idx = rng.choice(train_rows, size=50, replace=False)
background = torch.tensor(X[background_idx], dtype=torch.float32)


sample_idx = rng.choice(test_rows, size=1000, replace=False)

all_shap = []
for i, idx in enumerate(sample_idx):
    static_vec = torch.tensor(static[idx:idx+1], dtype=torch.float32)
    wrapper = TPCWrapperSingleStatic(model, static_vec)
    explainer = shap.GradientExplainer(wrapper, background)
    x_patient = torch.tensor(X[idx:idx+1], dtype=torch.float32)
    shap_vals = explainer.shap_values(x_patient)
    all_shap.append(np.array(shap_vals).squeeze())
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/1000...")

all_shap = np.stack(all_shap)   # (1000, 48, 24)
attr = np.abs(all_shap).mean(axis=2)   # (1000, 48)

total_mass = attr.sum(axis=1)
sofa_mass = attr[:, SOFA_IDX].sum(axis=1)
sofa_fraction = sofa_mass / total_mass

print(f"\n=== Real SHAP SOFA-attribution fraction (alignment-penalty model, n=1000) ===")
print(f"Mean: {sofa_fraction.mean():.4f}")
print(f"Median: {np.median(sofa_fraction):.4f}")
print(f"(compare: original fixed-TPC SHAP SOFA fraction = 0.1900)")
print(f"(compare: training-time Grad x Input proxy on this model = 0.4223)")

np.save("tpc_alignmentpenalty_shap_sample.npy", all_shap)
np.save("tpc_alignmentpenalty_shap_sample_idx.npy", sample_idx)
print("\nSaved: tpc_alignmentpenalty_shap_sample.npy, tpc_alignmentpenalty_shap_sample_idx.npy")