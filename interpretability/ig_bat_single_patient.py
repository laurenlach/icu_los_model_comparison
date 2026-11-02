import torch
import numpy as np
from captum.attr import IntegratedGradients
from icu_benchmarks.models.dl_models.bat import BAT, RegressionHead

# ---------------------------------------------------------------------------
# 1. Load the trained BAT model (bridged pretrained encoder + fine-tuned head)
# ---------------------------------------------------------------------------
model = BAT(
    input_size=(64, 48, 24),
    value_embed_size=128,
    layers=2,
    heads=1,
    dropout=0.36427,
    attn_dropout=0.20659,
    use_mask=False,
    prediction_head=RegressionHead,
    prediction_head_kwargs={"output_dim": 1},
    static_count=4,
)
model.load_state_dict(torch.load("bat_best_model.pt", map_location="cpu"))
model.eval()
print("Trained BAT model loaded successfully.")

# ---------------------------------------------------------------------------
# 2. Load one patient's data - same shapes BAT expects (recall the
#    transpose step used for BAT, unlike TPC)
# ---------------------------------------------------------------------------
X = np.load("bat_X_scaled.npy")
static = np.load("bat_static_scaled.npy")
mask = np.load("bat_mask.npy")
test_rows = np.load("bat_test_rows.npy")

patient_row = test_rows[0]

X_patient = np.transpose(X[patient_row:patient_row+1], (0, 2, 1))       # -> (1, 48, 24)
mask_patient = np.transpose(mask[patient_row:patient_row+1], (0, 2, 1)) # -> (1, 48, 24)
static_patient = static[patient_row:patient_row+1]
time_patient = np.tile(np.arange(24, dtype=np.float32), (1, 1))

X_patient_t = torch.tensor(X_patient, dtype=torch.float32, requires_grad=True)
static_t = torch.tensor(static_patient, dtype=torch.float32)
time_t = torch.tensor(time_patient, dtype=torch.float32)
mask_t = torch.tensor(mask_patient, dtype=torch.float32)

print("X_patient shape:", X_patient_t.shape)

# ---------------------------------------------------------------------------
# 3. Confirm the model runs correctly on this patient before adding
#    Integrated Gradients on top
# ---------------------------------------------------------------------------
with torch.no_grad():
    prediction = model(X_patient_t, static_t, time_t, mask_t)
print("Direct model prediction:", prediction.item())

# ---------------------------------------------------------------------------
# 4. Set up Integrated Gradients. additional_forward_args passes the
#    fixed inputs (static, time, mask) through to the model's forward()
#    unchanged, while X_patient_t is the one input being explained.
#    Baseline = all zeros (a common, simple choice - "no signal" reference).
# ---------------------------------------------------------------------------
ig = IntegratedGradients(model)

baseline = torch.zeros_like(X_patient_t)

print("\nComputing Integrated Gradients attributions...")
attributions = ig.attribute(
    X_patient_t,
    baselines=baseline,
    additional_forward_args=(static_t, time_t, mask_t),
    n_steps=50,
)

print("\nAttributions computed successfully.")
print("Attributions shape:", attributions.shape)
print("(Expect (1, 48, 24) - one value per sensor per timestep)")

# ---------------------------------------------------------------------------
# 5. Sanity check: Integrated Gradients has a "completeness" property too -
#    sum of attributions should roughly equal (prediction - baseline prediction)
# ---------------------------------------------------------------------------
with torch.no_grad():
    baseline_prediction = model(baseline, static_t, time_t, mask_t)

attr_sum = attributions.sum().item()
expected_diff = (prediction - baseline_prediction).item()

print(f"\nSanity check:")
print(f"Prediction: {prediction.item():.4f}")
print(f"Baseline (all-zero input) prediction: {baseline_prediction.item():.4f}")
print(f"Expected sum (prediction - baseline): {expected_diff:.4f}")
print(f"Actual sum of attributions: {attr_sum:.4f}")