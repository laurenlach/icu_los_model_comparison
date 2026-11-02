import torch
import numpy as np
from captum.attr import IntegratedGradients
from icu_benchmarks.models.dl_models.bat import BAT, RegressionHead

# ---------------------------------------------------------------------------
# 1. Load the trained BAT model (identical to ig_bat_single_patient.py)
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
# 2. Load test data and the NEW healthy baseline (Z-scored space)
# ---------------------------------------------------------------------------
X = np.load("bat_X_scaled.npy")
static = np.load("bat_static_scaled.npy")
mask = np.load("bat_mask.npy")
test_rows = np.load("bat_test_rows.npy")
healthy_baseline = np.load("healthy_baseline_zscored.npy")  # shape (48,)

# ---------------------------------------------------------------------------
# 3. Pick a subsample of test patients (200) - fast enough to run quickly,
#    large enough to draw a real conclusion from
# ---------------------------------------------------------------------------
np.random.seed(42)
n_sample = 200
sample_rows = np.random.choice(test_rows, size=n_sample, replace=False)

ig = IntegratedGradients(model)

# Noise level: 3% of each feature's Z-scored std (i.e. std=1, so noise_std=0.03)
# - small enough to represent measurement noise, not a different patient
NOISE_STD = 0.03

pred_diffs = []
attr_shift_fractions = []

for i, row in enumerate(sample_rows):
    X_patient = np.transpose(X[row:row+1], (0, 2, 1))       # (1, 48, 24)
    mask_patient = np.transpose(mask[row:row+1], (0, 2, 1))
    static_patient = static[row:row+1]
    time_patient = np.tile(np.arange(24, dtype=np.float32), (1, 1))

    X_patient_t = torch.tensor(X_patient, dtype=torch.float32, requires_grad=True)
    static_t = torch.tensor(static_patient, dtype=torch.float32)
    time_t = torch.tensor(time_patient, dtype=torch.float32)
    mask_t = torch.tensor(mask_patient, dtype=torch.float32)

    # Baseline: the healthy reference, broadcast across all 24 timesteps
    baseline = torch.tensor(
        np.tile(healthy_baseline.reshape(1, 48, 1), (1, 1, 24)),
        dtype=torch.float32
    )

    # --- Original prediction and attribution ---
    with torch.no_grad():
        pred_original = model(X_patient_t, static_t, time_t, mask_t).item()

    attr_original = ig.attribute(
        X_patient_t, baselines=baseline,
        additional_forward_args=(static_t, time_t, mask_t), n_steps=50,
    ).detach().numpy()

    # --- Add small noise, only to genuinely-observed values (respecting the mask) ---
    noise = np.random.normal(0, NOISE_STD, size=X_patient.shape)
    X_noisy = X_patient + noise * mask_patient  # only perturb observed values
    X_noisy_t = torch.tensor(X_noisy, dtype=torch.float32, requires_grad=True)

    with torch.no_grad():
        pred_noisy = model(X_noisy_t, static_t, time_t, mask_t).item()

    attr_noisy = ig.attribute(
        X_noisy_t, baselines=baseline,
        additional_forward_args=(static_t, time_t, mask_t), n_steps=50,
    ).detach().numpy()

    # --- Measure the shift ---
    pred_diff = abs(pred_noisy - pred_original)
    pred_diffs.append(pred_diff)

    # Fraction change in total attribution magnitude - how much did the
    # "story" of feature importance shift, relative to its original size
    attr_orig_mag = np.abs(attr_original).sum()
    attr_diff_mag = np.abs(attr_noisy - attr_original).sum()
    attr_shift_fractions.append(attr_diff_mag / (attr_orig_mag + 1e-8))

    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{n_sample} patients...")

pred_diffs = np.array(pred_diffs)
attr_shift_fractions = np.array(attr_shift_fractions)

print("\n=== BAT Automation-Bias / Stability Check ===")
print(f"Mean prediction shift under noise: {pred_diffs.mean():.4f} days "
      f"(median: {np.median(pred_diffs):.4f})")
print(f"Mean attribution shift (fraction of original magnitude): "
      f"{attr_shift_fractions.mean():.4f} (median: {np.median(attr_shift_fractions):.4f})")

np.save("bat_automation_bias_pred_diffs.npy", pred_diffs)
np.save("bat_automation_bias_attr_shifts.npy", attr_shift_fractions)
print("\nSaved: bat_automation_bias_pred_diffs.npy, bat_automation_bias_attr_shifts.npy")