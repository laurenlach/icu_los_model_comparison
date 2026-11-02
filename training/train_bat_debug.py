import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from icu_benchmarks.models.dl_models.bat import BAT, RegressionHead

# ---------------------------------------------------------------------------
# 1. Load our prepared data
# ---------------------------------------------------------------------------
X = np.load("bat_X.npy")   # shape: (51835, 24, 48)
y = np.load("bat_y.npy")   # shape: (51835,)

print("X shape:", X.shape, "| y shape:", y.shape)

# For this debug run, use a small subset - just to confirm everything runs
n_debug = 500
X_debug = X[:n_debug]
y_debug = y[:n_debug]

# BAT expects (batch, sensors, timepoints) - our data is (batch, timepoints, sensors)
# so we need to transpose the last two dimensions
X_debug = np.transpose(X_debug, (0, 2, 1))  # -> (n_debug, 48, 24)
print("X_debug shape after transpose:", X_debug.shape)

X_tensor = torch.tensor(X_debug, dtype=torch.float32)
y_tensor = torch.tensor(y_debug, dtype=torch.float32)

# BAT also needs static, time, and sensor_mask inputs - build simple placeholders for now
# static_tensor = torch.zeros((n_debug, 4), dtype=torch.float32)          # 4 static features
# time_tensor = torch.arange(24, dtype=torch.float32).repeat(n_debug, 1)  # hour index 0-23
# mask_tensor = torch.ones((n_debug, 48, 24), dtype=torch.float32)        # all values "observed"
static_full = np.load("bat_static.npy")
mask_full = np.load("bat_mask.npy")

static_debug = static_full[:n_debug]
mask_debug = mask_full[:n_debug]
mask_debug = np.transpose(mask_debug, (0, 2, 1))  # match X's (sensors, timepoints) transpose

static_tensor = torch.tensor(static_debug, dtype=torch.float32)
time_tensor = torch.arange(24, dtype=torch.float32).repeat(n_debug, 1)
mask_tensor = torch.tensor(mask_debug, dtype=torch.float32)

dataset = TensorDataset(X_tensor, static_tensor, time_tensor, mask_tensor, y_tensor)
train_len = int(0.8 * n_debug)
train_set, val_set = random_split(dataset, [train_len, n_debug - train_len])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16)

# ---------------------------------------------------------------------------
# 2. Rebuild the bridged model (same as bridge_bat_checkpoint.py)
# ---------------------------------------------------------------------------
ckpt = torch.load("pretrained_checkpoints/bat_pretrained.ckpt", map_location="cpu")
encoder_weights = {
    k: v for k, v in ckpt["state_dict"].items()
    if k.startswith("model.encoder_class.")
}

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
model.load_state_dict(encoder_weights, strict=False)
print("Pretrained encoder weights loaded.")

# ---------------------------------------------------------------------------
# 3. Tiny training loop - just a couple epochs, to confirm it runs end-to-end
# ---------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

model.train()
for epoch in range(2):
    total_loss = 0
    for data, static, time, mask, target in train_loader:
        optimizer.zero_grad()
        output = model(data, static, time, mask)
        output = output.squeeze()
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - train loss: {total_loss / len(train_loader):.4f}")

print("\nDebug training run completed successfully.")