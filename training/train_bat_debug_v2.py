import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from icu_benchmarks.models.dl_models.bat import BAT, RegressionHead

# ---------------------------------------------------------------------------
# 1. Load the FINAL, correct data: scaled, with real static/mask, real split
# ---------------------------------------------------------------------------
X = np.load("bat_X_scaled.npy")        # (51835, 24, 48) - Z-scored
static = np.load("bat_static_scaled.npy")  # (51835, 4) - Z-scored
mask = np.load("bat_mask.npy")         # (51835, 24, 48) - real observed indicator
y = np.load("bat_y.npy")               # (51835,)

train_rows = np.load("bat_train_rows.npy")
val_rows = np.load("bat_val_rows.npy")

print("Full data shapes:", X.shape, static.shape, mask.shape, y.shape)
print("Train patients:", len(train_rows), "| Val patients:", len(val_rows))

# ---------------------------------------------------------------------------
# 2. For this debug run: use a manageable slice of the REAL train/val split
#    (not the full 31k/10k - just enough to confirm everything works and
#    the loss trends sensibly, in a reasonable amount of time on CPU)
# ---------------------------------------------------------------------------
n_train_debug = 2000
n_val_debug = 500

train_idx = train_rows[:n_train_debug]
val_idx = val_rows[:n_val_debug]

def build_tensors(indices):
    X_sub = np.transpose(X[indices], (0, 2, 1))       # -> (n, 48, 24) for BAT
    mask_sub = np.transpose(mask[indices], (0, 2, 1))  # -> (n, 48, 24)
    static_sub = static[indices]
    y_sub = y[indices]
    time_sub = np.tile(np.arange(24, dtype=np.float32), (len(indices), 1))

    return (
        torch.tensor(X_sub, dtype=torch.float32),
        torch.tensor(static_sub, dtype=torch.float32),
        torch.tensor(time_sub, dtype=torch.float32),
        torch.tensor(mask_sub, dtype=torch.float32),
        torch.tensor(y_sub, dtype=torch.float32),
    )

train_tensors = build_tensors(train_idx)
val_tensors = build_tensors(val_idx)

train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(*val_tensors), batch_size=32)

# ---------------------------------------------------------------------------
# 3. Rebuild the bridged model (pretrained encoder + fresh regression head)
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
result = model.load_state_dict(encoder_weights, strict=False)
print("\nPretrained encoder weights loaded.")
print("Missing keys (expect only the new head):", result.missing_keys)
print("Unexpected keys (expect empty):", result.unexpected_keys)

# ---------------------------------------------------------------------------
# 4. Real training loop, with validation loss tracked too
# ---------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

n_epochs = 5

for epoch in range(n_epochs):
    model.train()
    train_loss_total = 0
    for data, static_b, time_b, mask_b, target in train_loader:
        optimizer.zero_grad()
        output = model(data, static_b, time_b, mask_b).squeeze()
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss_total += loss.item()

    model.eval()
    val_loss_total = 0
    with torch.no_grad():
        for data, static_b, time_b, mask_b, target in val_loader:
            output = model(data, static_b, time_b, mask_b).squeeze()
            val_loss_total += loss_fn(output, target).item()

    print(
        f"Epoch {epoch+1} - "
        f"train loss: {train_loss_total / len(train_loader):.4f} | "
        f"val loss: {val_loss_total / len(val_loader):.4f}"
    )

print("\nFinal debug training run (real scaled data, real split) completed successfully.")