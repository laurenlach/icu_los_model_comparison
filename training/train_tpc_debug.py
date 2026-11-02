import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.dl_models.tpc import TPC

# ---------------------------------------------------------------------------
# 1. Load TPC's final, scaled, properly-split data
# ---------------------------------------------------------------------------
X = np.load("tpc_X_scaled.npy")        # (51835, 48, 24) - already channels-first
static = np.load("tpc_static_scaled.npy")  # (51835, 4)
y = np.load("tpc_y.npy")               # (51835,)

train_rows = np.load("tpc_train_rows.npy")
val_rows = np.load("tpc_val_rows.npy")

print("Full data shapes:", X.shape, static.shape, y.shape)
print("Train patients:", len(train_rows), "| Val patients:", len(val_rows))

# ---------------------------------------------------------------------------
# 2. Same debug-scale slice as BAT used, for a fair, comparable first test
# ---------------------------------------------------------------------------
n_train_debug = 2000
n_val_debug = 500

train_idx = train_rows[:n_train_debug]
val_idx = val_rows[:n_val_debug]

def build_tensors(indices):
    X_sub = X[indices]           # already (n, 48, 24), no transpose needed
    static_sub = static[indices]
    y_sub = y[indices]
    return (
        torch.tensor(X_sub, dtype=torch.float32),
        torch.tensor(static_sub, dtype=torch.float32),
        torch.tensor(y_sub, dtype=torch.float32),
    )

train_tensors = build_tensors(train_idx)
val_tensors = build_tensors(val_idx)

train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(*val_tensors), batch_size=32)

# ---------------------------------------------------------------------------
# 3. Build a fresh TPC model (trained from scratch - no pretrained checkpoint,
#    unlike BAT, since TPC has no foundation-model equivalent)
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
print("\nTPC model built. Trainable parameters:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))

# ---------------------------------------------------------------------------
# 4. Real training loop, with validation loss tracked - TPC's first ever
#    run with real data and an actual backward pass.
# ---------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

n_epochs = 5

for epoch in range(n_epochs):
    model.train()
    train_loss_total = 0
    for data, static_b, target in train_loader:
        optimizer.zero_grad()
        output = model(data, static_b).squeeze()
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss_total += loss.item()

    model.eval()
    val_loss_total = 0
    with torch.no_grad():
        for data, static_b, target in val_loader:
            output = model(data, static_b).squeeze()
            val_loss_total += loss_fn(output, target).item()

    print(
        f"Epoch {epoch+1} - "
        f"train loss: {train_loss_total / len(train_loader):.4f} | "
        f"val loss: {val_loss_total / len(val_loader):.4f}"
    )

print("\nTPC debug training run (real data, first-ever backward pass) completed successfully.")