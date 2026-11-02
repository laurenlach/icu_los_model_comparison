import torch
import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.dl_models.tpc import TPC


X = np.load("tpc_X_scaled.npy")
static = np.load("tpc_static_scaled.npy")
y = np.load("tpc_y.npy")

train_rows = np.load("tpc_train_rows.npy")
val_rows = np.load("tpc_val_rows.npy")
test_rows = np.load("tpc_test_rows.npy")

print(f"Full dataset: {X.shape[0]} patients")
print(f"Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)}")

def build_tensors(indices):
    return (
        torch.tensor(X[indices], dtype=torch.float32),
        torch.tensor(static[indices], dtype=torch.float32),
        torch.tensor(y[indices], dtype=torch.float32),
    )

train_tensors = build_tensors(train_rows)
val_tensors = build_tensors(val_rows)
test_tensors = build_tensors(test_rows)

batch_size = 64
train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(*val_tensors), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(*test_tensors), batch_size=batch_size)


DROPOUT_FRACTION = 0.15  # fraction of the 48 sensors dropped per patient per batch

def feature_dropout(x_batch, frac=DROPOUT_FRACTION):
    B, F, T = x_batch.shape
    mask = (torch.rand(B, F) >= frac).float().unsqueeze(-1)  # (B, F, 1)
    return x_batch * mask


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
print(f"\nTPC model built (feature-dropout side experiment). Trainable parameters: "
      f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

max_epochs = 200
patience = 10
best_val_loss = float("inf")
epochs_without_improvement = 0

log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)

log(f"Starting TPC feature-dropout training (frac={DROPOUT_FRACTION}) - max {max_epochs} epochs, patience {patience}")
start_time = time.time()

for epoch in range(max_epochs):
    model.train()
    train_loss_total = 0
    for data, static_b, target in train_loader:
        data = feature_dropout(data)          # <-- the only change from train_tpc_fixed.py
        optimizer.zero_grad()
        output = model(data, static_b).squeeze()
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss_total += loss.item()
    train_loss_avg = train_loss_total / len(train_loader)

    model.eval()
    val_loss_total = 0
    with torch.no_grad():
        for data, static_b, target in val_loader:
            output = model(data, static_b).squeeze()
            val_loss_total += loss_fn(output, target).item()
    val_loss_avg = val_loss_total / len(val_loader)

    elapsed = time.time() - start_time
    log(f"Epoch {epoch+1} - train loss: {train_loss_avg:.4f} | val loss: {val_loss_avg:.4f} | elapsed: {elapsed/60:.1f} min")

    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "tpc_best_model_featuredropout.pt")
        log(f"  -> New best val loss ({val_loss_avg:.4f}), model saved.")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            log(f"\nEarly stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs).")
            break

log(f"\nTraining finished. Best val loss: {best_val_loss:.4f}")


model.load_state_dict(torch.load("tpc_best_model_featuredropout.pt"))
model.eval()

all_preds, all_targets = [], []
with torch.no_grad():
    for data, static_b, target in test_loader:
        output = model(data, static_b).squeeze()
        all_preds.append(output)
        all_targets.append(target)

all_preds = torch.cat(all_preds)
all_targets = torch.cat(all_targets)

mae = torch.mean(torch.abs(all_preds - all_targets)).item()
rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2)).item()

log(f"\n=== FINAL TEST SET RESULTS (TPC, feature-dropout) ===")
log(f"MAE: {mae:.4f} days")
log(f"RMSE: {rmse:.4f} days")

np.save("tpc_featuredropout_test_predictions.npy", all_preds.numpy())
np.save("tpc_featuredropout_test_targets.npy", all_targets.numpy())

with open("tpc_featuredropout_training_log.txt", "w") as f:
    f.write("\n".join(log_lines))

print("\nSaved: tpc_best_model_featuredropout.pt, tpc_featuredropout_test_predictions.npy, "
      "tpc_featuredropout_test_targets.npy, tpc_featuredropout_training_log.txt")
print("Original tpc_best_model_fixed.pt was NOT touched.")