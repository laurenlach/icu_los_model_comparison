import torch
import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.dl_models.tpc import TPC

X = np.load("tpc_X_scaled.npy")
static = np.load("tpc_static_scaled.npy")
y = np.load("tpc_y.npy")   # raw LOS in days, NOT log-transformed -- lesson from the log-target failure

train_rows = np.load("tpc_train_rows.npy")
val_rows = np.load("tpc_val_rows.npy")
test_rows = np.load("tpc_test_rows.npy")

print(f"Full dataset: {X.shape[0]} patients")
print(f"Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)}")


# Quintile cutoffs computed ONLY from training data

y_train_raw = y[train_rows]
cutoffs = np.quantile(y_train_raw, [0.2, 0.4, 0.6, 0.8])
print(f"\nQuintile cutoffs (from train set): {cutoffs}")

WEIGHTS_BY_QUINTILE = np.array([1.0, 1.0, 1.0, 2.0, 4.0])  # Q1..Q5 -- a chosen, adjustable judgment call

def quintile_weights(y_vals):
    bins = np.digitize(y_vals, cutoffs)   # 0..4
    return WEIGHTS_BY_QUINTILE[bins]

def build_tensors(indices):
    yv = y[indices]
    w = quintile_weights(yv)
    return (
        torch.tensor(X[indices], dtype=torch.float32),
        torch.tensor(static[indices], dtype=torch.float32),
        torch.tensor(yv, dtype=torch.float32),
        torch.tensor(w, dtype=torch.float32),
    )

train_tensors = build_tensors(train_rows)
val_tensors = build_tensors(val_rows)      # weights computed but NOT used for val loss/early stopping (see below)
test_tensors = build_tensors(test_rows)

batch_size = 64
train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(*val_tensors), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(*test_tensors), batch_size=batch_size)

model = TPC(
    input_size=(64, 48, 24), static_size=4, n_layers=3, kernel_size=3,
    temp_kernels=4, point_size=8, dropout=0.1, run_mode=RunMode.regression,
)
print(f"\nTPC model built (quintile-weighted). Trainable parameters: "
      f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def weighted_mse(pred, target, weight):
    return (weight * (pred - target) ** 2).mean()

max_epochs = 200
patience = 10
best_val_loss = float("inf")
epochs_without_improvement = 0
log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)

log(f"Starting TPC quintile-weighted training - max {max_epochs} epochs, patience {patience}")
log(f"Weights by quintile: {WEIGHTS_BY_QUINTILE.tolist()}")
start_time = time.time()

for epoch in range(max_epochs):
    model.train()
    train_loss_total = 0
    for data, static_b, target, weight in train_loader:
        optimizer.zero_grad()
        output = model(data, static_b).squeeze()
        loss = weighted_mse(output, target, weight)
        loss.backward()
        optimizer.step()
        train_loss_total += loss.item()
    train_loss_avg = train_loss_total / len(train_loader)

    # Val loss is UNWEIGHTED plain MSE -- early stopping should reflect true
    # generalization, not the artificially reweighted training objective.
    model.eval()
    val_loss_total = 0
    with torch.no_grad():
        for data, static_b, target, weight in val_loader:
            output = model(data, static_b).squeeze()
            val_loss_total += torch.nn.functional.mse_loss(output, target).item()
    val_loss_avg = val_loss_total / len(val_loader)

    elapsed = time.time() - start_time
    log(f"Epoch {epoch+1} - train loss (weighted): {train_loss_avg:.4f} | val loss (plain MSE): {val_loss_avg:.4f} | elapsed: {elapsed/60:.1f} min")

    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "tpc_best_model_quintileweighted.pt")
        log(f"  -> New best val loss ({val_loss_avg:.4f}), model saved.")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            log(f"\nEarly stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs).")
            break

log(f"\nTraining finished. Best val loss (plain MSE): {best_val_loss:.4f}")

model.load_state_dict(torch.load("tpc_best_model_quintileweighted.pt"))
model.eval()

all_preds, all_targets = [], []
with torch.no_grad():
    for data, static_b, target, weight in test_loader:
        output = model(data, static_b).squeeze()
        all_preds.append(output)
        all_targets.append(target)

all_preds = torch.cat(all_preds)
all_targets = torch.cat(all_targets)

mae = torch.mean(torch.abs(all_preds - all_targets)).item()
rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2)).item()
bias = torch.mean(all_preds - all_targets).item()

log(f"\n=== FINAL TEST SET RESULTS (TPC, quintile-weighted) ===")
log(f"MAE: {mae:.4f} days")
log(f"RMSE: {rmse:.4f} days")
log(f"Mean error (bias): {bias:+.4f} days")

np.save("tpc_quintileweighted_test_predictions.npy", all_preds.numpy())
np.save("tpc_quintileweighted_test_targets.npy", all_targets.numpy())

with open("tpc_quintileweighted_training_log.txt", "w") as f:
    f.write("\n".join(log_lines))

print("\nSaved: tpc_best_model_quintileweighted.pt, tpc_quintileweighted_test_predictions.npy, "
      "tpc_quintileweighted_test_targets.npy, tpc_quintileweighted_training_log.txt")
print("Original tpc_best_model_fixed.pt was NOT touched.")