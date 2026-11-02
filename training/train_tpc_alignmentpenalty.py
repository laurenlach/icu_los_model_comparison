import torch
import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader
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
sofa_mask = torch.zeros(48)
sofa_mask[SOFA_IDX] = 1.0

X = np.load("tpc_X_scaled.npy")
static = np.load("tpc_static_scaled.npy")
y = np.load("tpc_y.npy")

train_rows = np.load("tpc_train_rows.npy")
val_rows = np.load("tpc_val_rows.npy")
test_rows = np.load("tpc_test_rows.npy")

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

model = TPC(
    input_size=(64, 48, 24), static_size=4, n_layers=3, kernel_size=3,
    temp_kernels=4, point_size=8, dropout=0.1, run_mode=RunMode.regression,
)
print(f"TPC model built (alignment-penalty). Trainable parameters: "
      f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
mse_fn = torch.nn.MSELoss()
LAMBDA = 0.5   # weight on the alignment penalty relative to MSE -- a chosen, adjustable judgment call

def alignment_penalty(x_batch, static_b, pred):
    # Gradient x Input attribution, computed w.r.t. x_batch, with create_graph=True
    # so this penalty's own gradient can backprop into the model's weights (double backward).
    grads = torch.autograd.grad(pred.sum(), x_batch, create_graph=True)[0]   # (B, 48, 24)
    attr = (grads * x_batch).abs().sum(dim=2)   # (B, 48) -- sum over time
    total = attr.sum(dim=1) + 1e-8               # (B,)
    sofa_attr = (attr * sofa_mask.to(attr.device)).sum(dim=1)   # (B,)
    sofa_fraction = sofa_attr / total
    return (1.0 - sofa_fraction).mean(), sofa_fraction.mean().item()

max_epochs = 200
patience = 10
best_val_loss = float("inf")
epochs_without_improvement = 0
log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)

log(f"Starting TPC alignment-penalty training - max {max_epochs} epochs, patience {patience}, lambda={LAMBDA}")
start_time = time.time()

for epoch in range(max_epochs):
    model.train()
    train_loss_total, sofa_frac_total, n_batches = 0, 0, 0
    for data, static_b, target in train_loader:
        data.requires_grad_(True)
        optimizer.zero_grad()
        output = model(data, static_b).squeeze()
        mse = mse_fn(output, target)
        penalty, sofa_frac = alignment_penalty(data, static_b, output)
        loss = mse + LAMBDA * penalty
        loss.backward()
        optimizer.step()
        train_loss_total += mse.item()   # log plain MSE for comparability, not the combined loss
        sofa_frac_total += sofa_frac
        n_batches += 1
    train_loss_avg = train_loss_total / n_batches
    sofa_frac_avg = sofa_frac_total / n_batches

    # Val loss: plain MSE, no gradients needed, no penalty (same convention as earlier experiments)
    model.eval()
    val_loss_total = 0
    with torch.no_grad():
        for data, static_b, target in val_loader:
            output = model(data, static_b).squeeze()
            val_loss_total += mse_fn(output, target).item()
    val_loss_avg = val_loss_total / len(val_loader)

    elapsed = time.time() - start_time
    log(f"Epoch {epoch+1} - train MSE: {train_loss_avg:.4f} | train SOFA-attr fraction: {sofa_frac_avg:.4f} "
        f"| val MSE: {val_loss_avg:.4f} | elapsed: {elapsed/60:.1f} min")

    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "tpc_best_model_alignmentpenalty.pt")
        log(f"  -> New best val loss ({val_loss_avg:.4f}), model saved.")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            log(f"\nEarly stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs).")
            break

log(f"\nTraining finished. Best val loss (plain MSE): {best_val_loss:.4f}")

model.load_state_dict(torch.load("tpc_best_model_alignmentpenalty.pt"))
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
bias = torch.mean(all_preds - all_targets).item()

log(f"\n=== FINAL TEST SET RESULTS (TPC, alignment-penalty) ===")
log(f"MAE: {mae:.4f} days")
log(f"RMSE: {rmse:.4f} days")
log(f"Mean error (bias): {bias:+.4f} days")
log(f"Final train SOFA-attribution fraction (Grad x Input proxy): {sofa_frac_avg:.4f}")
log(f"(compare: original 48-sensor TPC -- MAE 2.903, bias +0.04)")

np.save("tpc_alignmentpenalty_test_predictions.npy", all_preds.numpy())
np.save("tpc_alignmentpenalty_test_targets.npy", all_targets.numpy())
with open("tpc_alignmentpenalty_training_log.txt", "w") as f:
    f.write("\n".join(log_lines))

print("\nSaved: tpc_best_model_alignmentpenalty.pt, tpc_alignmentpenalty_test_predictions.npy, "
      "tpc_alignmentpenalty_test_targets.npy, tpc_alignmentpenalty_training_log.txt")
print("Original tpc_best_model_fixed.pt was NOT touched.")