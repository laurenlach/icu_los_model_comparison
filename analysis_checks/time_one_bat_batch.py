import time
import numpy as np
import torch
from icu_benchmarks.models.dl_models.bat import BAT, RegressionHead

model = BAT(
    input_size=(64, 48, 24), value_embed_size=128, layers=2, heads=1,
    dropout=0.36427, attn_dropout=0.20659, use_mask=False,
    prediction_head=RegressionHead, prediction_head_kwargs={"output_dim": 1},
    static_count=4,
)
model.load_state_dict(torch.load("bat_best_model.pt", map_location="cpu"))
model.eval()
print("Model loaded.")

X = np.load("bat_X_scaled.npy")
static = np.load("bat_static_scaled.npy")
mask = np.load("bat_mask.npy")
test_rows = np.load("bat_test_rows.npy")

X_test_raw = X[test_rows]
static_test = static[test_rows]
mask_test_raw = mask[test_rows]
time_arr = np.arange(24, dtype=np.float32)

batch_size = 256
n_batches_to_time = 5  # just time a handful of batches, not the whole set

start = time.time()
with torch.no_grad():
    for i in range(n_batches_to_time):
        s = i * batch_size
        e = s + batch_size
        xb_raw = X_test_raw[s:e]
        mb_raw = mask_test_raw[s:e]
        sb = static_test[s:e]
        bsz = len(xb_raw)

        xb = torch.tensor(np.transpose(xb_raw, (0, 2, 1)), dtype=torch.float32)
        mb = torch.tensor(np.transpose(mb_raw, (0, 2, 1)), dtype=torch.float32)
        sb_t = torch.tensor(sb, dtype=torch.float32)
        tb = torch.tensor(np.tile(time_arr, (bsz, 1)), dtype=torch.float32)

        out = model(xb, sb_t, tb, mb)
        elapsed = time.time() - start
        print(f"Batch {i+1}/{n_batches_to_time} done, {elapsed:.1f}s elapsed so far")

total = time.time() - start
per_batch = total / n_batches_to_time
n_total_batches = len(test_rows) / batch_size
print(f"\nPer-batch time: {per_batch:.2f}s")
print(f"Estimated time for ONE full pass over test set: {per_batch * n_total_batches / 60:.1f} min")
print(f"Estimated time for full permutation run (48 features x 5 repeats): "
      f"{per_batch * n_total_batches * 48 * 5 / 60:.1f} min")