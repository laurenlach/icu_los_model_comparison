import numpy as np

tpc_pred = np.load("tpc_test_predictions.npy")
tpc_true = np.load("tpc_test_targets.npy")
bat_pred = np.load("bat_test_predictions.npy")
bat_true = np.load("bat_test_targets.npy")

for name, pred, true in [("TPC", tpc_pred, tpc_true), ("BAT", bat_pred, bat_true)]:
    error = pred - true
    dangerous = error <= -3.0
    quintiles = np.quantile(true, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(true, quintiles)

    print(f"\n=== {name}: dangerous cases (error <= -3) by LOS quintile ===")
    print(f"Total dangerous: {dangerous.sum()} ({100*dangerous.mean():.2f}%)")
    print("Quintile | n patients | n dangerous | % dangerous within quintile | std of error")
    for b in range(5):
        mask = bins == b
        n_dang = dangerous[mask].sum()
        print(f"  Q{b+1}     | {mask.sum():5d}      | {n_dang:5d}       | "
              f"{100*n_dang/mask.sum():6.2f}%                     | {error[mask].std():.3f}")