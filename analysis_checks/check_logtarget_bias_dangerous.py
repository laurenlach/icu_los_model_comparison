import numpy as np

def report(name, pred, true):
    error = pred - true
    dangerous = error <= -3.0
    print(f"\n=== {name} ===")
    print(f"MAE: {np.abs(error).mean():.4f} | Mean error (bias): {error.mean():+.4f}")
    print(f"Dangerous cases (error <= -3): {dangerous.sum()} ({100*dangerous.mean():.2f}%)")

    quintiles = np.quantile(true, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(true, quintiles)
    print("Quintile | n | mean true LOS | mean error | % dangerous")
    for b in range(5):
        mask = bins == b
        print(f"  Q{b+1}     | {mask.sum():5d} | {true[mask].mean():6.2f} | {error[mask].mean():+.3f}    | {100*dangerous[mask].mean():.2f}%")

tpc_pred = np.load("tpc_logtarget_test_predictions_days.npy")
tpc_true = np.load("tpc_logtarget_test_targets_days.npy")
report("TPC (log-target)", tpc_pred, tpc_true)

bat_pred = np.load("bat_logtarget_test_predictions_days.npy")
bat_true = np.load("bat_logtarget_test_targets_days.npy")
report("BAT (log-target)", bat_pred, bat_true)