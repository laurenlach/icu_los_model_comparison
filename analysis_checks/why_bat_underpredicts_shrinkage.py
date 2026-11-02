import numpy as np
from scipy.stats import linregress, spearmanr

tpc_pred = np.load("tpc_test_predictions.npy")
tpc_true = np.load("tpc_test_targets.npy")
bat_pred = np.load("bat_test_predictions.npy")   # adjust path if it's in BAT_colab locally
bat_true = np.load("bat_test_targets.npy")

for name, pred, true in [("TPC", tpc_pred, tpc_true), ("BAT", bat_pred, bat_true)]:
    slope, intercept, r, p, se = linregress(true, pred)
    error = pred - true
    rho_err_los = spearmanr(true, error)[0]
    print(f"\n=== {name} ===")
    print(f"Predicted = {slope:.3f} * True + {intercept:.3f}  (slope=1 would mean no shrinkage)")
    print(f"R^2 = {r**2:.3f}")
    print(f"Spearman(true LOS, error) = {rho_err_los:+.3f}  (strongly negative = shrinkage: error worsens as true LOS grows)")

    # Break into LOS quintiles to see where the under-prediction concentrates
    quintiles = np.quantile(true, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(true, quintiles)
    print("LOS quintile | n | mean true LOS | mean error (pred - true)")
    for b in range(5):
        mask = bins == b
        print(f"  Q{b+1}        | {mask.sum():5d} | {true[mask].mean():6.2f} | {error[mask].mean():+.3f}")