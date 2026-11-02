import numpy as np
from scipy.stats import spearmanr

bat_error = np.load("bat_error.npy")           # predicted - true
tpc_error = np.load("tpc_error.npy")
y_test = np.load("tpc_test_targets_fixed.npy")  # true LOS, same test set/order for both models

rho_bat, p_bat = spearmanr(y_test, bat_error)
rho_tpc, p_tpc = spearmanr(y_test, tpc_error)

print(f"BAT: correlation(true LOS, signed error) = {rho_bat:+.4f} (p={p_bat:.3g})")
print(f"TPC: correlation(true LOS, signed error) = {rho_tpc:+.4f} (p={p_tpc:.3g})")
print("\nStrongly negative = error gets more negative (more under-prediction) as true LOS grows")

# Bin by true LOS to see the shape directly
bins = np.quantile(y_test, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
print("\nMean signed error by true-LOS quintile:")
print(f"{'Quintile':10s} {'LOS range':20s} {'BAT mean error':>15s} {'TPC mean error':>15s}")
for i in range(5):
    lo, hi = bins[i], bins[i+1]
    mask = (y_test >= lo) & (y_test <= hi if i == 4 else y_test < hi)
    print(f"{i+1:<10d} {lo:.1f}-{hi:.1f} days{'':6s} {bat_error[mask].mean():>15.3f} {tpc_error[mask].mean():>15.3f}")