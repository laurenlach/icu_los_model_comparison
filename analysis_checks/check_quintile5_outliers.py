import numpy as np

bat_error = np.load("bat_error.npy")
tpc_error = np.load("tpc_error.npy")
y_test = np.load("tpc_test_targets_fixed.npy")

bins = np.quantile(y_test, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
mask = (y_test >= bins[4]) & (y_test <= bins[5])
n = mask.sum()
los_q5 = y_test[mask]
bat_q5 = bat_error[mask]
tpc_q5 = tpc_error[mask]

print(f"Quintile 5: n={n} patients, LOS range {los_q5.min():.1f}-{los_q5.max():.1f} days\n")

print("Mean vs median error (big gap = outliers dragging the mean):")
print(f"BAT: mean={bat_q5.mean():.3f}, median={np.median(bat_q5):.3f}")
print(f"TPC: mean={tpc_q5.mean():.3f}, median={np.median(tpc_q5):.3f}\n")

# Sub-bin quintile 5 itself to see where the error actually lives
sub_edges = np.quantile(los_q5, [0, 0.25, 0.5, 0.75, 1.0])
print("Sub-bins within quintile 5:")
print(f"{'Sub-bin':10s} {'LOS range':18s} {'n':>6s} {'BAT mean':>10s} {'BAT median':>11s} {'TPC mean':>10s} {'TPC median':>11s}")
for i in range(4):
    lo, hi = sub_edges[i], sub_edges[i+1]
    m = (los_q5 >= lo) & (los_q5 <= hi if i == 3 else los_q5 < hi)
    print(f"{i+1:<10d} {lo:.1f}-{hi:.1f} days{'':4s} {m.sum():>6d} "
          f"{bat_q5[m].mean():>10.3f} {np.median(bat_q5[m]):>11.3f} "
          f"{tpc_q5[m].mean():>10.3f} {np.median(tpc_q5[m]):>11.3f}")

# How many patients actually drive the extreme tail
print(f"\nPatients with LOS > 10 days (the proposal's own cohort upper bound): "
      f"{(los_q5 > 10).sum()} of {n} in this quintile "
      f"({100*(los_q5 > 10).sum()/n:.1f}%)")
print(f"Patients with LOS > 10 days, BAT mean error: {bat_q5[los_q5 > 10].mean():.3f}")
print(f"Patients with LOS > 10 days, TPC mean error: {tpc_q5[los_q5 > 10].mean():.3f}")