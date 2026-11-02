import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

tpc_pred = np.load("tpc_test_predictions_fixed.npy").ravel().astype(float)
tpc_targ = np.load("tpc_test_targets_fixed.npy").ravel().astype(float)
bat_pred = np.load("bat_test_predictions.npy").ravel().astype(float)
bat_targ = np.load("bat_test_targets.npy").ravel().astype(float)

# Both models were evaluated on the same stays in the same order, so the
# errors are paired. Assert it rather than assume it, since a silent
# mismatch here would invalidate the test without raising an error.
assert np.allclose(tpc_targ, bat_targ), "Targets differ - splits are not aligned."

tpc_abs = np.abs(tpc_pred - tpc_targ)
bat_abs = np.abs(bat_pred - bat_targ)

stat, p = wilcoxon(tpc_abs, bat_abs)

print(f"n stays                : {len(tpc_abs)}")
print(f"TPC MAE                : {tpc_abs.mean():.4f}")
print(f"BAT MAE                : {bat_abs.mean():.4f}")
print(f"Advantage to BAT       : {tpc_abs.mean() - bat_abs.mean():.4f} days")
print(f"Wilcoxon statistic     : {stat:.1f}")
print(f"Wilcoxon p             : {p:.3g}")
print(f"BAT lower abs error on : {(bat_abs < tpc_abs).sum()} of {len(tpc_abs)} stays")

pd.DataFrame([{
    "n_stays": len(tpc_abs),
    "TPC_MAE": tpc_abs.mean(),
    "BAT_MAE": bat_abs.mean(),
    "advantage_to_BAT_days": tpc_abs.mean() - bat_abs.mean(),
    "wilcoxon_statistic": stat,
    "p_value": p,
    "BAT_lower_count": int((bat_abs < tpc_abs).sum()),
}]).to_csv("significance_test_overall.csv", index=False)
print("\nSaved: significance_test_overall.csv")