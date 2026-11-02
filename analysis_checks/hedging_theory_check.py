import numpy as np
import pandas as pd
from scipy.stats import spearmanr

bat_error = np.load("bat_error.npy")  # predicted - true
bat_y = np.load("bat_y.npy")
bat_test_rows = np.load("bat_test_rows.npy")
true_los = bat_y[bat_test_rows]

# ---------------------------------------------------------------------------
# 1. Direct correlation: does error get more negative as true LOS increases?
#    A negative correlation here means "the longer the real stay, the more
#    BAT under-predicts" - exactly what the hedging theory predicts.
# ---------------------------------------------------------------------------
rho, p = spearmanr(true_los, bat_error)
print(f"Correlation between true LOS and error: rho={rho:.4f} (p={p:.4g})")
print("(Negative = under-prediction worsens as true LOS grows, supporting hedging)")

# ---------------------------------------------------------------------------
# 2. Bucket patients by true LOS, show mean error per bucket - lets us see
#    the actual SHAPE, not just one correlation number
# ---------------------------------------------------------------------------
buckets = [(0, 3), (3, 5), (5, 7), (7, 10), (10, 15), (15, 20), (20, 999)]
print("\n=== Mean error by true-LOS bucket ===")
print(f"{'True LOS range':<18}{'n':<8}{'Mean error (days)':<20}{'Mean |error|':<15}")
for low, high in buckets:
    mask = (true_los >= low) & (true_los < high)
    n = mask.sum()
    if n == 0:
        continue
    mean_err = bat_error[mask].mean()
    mean_abs_err = np.abs(bat_error[mask]).mean()
    label = f"{low}-{high}" if high < 999 else f"{low}+"
    print(f"{label:<18}{n:<8}{mean_err:<20.3f}{mean_abs_err:<15.3f}")

# ---------------------------------------------------------------------------
# 3. Save for reference
# ---------------------------------------------------------------------------
df = pd.DataFrame({"true_los": true_los, "error": bat_error})
df.to_csv("hedging_theory_check.csv", index=False)
print("\nSaved: hedging_theory_check.csv")