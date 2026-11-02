import numpy as np

# ---------------------------------------------------------------------------
# 1. Load the per-patient errors and targets already saved from Phase D
# ---------------------------------------------------------------------------
tpc_error = np.load("tpc_error.npy")
bat_error = np.load("bat_error.npy")

tpc_targets = np.load("tpc_test_targets_200epoch.npy")
bat_targets = np.load("bat_test_targets.npy")

n_total = len(tpc_error)
print(f"Test set size: {n_total}\n")

# ---------------------------------------------------------------------------
# 2. Test several thresholds, not just -3.0, to check the finding is robust
#    and not an artifact of one arbitrary cutoff choice.
# ---------------------------------------------------------------------------
thresholds = [-1.0, -2.0, -3.0, -4.0, -5.0]

results = {"threshold": [], "tpc_n_dangerous": [], "tpc_pct_dangerous": [],
           "bat_n_dangerous": [], "bat_pct_dangerous": []}

print(f"{'Threshold':>10} | {'TPC n (%)':>15} | {'BAT n (%)':>15}")
print("-" * 50)

for t in thresholds:
    tpc_mask = tpc_error <= t
    bat_mask = bat_error <= t

    tpc_n = tpc_mask.sum()
    bat_n = bat_mask.sum()
    tpc_pct = 100 * tpc_n / n_total
    bat_pct = 100 * bat_n / n_total

    print(f"{t:>10.1f} | {tpc_n:>6} ({tpc_pct:>5.2f}%) | {bat_n:>6} ({bat_pct:>5.2f}%)")

    results["threshold"].append(t)
    results["tpc_n_dangerous"].append(tpc_n)
    results["tpc_pct_dangerous"].append(tpc_pct)
    results["bat_n_dangerous"].append(bat_n)
    results["bat_pct_dangerous"].append(bat_pct)

# ---------------------------------------------------------------------------
# 3. Save a dangerous-case mask for EACH threshold, for both models,
#    with explicit, clearly-named files - so any of these can be
#    referenced directly later without re-running this script.
# ---------------------------------------------------------------------------
for t in thresholds:
    t_label = str(abs(t)).replace(".", "_")  # e.g. -3.0 -> "3_0"

    tpc_mask = tpc_error <= t
    bat_mask = bat_error <= t

    np.save(f"tpc_dangerous_mask_threshold_{t_label}days.npy", tpc_mask)
    np.save(f"bat_dangerous_mask_threshold_{t_label}days.npy", bat_mask)

# ---------------------------------------------------------------------------
# 4. Also save the summary table itself as a CSV, for easy reference
#    or inclusion in the report as a table.
# ---------------------------------------------------------------------------
import csv

with open("dangerous_case_threshold_sensitivity.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["threshold_days", "tpc_n_dangerous", "tpc_pct_dangerous",
                      "bat_n_dangerous", "bat_pct_dangerous"])
    for i in range(len(thresholds)):
        writer.writerow([
            results["threshold"][i],
            results["tpc_n_dangerous"][i],
            f"{results['tpc_pct_dangerous'][i]:.2f}",
            results["bat_n_dangerous"][i],
            f"{results['bat_pct_dangerous'][i]:.2f}",
        ])

print("\nSaved per-threshold dangerous-case masks:")
for t in thresholds:
    t_label = str(abs(t)).replace(".", "_")
    print(f"  tpc_dangerous_mask_threshold_{t_label}days.npy, bat_dangerous_mask_threshold_{t_label}days.npy")
print("\nSaved summary table: dangerous_case_threshold_sensitivity.csv")
print("Done.")