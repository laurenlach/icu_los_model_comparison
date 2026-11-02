
# Checks after the BAT re-run on the MIMIC-IV + eICU checkpoint.
# 1. Does TPC still have a safety buffer on stays of about a week that BAT lacks?
# 2. Which disease groups changed in how often TPC has fewer dangerous cases?
# Run from the YAIB folder. Saves two new _ivEicu CSV files.

import numpy as np
import pandas as pd

y = np.load("tpc_test_targets_fixed.npy").ravel().astype(np.float64)
preds = {
    "TPC": np.load("tpc_test_predictions_fixed.npy").ravel().astype(np.float64),
    "BAT original": np.load("bat_test_predictions.npy").ravel().astype(np.float64),
    "BAT new": np.load("bat_test_predictions_ivEicu.npy").ravel().astype(np.float64),
}
err = {k: p - y for k, p in preds.items()}
assert all(len(v) == len(y) == 10367 for v in preds.values()), "STOP: wrong number of stays"


# CHECK 1
week = (y >= 5) & (y < 10)
print(f"=== CHECK 1: stays with true length 5 to 10 days (n = {week.sum()}) ===")
rows = []
for name, e in err.items():
    w = e[week]
    rows.append({
        "model": name,
        "median_error_days": np.median(w),
        "pct_over_predicted (error > 0)": 100 * (w > 0).mean(),
        "pct_short_but_safe (-3 < error <= 0)": 100 * ((w > -3) & (w <= 0)).mean(),
        "pct_dangerous (error <= -3)": 100 * (w <= -3).mean(),
        "dangerous_count": int((w <= -3).sum()),
    })
c1 = pd.DataFrame(rows).set_index("model")
print(c1.round(2).to_string())
print("\n  The claim holds if TPC over-predicts more of these stays than BAT does,")
print("  and BAT has the larger share past the 3-day line.")
print(f"  TPC over-predicts more than new BAT:    "
      f"{c1.loc['TPC', 'pct_over_predicted (error > 0)'] > c1.loc['BAT new', 'pct_over_predicted (error > 0)']}")
print(f"  New BAT has more dangerous than TPC:    "
      f"{c1.loc['BAT new', 'dangerous_count'] > c1.loc['TPC', 'dangerous_count']}")
c1.to_csv("check_rq1_week_long_stays_ivEicu.csv")


# CHECK 2

test_rows = np.load("bat_test_rows.npy")
stay_ids = pd.read_csv("bat_stay_id_order.csv")["stay_id"].values[test_rows]
df = pd.DataFrame({"stay_id": stay_ids, "y": y,
                   **{f"d_{k}": e <= -3 for k, e in err.items()}})
df = df.merge(pd.read_csv("cohort_icd9_phenotype_mapping_with_fallback.csv"),
              on="stay_id", how="inner").dropna(subset=["phenotype"])

g = df.groupby("phenotype").agg(
    n=("y", "size"),
    pct_week_long=("y", lambda s: 100 * ((s >= 5) & (s < 10)).mean()),
    TPC=("d_TPC", "sum"),
    BAT_original=("d_BAT original", "sum"),
    BAT_new=("d_BAT new", "sum"),
)
g["new_minus_TPC"] = g["BAT_new"] - g["TPC"]
g["orig_minus_TPC"] = g["BAT_original"] - g["TPC"]

def verdict(d):
    return "TPC safer" if d > 0 else ("tie" if d == 0 else "BAT safer")
g["verdict_original"] = g["orig_minus_TPC"].apply(verdict)
g["verdict_new"] = g["new_minus_TPC"].apply(verdict)
g = g.sort_values("n", ascending=False)

print("\n=== CHECK 2: dangerous cases (error <= -3) per disease group, as stay counts ===")
print(g.to_string())
print(f"\n  Original BAT: TPC safer in {(g['verdict_original'] == 'TPC safer').sum()}, "
      f"tie in {(g['verdict_original'] == 'tie').sum()}, BAT safer in {(g['verdict_original'] == 'BAT safer').sum()}   "
      f"[report: 24, 1, 0]")
print(f"  New BAT:      TPC safer in {(g['verdict_new'] == 'TPC safer').sum()}, "
      f"tie in {(g['verdict_new'] == 'tie').sum()}, BAT safer in {(g['verdict_new'] == 'BAT safer').sum()}")

changed = g[g["verdict_new"] != "TPC safer"]
print("\n=== Groups where TPC is no longer safer ===")
print(changed[["n", "pct_week_long", "TPC", "BAT_original", "BAT_new", "new_minus_TPC", "verdict_new"]].to_string())
print(f"\n  Largest gap in these groups: {int(changed['new_minus_TPC'].abs().max())} stays")
print(f"  Total stays in these groups: {int(changed['n'].sum())} of {len(df)} grouped stays")
print(f"  Cohort-wide share of week-long stays: {100 * ((df['y'] >= 5) & (df['y'] < 10)).mean():.1f}%")
g.to_csv("check_rq3_dangerous_by_group_ivEicu.csv")
print("\nSaved: check_rq1_week_long_stays_ivEicu.csv, check_rq3_dangerous_by_group_ivEicu.csv")