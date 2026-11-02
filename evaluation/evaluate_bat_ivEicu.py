#evaluate the new BAT (MIMIC-IV + eICU checkpoint)


import os
import csv
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score

TAG = "_ivEicu"


# 1. Load predictions and check alignment BEFORE computing anything

tpc_preds = np.load("tpc_test_predictions_fixed.npy").ravel().astype(np.float64)
tpc_targets = np.load("tpc_test_targets_fixed.npy").ravel().astype(np.float64)
old_preds = np.load("bat_test_predictions.npy").ravel().astype(np.float64)
old_targets = np.load("bat_test_targets.npy").ravel().astype(np.float64)
new_preds = np.load(f"bat_test_predictions{TAG}.npy").ravel().astype(np.float64)
new_targets = np.load(f"bat_test_targets{TAG}.npy").ravel().astype(np.float64)
tpc_test_rows = np.load("tpc_test_rows.npy")
bat_test_rows = np.load("bat_test_rows.npy")

print("=== Alignment checks (all must say True) ===")
checks = {
    "All three models have 10367 test predictions":
        len(tpc_preds) == len(old_preds) == len(new_preds) == 10367,
    "TPC and BAT use the same test rows in the same order":
        np.array_equal(tpc_test_rows, bat_test_rows),
    "New BAT targets match TPC targets patient by patient":
        np.allclose(new_targets, tpc_targets),
    "Original BAT targets match TPC targets patient by patient":
        np.allclose(old_targets, tpc_targets),
    "No missing or infinite values in new BAT predictions":
        bool(np.isfinite(new_preds).all()),
    "New BAT predictions differ from original BAT (really a new model)":
        not np.allclose(new_preds, old_preds),
}
for name, ok in checks.items():
    print(f"  {name}: {ok}")
if not all(checks.values()):
    raise SystemExit("\nSTOP: an alignment check failed. Paste this output to Claude before going further.")

# One shared set of targets, so all three models are scored against the same truth.
targets = tpc_targets
models = {
    "TPC (fixed)": tpc_preds,
    "BAT original": old_preds,
    "BAT new (IV+eICU)": new_preds,
}
errors = {name: preds - targets for name, preds in models.items()}


# 2. PRIMARY: continuous error
def row(label, fmt, func):
    print(f"{label:<34}" + "".join(f"{format(func(n), fmt):>20}" for n in models))

print("\n=== Continuous error (PRIMARY) ===")
print(f"{'':<34}" + "".join(f"{n:>20}" for n in models))
row("MAE (days)", ".4f", lambda n: np.abs(errors[n]).mean())
row("RMSE (days)", ".4f", lambda n: np.sqrt((errors[n] ** 2).mean()))
row("Mean error / bias (days)", ".4f", lambda n: errors[n].mean())
row("Std of error (days)", ".4f", lambda n: errors[n].std())
row("Median absolute error (days)", ".4f", lambda n: np.median(np.abs(errors[n])))


# 3. Dangerous cases at -3 days, plus the -1 to -5 sensitivity check.
print("\n=== Dangerous cases (error <= -3 days) ===")
print(f"{'':<34}" + "".join(f"{n:>20}" for n in models))
row("Count", "d", lambda n: int((errors[n] <= -3.0).sum()))
row("Percent of test set", ".2f", lambda n: 100 * (errors[n] <= -3.0).mean())
row("Mean true LOS of dangerous (days)", ".2f", lambda n: targets[errors[n] <= -3.0].mean())
row("Mean error of dangerous (days)", ".2f", lambda n: errors[n][errors[n] <= -3.0].mean())

thresholds = [-1.0, -2.0, -3.0, -4.0, -5.0]
print("\n=== Threshold sensitivity: percent of test set that is dangerous ===")
print(f"{'Threshold':<34}" + "".join(f"{n:>20}" for n in models))
for t in thresholds:
    print(f"{t:<34}" + "".join(f"{100 * (errors[n] <= t).mean():>20.2f}" for n in models))


# 4. SECONDARY: binary 7-day cutoff. Same as phase_d_evaluation.py
true_binary = (targets >= 7.0).astype(int)
print(f"\n=== Binary 7-day cutoff (SECONDARY), prolonged stays: "
      f"{true_binary.sum()} ({100 * true_binary.mean():.1f}%) ===")
print(f"{'':<34}" + "".join(f"{n:>20}" for n in models))
row("AUROC", ".4f", lambda n: roc_auc_score(true_binary, models[n]))
row("AUPRC", ".4f", lambda n: average_precision_score(true_binary, models[n]))
row("Precision", ".4f", lambda n: precision_score(true_binary, (models[n] >= 7.0).astype(int), zero_division=0))


# 5. Paired Wilcoxon on absolute error over the whole test set.
abs_tpc = np.abs(errors["TPC (fixed)"])
abs_old = np.abs(errors["BAT original"])
abs_new = np.abs(errors["BAT new (IV+eICU)"])
print("\n=== Paired Wilcoxon on absolute error, whole test set ===")
print(f"  TPC vs BAT new:          p = {wilcoxon(abs_tpc, abs_new).pvalue:.3e}")
print(f"  TPC vs BAT original:     p = {wilcoxon(abs_tpc, abs_old).pvalue:.3e}   (for reference)")
print(f"  BAT original vs BAT new: p = {wilcoxon(abs_old, abs_new).pvalue:.3e}   (did the checkpoint change accuracy?)")


# 6. Per disease group: MAE, Wilcoxon and dangerous-case rate.

groups_done = False
if os.path.exists("bat_stay_id_order.csv") and os.path.exists("cohort_icd9_phenotype_mapping_with_fallback.csv"):
    stay_ids = pd.read_csv("bat_stay_id_order.csv")["stay_id"].iloc[bat_test_rows].values
    df = pd.DataFrame({
        "stay_id": stay_ids,
        "tpc_abs_error": abs_tpc, "bat_abs_error": abs_new,
        "tpc_dangerous": errors["TPC (fixed)"] <= -3.0,
        "bat_dangerous": errors["BAT new (IV+eICU)"] <= -3.0,
    })
    df = df.merge(pd.read_csv("cohort_icd9_phenotype_mapping_with_fallback.csv"),
                  on="stay_id", how="left").dropna(subset=["phenotype"])

    results = []
    for phenotype, g in df.groupby("phenotype"):
        n = len(g)
        tpc_mae, bat_mae = g["tpc_abs_error"].mean(), g["bat_abs_error"].mean()
        if n < 10:
            pval, note = np.nan, "n<10"
        else:
            pval = wilcoxon(g["tpc_abs_error"], g["bat_abs_error"]).pvalue
            note = "significant" if pval < 0.05 else "not significant"
        results.append({
            "phenotype": phenotype, "n_patients": n,
            "TPC_MAE_fixed": round(tpc_mae, 4), "BAT_MAE_ivEicu": round(bat_mae, 4),
            "BAT_better": bat_mae < tpc_mae, "p_value": pval, "result": note,
            "TPC_dangerous_pct": round(100 * g["tpc_dangerous"].mean(), 2),
            "BAT_dangerous_pct_ivEicu": round(100 * g["bat_dangerous"].mean(), 2),
            "TPC_fewer_dangerous": g["tpc_dangerous"].mean() < g["bat_dangerous"].mean(),
        })
    res = pd.DataFrame(results).sort_values("n_patients", ascending=False)
    print("\n=== Per disease group, TPC (fixed) vs BAT new ===")
    print(res.to_string(index=False))
    print(f"\n  Test stays assigned to a group: {len(df)}")
    print(f"  BAT new has lower MAE in {int(res['BAT_better'].sum())}/{len(res)} groups")
    print(f"  Significant (p < 0.05): {int((res['p_value'] < 0.05).sum())}/{int(res['p_value'].notna().sum())} tested groups")
    print(f"  TPC has fewer dangerous cases in {int(res['TPC_fewer_dangerous'].sum())}/{len(res)} groups")
    res.to_csv(f"significance_test_by_phenotype{TAG}.csv", index=False)
    groups_done = True
else:
    print("\n(Per disease group skipped: bat_stay_id_order.csv or "
          "cohort_icd9_phenotype_mapping_with_fallback.csv not found in this folder.)")


# 7. Did the main findings hold? 

e_tpc, e_new = errors["TPC (fixed)"], errors["BAT new (IV+eICU)"]
print("\n=== Do the original findings still hold with the new checkpoint? ===")
print(f"  BAT still has lower MAE than TPC:                 {np.abs(e_new).mean() < np.abs(e_tpc).mean()}")
print(f"  BAT still has lower RMSE than TPC:                {np.sqrt((e_new**2).mean()) < np.sqrt((e_tpc**2).mean())}")
print(f"  BAT still under-predicts on average (bias < 0):   {e_new.mean() < 0}")
print(f"  BAT still has more dangerous cases at -3 days:    {(e_new <= -3).sum() > (e_tpc <= -3).sum()}")
print(f"  ...and at every threshold from -1 to -5:          {all((e_new <= t).sum() > (e_tpc <= t).sum() for t in thresholds)}")
print(f"  BAT still has higher AUROC and AUPRC:             "
      f"{roc_auc_score(true_binary, new_preds) > roc_auc_score(true_binary, tpc_preds) and average_precision_score(true_binary, new_preds) > average_precision_score(true_binary, tpc_preds)}")



# 8. Save
np.save(f"bat_error{TAG}.npy", e_new)
np.save(f"bat_dangerous_mask{TAG}.npy", e_new <= -3.0)
for t in thresholds:
    np.save(f"bat_dangerous_mask_threshold_{str(abs(t)).replace('.', '_')}days{TAG}.npy", e_new <= t)
with open(f"dangerous_case_threshold_sensitivity{TAG}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["threshold_days", "tpc_n_dangerous", "tpc_pct_dangerous",
                "bat_n_dangerous_ivEicu", "bat_pct_dangerous_ivEicu"])
    for t in thresholds:
        w.writerow([t, int((e_tpc <= t).sum()), f"{100 * (e_tpc <= t).mean():.2f}",
                    int((e_new <= t).sum()), f"{100 * (e_new <= t).mean():.2f}"])
print(f"\nSaved: bat_error{TAG}.npy, bat_dangerous_mask{TAG}.npy, five threshold masks, "
      f"dangerous_case_threshold_sensitivity{TAG}.csv"
      + (f", significance_test_by_phenotype{TAG}.csv" if groups_done else ""))
print("Done.")