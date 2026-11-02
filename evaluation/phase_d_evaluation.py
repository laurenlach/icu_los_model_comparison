import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score


# 1. Load both models' predictions and true values
tpc_preds = np.load("tpc_test_predictions_200epoch.npy")
tpc_targets = np.load("tpc_test_targets_200epoch.npy")

bat_preds = np.load("bat_test_predictions.npy")
bat_targets = np.load("bat_test_targets.npy")

print("TPC test set size:", len(tpc_preds))
print("BAT test set size:", len(bat_preds))


# 2. PRIMARY ANALYSIS: continuous prediction error

def error_analysis(preds, targets, model_name):
    error = preds - targets  # per-patient error, in days
    abs_error = np.abs(error)

    mae = abs_error.mean()
    rmse = np.sqrt((error ** 2).mean())

    print(f"\n=== {model_name}: Prediction Error Analysis (PRIMARY) ===")
    print(f"MAE: {mae:.4f} days")
    print(f"RMSE: {rmse:.4f} days")
    print(f"Mean error (bias): {error.mean():.4f} days "
          f"({'tends to over-predict' if error.mean() > 0 else 'tends to under-predict'})")
    print(f"Std of error: {error.std():.4f} days")
    print(f"Median absolute error: {np.median(abs_error):.4f} days")

    return error, abs_error

tpc_error, tpc_abs_error = error_analysis(tpc_preds, tpc_targets, "TPC")
bat_error, bat_abs_error = error_analysis(bat_preds, bat_targets, "BAT")


# 3. Identify DANGEROUS cases

def dangerous_case_analysis(error, targets, model_name, threshold=-3.0):
    dangerous_mask = error <= threshold
    n_dangerous = dangerous_mask.sum()
    pct_dangerous = 100 * n_dangerous / len(error)

    print(f"\n=== {model_name}: Dangerous Case Analysis (error <= {threshold} days) ===")
    print(f"Number of dangerous cases: {n_dangerous} ({pct_dangerous:.2f}% of test set)")
    if n_dangerous > 0:
        print(f"Mean true LOS for dangerous cases: {targets[dangerous_mask].mean():.2f} days")
        print(f"Mean error for dangerous cases: {error[dangerous_mask].mean():.2f} days")

    return dangerous_mask

tpc_dangerous = dangerous_case_analysis(tpc_error, tpc_targets, "TPC")
bat_dangerous = dangerous_case_analysis(bat_error, bat_targets, "BAT")


# 4. SECONDARY ANALYSIS: 7-day cutoff binary classification 

def binary_cutoff_analysis(preds, targets, model_name, cutoff=7.0):
    true_binary = (targets >= cutoff).astype(int)
    pred_binary = (preds >= cutoff).astype(int)

    # Use raw predicted LOS as a continuous "score" for AUROC/AUPRC,
    # since it's a natural ranking signal for "how prolonged" a stay is predicted to be
    auroc = roc_auc_score(true_binary, preds)
    auprc = average_precision_score(true_binary, preds)
    precision = precision_score(true_binary, pred_binary, zero_division=0)

    n_prolonged = true_binary.sum()
    print(f"\n=== {model_name}: Binary 7-Day Cutoff Analysis (SECONDARY) ===")
    print(f"Prolonged stays in test set: {n_prolonged} ({100*n_prolonged/len(targets):.1f}%)")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"Precision: {precision:.4f}")

    return true_binary, pred_binary

tpc_true_bin, tpc_pred_bin = binary_cutoff_analysis(tpc_preds, tpc_targets, "TPC")
bat_true_bin, bat_pred_bin = binary_cutoff_analysis(bat_preds, bat_targets, "BAT")


# 5. Save 

np.save("tpc_error.npy", tpc_error)
np.save("tpc_dangerous_mask.npy", tpc_dangerous)
np.save("bat_error.npy", bat_error)
np.save("bat_dangerous_mask.npy", bat_dangerous)

print("\nSaved: tpc_error.npy, tpc_dangerous_mask.npy, bat_error.npy, bat_dangerous_mask.npy")
print("Done.")