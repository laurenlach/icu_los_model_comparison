import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Sensor-to-SOFA mapping
ORGAN_SENSOR_MAP = {
    "sofa_cardio_comp": [27],       # map
    "sofa_renal_comp":  [9, 15],    # bun, crea
    "sofa_liver_comp":  [6],        # bili
    "sofa_coag_comp":   [39],       # plt
    "sofa_resp_comp":   [19, 35],   # fio2, o2sat
}

# Load SOFA scores
sofa = pd.read_csv("sofa_24h_scores.csv")
sofa = sofa.rename(columns={"icustay_id": "stay_id"})

# Load TPC SHAP + BAT IG attribution arrays and their patient order
tpc_shap = np.load("tpc_shap_values_full_testset.npy")
if tpc_shap.ndim == 4:
    tpc_shap = tpc_shap.squeeze(-1)  # (N, 48, 24, 1) -> (N, 48, 24)
tpc_ids = pd.read_csv("tpc_shap_full_testset_stay_ids.csv")

bat_ig = np.load("bat_ig_values_full_testset.npy")
bat_ids = pd.read_csv("bat_ig_full_testset_stay_ids.csv")

# Load prediction error and the existing -3 day dangerous-case masks 
tpc_error = np.load("tpc_error.npy")
bat_error = np.load("bat_error.npy")
tpc_dangerous = np.load("tpc_dangerous_mask_threshold_3_0days.npy")
bat_dangerous = np.load("bat_dangerous_mask_threshold_3_0days.npy")

assert tpc_shap.shape[0] == len(tpc_ids) == len(tpc_error) == len(tpc_dangerous)
assert bat_ig.shape[0] == len(bat_ids) == len(bat_error) == len(bat_dangerous)


def attribution_mass(values, ids_df, sensor_indices):
    """Sum |attribution| over the 24h window for the given sensor channel(s)."""
    summed = np.abs(values[:, sensor_indices, :]).sum(axis=(1, 2))
    return pd.DataFrame({"stay_id": ids_df["stay_id"], "attribution": summed})


def run_correlations(values, ids_df, dangerous_mask, model_name):
    
    accurate_mask = ~dangerous_mask

    results = []
    for organ, sensor_idx in ORGAN_SENSOR_MAP.items():
        attr_df = attribution_mass(values, ids_df, sensor_idx)
        merged = attr_df.merge(sofa[["stay_id", organ]], on="stay_id", how="inner")
        merged = merged.dropna(subset=[organ])


        group_lookup = pd.DataFrame({"stay_id": ids_df["stay_id"], "dangerous": dangerous_mask})
        merged = merged.merge(group_lookup, on="stay_id", how="inner")

        for group_name, group_mask in [("accurate", ~merged["dangerous"]), ("dangerous", merged["dangerous"])]:
            subset = merged[group_mask]
            n = len(subset)
            if n < 10:
                rho, pval = np.nan, np.nan
            else:
                rho, pval = spearmanr(subset[organ], subset["attribution"])
            results.append({
                "model": model_name,
                "organ": organ,
                "group": group_name,
                "n_patients": n,
                "spearman_rho": rho,
                "p_value": pval,
            })
    return results


tpc_results = run_correlations(tpc_shap, tpc_ids, tpc_dangerous, "TPC")
bat_results = run_correlations(bat_ig, bat_ids, bat_dangerous, "BAT")

results_df = pd.DataFrame(tpc_results + bat_results)
results_df.to_csv("sofa_alignment_by_accuracy_group.csv", index=False)

print(results_df.to_string(index=False))
print("\nSaved: sofa_alignment_by_accuracy_group.csv")