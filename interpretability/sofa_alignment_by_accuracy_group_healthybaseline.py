import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ORGAN_SENSOR_MAP = {
    "sofa_cardio_comp": [27],
    "sofa_renal_comp":  [9, 15],
    "sofa_liver_comp":  [6],
    "sofa_coag_comp":   [39],
    "sofa_resp_comp":   [19, 35],
}

sofa = pd.read_csv("sofa_24h_scores.csv")
sofa = sofa.rename(columns={"icustay_id": "stay_id"})

# NEW: healthy-baseline BAT IG values, instead of the old all-zeros ones
bat_ig = np.load("bat_ig_values_full_testset_healthybaseline.npy")
bat_ids = pd.read_csv("bat_ig_full_testset_stay_ids_healthybaseline.csv")

# Error and dangerous-case masks are unchanged - these come from the model's
# actual predictions, not from attribution, so the baseline change doesn't affect them
bat_error = np.load("bat_error.npy")
bat_dangerous = np.load("bat_dangerous_mask_threshold_3_0days.npy")

assert bat_ig.shape[0] == len(bat_ids) == len(bat_error) == len(bat_dangerous)


def attribution_mass(values, ids_df, sensor_indices):
    summed = np.abs(values[:, sensor_indices, :]).sum(axis=(1, 2))
    return pd.DataFrame({"stay_id": ids_df["stay_id"], "attribution": summed})


def run_correlations(values, ids_df, dangerous_mask, model_name):
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


bat_results = run_correlations(bat_ig, bat_ids, bat_dangerous, "BAT_healthybaseline")
results_df = pd.DataFrame(bat_results)
results_df.to_csv("sofa_alignment_by_accuracy_group_healthybaseline.csv", index=False)

print(results_df.to_string(index=False))
print("\nSaved: sofa_alignment_by_accuracy_group_healthybaseline.csv")