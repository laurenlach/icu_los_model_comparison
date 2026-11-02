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

# Direction of "worst" per sensor index, matching SOFA convention
WORST_DIRECTION = {
    27: "min", 39: "min", 35: "min",           # map, plt, o2sat - lower is worse
    15: "max", 9: "max", 6: "max", 19: "max",  # crea, bun, bili, fio2 - higher is worse
}

sofa = pd.read_csv("sofa_24h_scores.csv")
sofa = sofa.rename(columns={"icustay_id": "stay_id"})

bat_ig = np.load("bat_ig_values_full_testset_healthybaseline.npy")   # (10367, 48, 24)
bat_ids = pd.read_csv("bat_ig_full_testset_stay_ids_healthybaseline.csv")

# Need BAT's raw scaled X, subset to the same test patients, sensor-first orientation
bat_X = np.load("bat_X_scaled.npy")            # (51835, 24, 48)
bat_test_rows = np.load("bat_test_rows.npy")
bat_X_test = np.transpose(bat_X[bat_test_rows], (0, 2, 1))  # -> (10367, 48, 24)

bat_error = np.load("bat_error.npy")
bat_dangerous = np.load("bat_dangerous_mask_threshold_3_0days.npy")

assert bat_ig.shape[0] == bat_X_test.shape[0] == len(bat_ids) == len(bat_error) == len(bat_dangerous)


def worst_moment_attribution(attr_array, X_array, sensor_idx):
    """|attribution| at the timestep where this sensor's raw value was worst."""
    sensor_series = X_array[:, sensor_idx, :]  # (n_patients, 24)
    if WORST_DIRECTION[sensor_idx] == "min":
        worst_t = np.argmin(sensor_series, axis=1)
    else:
        worst_t = np.argmax(sensor_series, axis=1)
    n = attr_array.shape[0]
    return np.array([np.abs(attr_array[i, sensor_idx, worst_t[i]]) for i in range(n)])


def run_correlations(attr_array, X_array, ids_df, dangerous_mask, model_name):
    results = []
    for organ, sensor_idx_list in ORGAN_SENSOR_MAP.items():
        # Average worst-moment attribution across the organ's sensor(s)
        sensor_vals = [worst_moment_attribution(attr_array, X_array, idx) for idx in sensor_idx_list]
        attribution = np.mean(sensor_vals, axis=0)

        attr_df = pd.DataFrame({"stay_id": ids_df["stay_id"], "attribution": attribution})
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
                "model": model_name, "organ": organ, "group": group_name,
                "n_patients": n, "spearman_rho": rho, "p_value": pval,
            })
    return results


results = run_correlations(bat_ig, bat_X_test, bat_ids, bat_dangerous, "BAT_healthybaseline_worstmoment")
results_df = pd.DataFrame(results)
results_df.to_csv("sofa_alignment_by_accuracy_group_worstmoment_healthybaseline.csv", index=False)

print(results_df.to_string(index=False))
print("\nSaved: sofa_alignment_by_accuracy_group_worstmoment_healthybaseline.csv")