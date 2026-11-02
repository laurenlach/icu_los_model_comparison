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
WORST_DIRECTION = {27: "min", 39: "min", 35: "min", 15: "max", 9: "max", 6: "max", 19: "max"}

sofa = pd.read_csv("sofa_24h_scores.csv")
sofa = sofa.rename(columns={"icustay_id": "stay_id"})

tpc_shap = np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1)  # (10367, 48, 24)
tpc_ids = pd.read_csv("tpc_shap_full_testset_stay_ids_fixed.csv")

X = np.load("tpc_X_scaled.npy")
test_rows = np.load("tpc_test_rows.npy")
X_test = X[test_rows]  # (10367, 48, 24)

def worst_moment_attribution(attr_array, X_arr, sensor_idx):
    sensor_series = X_arr[:, sensor_idx, :]
    if WORST_DIRECTION[sensor_idx] == "min":
        worst_t = np.argmin(sensor_series, axis=1)
    else:
        worst_t = np.argmax(sensor_series, axis=1)
    n = attr_array.shape[0]
    return np.array([np.abs(attr_array[i, sensor_idx, worst_t[i]]) for i in range(n)])

results = []
for organ, sensor_idx_list in ORGAN_SENSOR_MAP.items():
    vals = [worst_moment_attribution(tpc_shap, X_test, idx) for idx in sensor_idx_list]
    attribution = np.mean(vals, axis=0)

    df = pd.DataFrame({"stay_id": tpc_ids["stay_id"], "attribution": attribution})
    df = df.merge(sofa[["stay_id", organ]], on="stay_id", how="inner")
    df = df.dropna(subset=[organ])

    rho, p = spearmanr(df[organ], df["attribution"])
    results.append({"organ": organ, "n": len(df), "tpc_fixed_worst_rho": rho, "p": p})

results_df = pd.DataFrame(results)
print("=== TPC (FIXED) SOFA-Alignment, worst-moment ===")
print(results_df.to_string(index=False))

print("\n(OLD buggy TPC worst-moment: cardio=0.0032, renal=0.1104, hepatic=0.0864, coag=0.0324, resp=-0.0304)")
print("(BAT worst-moment: cardio=0.2968, renal=0.4807, hepatic=0.5207, coag=0.4341, resp=0.3456)")

results_df.to_csv("sofa_alignment_fixed_worstmoment.csv", index=False)
print("\nSaved: sofa_alignment_fixed_worstmoment.csv")