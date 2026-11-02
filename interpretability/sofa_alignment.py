import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# --- Sensor-to-SOFA mapping, confirmed in Step 3 ---
SENSOR_INDEX = {
    "map": 27, "crea": 15, "bun": 9, "bili": 6,
    "plt": 39, "fio2": 19, "o2sat": 35,
}

ORGAN_SENSOR_MAP = {
    "sofa_cardio_comp": ["map"],
    "sofa_renal_comp":  ["crea", "bun"],
    "sofa_liver_comp":  ["bili"],
    "sofa_coag_comp":   ["plt"],
    "sofa_resp_comp":   ["fio2", "o2sat"],
}

# --- Load SOFA scores (from Step 1) ---
sofa = pd.read_csv("sofa_24h_scores.csv")
sofa = sofa.rename(columns={"icustay_id": "stay_id"})  # adjust if this file already uses stay_id

# --- Load TPC SHAP + BAT IG attribution arrays ---
tpc_shap = np.load("tpc_shap_values_full_testset.npy")   # expect (N, 48, 24, 1)
if tpc_shap.ndim == 4:
    tpc_shap = tpc_shap.squeeze(-1)                        # -> (N, 48, 24)
tpc_ids = pd.read_csv("tpc_shap_full_testset_stay_ids.csv")

bat_ig = np.load("bat_ig_values_full_testset.npy")        # expect (N, 48, 24)
bat_ids = pd.read_csv("bat_ig_full_testset_stay_ids.csv")

assert tpc_shap.shape[0] == len(tpc_ids), "TPC attribution array and ID file length mismatch"
assert bat_ig.shape[0] == len(bat_ids), "BAT attribution array and ID file length mismatch"


def attribution_mass(values, ids_df, id_col, sensor_indices):
    """
    Sum |attribution| over the 24h window for the given sensor channel(s).
    Multiple sensors (e.g. renal = creatinine + BUN) are summed together,
    since SOFA doesn't distinguish which one drove the organ's severity score.
    """
    mass = np.abs(values[:, sensor_indices, :]).sum(axis=(1, 2))
    return pd.DataFrame({"stay_id": ids_df[id_col].values, "attribution_mass": mass})


results = []

for organ_col, sensors in ORGAN_SENSOR_MAP.items():
    sensor_idx = [SENSOR_INDEX[s] for s in sensors]

    tpc_mass = attribution_mass(tpc_shap, tpc_ids, "stay_id", sensor_idx)
    bat_mass = attribution_mass(bat_ig, bat_ids, "stay_id", sensor_idx)

    for model_name, mass_df in [("TPC", tpc_mass), ("BAT", bat_mass)]:
        merged = mass_df.merge(sofa[["stay_id", organ_col]], on="stay_id", how="inner")

        # Drop NA SOFA sub-scores (no lab/vital drawn in the 24h window) rather
        # than imputing 0 — imputing would fabricate a "healthy" ground truth
        # for patients who simply weren't tested (see Step 1 discussion).
        merged = merged.dropna(subset=[organ_col])

        rho, pval = spearmanr(merged[organ_col], merged["attribution_mass"])

        results.append({
            "organ_system": organ_col.replace("sofa_", "").replace("_comp", ""),
            "model": model_name,
            "n_patients": len(merged),
            "spearman_rho": rho,
            "p_value": pval,
        })

        # Save the per-patient merged table too — Step 5 needs this joined
        # against ICD-9 phenotype to break the alignment score out by disease group.
        merged.to_csv(f"alignment_{organ_col}_{model_name}.csv", index=False)

summary = pd.DataFrame(results)
print(summary.to_string(index=False))
summary.to_csv("sofa_alignment_summary.csv", index=False)