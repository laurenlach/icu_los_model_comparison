import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ORGAN_SENSOR_MAP = {
    "sofa_cardio_comp": [27],
    #"sofa_renal_comp":  [9, 15],
    "sofa_renal_comp":  [15, 46],
    "sofa_liver_comp":  [6],
    "sofa_coag_comp":   [39],
    "sofa_resp_comp":   [19, 35],
}

sofa = pd.read_csv("sofa_24h_scores.csv")
sofa = sofa.rename(columns={"icustay_id": "stay_id"})

tpc_shap = np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1)  # (10367, 48, 24)
tpc_ids = pd.read_csv("tpc_shap_full_testset_stay_ids_fixed.csv")

results = []
for organ, sensor_idx in ORGAN_SENSOR_MAP.items():
    attribution_mass = np.abs(tpc_shap[:, sensor_idx, :]).sum(axis=(1, 2))
    df = pd.DataFrame({"stay_id": tpc_ids["stay_id"], "attribution": attribution_mass})
    df = df.merge(sofa[["stay_id", organ]], on="stay_id", how="inner")
    df = df.dropna(subset=[organ])

    rho, p = spearmanr(df[organ], df["attribution"])
    results.append({"organ": organ, "n": len(df), "tpc_fixed_rho": rho, "p": p})

    # Save per-organ alignment file
    out_df = df[["stay_id", organ, "attribution"]].rename(columns={"attribution": "attribution_mass"})
    out_df.to_csv(f"alignment_{organ}_TPC_fixed.csv", index=False)

results_df = pd.DataFrame(results)
print("=== TPC (FIXED) SOFA-Alignment, window-averaged ===")
print(results_df.to_string(index=False))

print("\n(OLD buggy TPC: cardio=-0.2833, renal=0.2779, hepatic=0.4702, coag=-0.0350, resp=-0.0025)")
print("(BAT: cardio=0.2547, renal=0.5761, hepatic=0.5347, coag=0.5424, resp=0.4750)")

results_df.to_csv("sofa_alignment_fixed_window.csv", index=False)
print("\nSaved: sofa_alignment_fixed_window.csv + per-organ alignment_*_TPC_fixed.csv files")