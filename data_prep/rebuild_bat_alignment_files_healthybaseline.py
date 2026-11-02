import numpy as np
import pandas as pd

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

bat_ig = np.load("bat_ig_values_full_testset_healthybaseline.npy")
bat_ids = pd.read_csv("bat_ig_full_testset_stay_ids_healthybaseline.csv")

assert bat_ig.shape[0] == len(bat_ids)

for organ, sensor_idx in ORGAN_SENSOR_MAP.items():
    attribution_mass = np.abs(bat_ig[:, sensor_idx, :]).sum(axis=(1, 2))

    out_df = pd.DataFrame({
        "stay_id": bat_ids["stay_id"],
        "attribution_mass": attribution_mass,
    })
    out_df = out_df.merge(sofa[["stay_id", organ]], on="stay_id", how="inner")


    before = len(out_df)
    out_df = out_df.dropna(subset=[organ])
    after = len(out_df)
    print(f"{organ}: dropped {before - after} patients with missing SOFA sub-score "
          f"({after} remaining)")

    out_df = out_df[["stay_id", organ, "attribution_mass"]]

    filename = f"alignment_{organ}_BAT_healthybaseline.csv"
    out_df.to_csv(filename, index=False)
    print(f"Saved: {filename} ({len(out_df)} patients)\n")

print("Done. Original alignment_*_BAT.csv (old baseline) and all TPC files were NOT touched.")