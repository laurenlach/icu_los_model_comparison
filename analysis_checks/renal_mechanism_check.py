import numpy as np, pandas as pd
from scipy.stats import spearmanr

CREA, URINE = 15, 46
ORGAN = "sofa_renal_comp"

sofa = pd.read_csv("sofa_24h_scores.csv").rename(
    columns={"icustay_id": "stay_id"})

def check(name, arr, ids):
    out = pd.DataFrame({
        "stay_id": ids["stay_id"],
        "crea":  np.abs(arr[:, CREA,  :]).sum(axis=1),
        "urine": np.abs(arr[:, URINE, :]).sum(axis=1),
    })
    out["both"] = out.crea + out.urine
    out = out.merge(sofa[["stay_id", ORGAN]], on="stay_id").dropna(subset=[ORGAN])
    print(f"{name}  (n={len(out)})")
    for col in ["crea", "urine", "both"]:
        r, p = spearmanr(out[ORGAN], out[col])
        print(f"   {col:6s} vs renal subscore   rho {r:+.3f}   p {p:.1e}")
    print()

bat = np.load("bat_ig_values_full_testset_healthybaseline.npy")
bat_ids = pd.read_csv("bat_ig_full_testset_stay_ids_healthybaseline.csv")
check("BAT", bat, bat_ids)

tpc = np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1)
tpc_ids = pd.read_csv("tpc_shap_full_testset_stay_ids_fixed.csv")
check("TPC", tpc, tpc_ids)