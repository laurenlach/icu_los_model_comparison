import numpy as np
import pandas as pd

feature_names = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun",
    "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2",
    "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
    "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt",
    "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine", "wbc"
]

N_BOOTSTRAP = 1000
np.random.seed(42)

def bootstrap_top10_frequency(attr_array, model_name):
    n_patients = attr_array.shape[0]
    mean_attr_full = np.abs(attr_array).mean(axis=2)  # (n, 48) per-patient, per-feature

    top10_counts = np.zeros(48, dtype=int)

    for b in range(N_BOOTSTRAP):
        sample_idx = np.random.choice(n_patients, size=n_patients, replace=True)
        sample_importance = mean_attr_full[sample_idx].mean(axis=0)
        top10 = np.argsort(-sample_importance)[:10]
        top10_counts[top10] += 1

    freq_pct = 100 * top10_counts / N_BOOTSTRAP
    df = pd.DataFrame({"feature": feature_names, "top10_frequency_pct": freq_pct})
    df = df.sort_values("top10_frequency_pct", ascending=False)

    print(f"\n=== {model_name}: % of {N_BOOTSTRAP} bootstrap samples where feature is in top-10 ===")
    print(df.head(15).to_string(index=False))

    return df

tpc_shap = np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1)
tpc_boot = bootstrap_top10_frequency(tpc_shap, "TPC (fixed)")
tpc_boot.to_csv("bootstrap_stability_tpc_fixed.csv", index=False)

bat_ig = np.load("bat_ig_values_full_testset_healthybaseline.npy")
bat_boot = bootstrap_top10_frequency(bat_ig, "BAT")
bat_boot.to_csv("bootstrap_stability_bat.csv", index=False)

print("\nSaved: bootstrap_stability_tpc_fixed.csv, bootstrap_stability_bat.csv")