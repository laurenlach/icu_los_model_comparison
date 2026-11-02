import numpy as np
import pandas as pd

feature_names = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun",
    "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2",
    "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
    "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt",
    "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine", "wbc"
]

def top10_by_group(attr_array, dangerous_mask, model_name):
    mean_attr = np.abs(attr_array).mean(axis=2)  # (n, 48)
    accurate = mean_attr[~dangerous_mask].mean(axis=0)
    dangerous = mean_attr[dangerous_mask].mean(axis=0)

    acc_rank = np.argsort(-accurate)[:10]
    dan_rank = np.argsort(-dangerous)[:10]

    print(f"\n=== {model_name}: top-10, accurate vs dangerous ===")
    print(f"{'Accurate':<12}{'Dangerous':<12}")
    for i in range(10):
        print(f"{feature_names[acc_rank[i]]:<12}{feature_names[dan_rank[i]]:<12}")

    return pd.DataFrame({
        "feature": feature_names,
        "accurate_mean_attr": accurate,
        "dangerous_mean_attr": dangerous,
    })

# TPC (fixed)
tpc_shap = np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1)
tpc_dangerous = np.load("tpc_dangerous_mask_fixed.npy")
tpc_df = top10_by_group(tpc_shap, tpc_dangerous, "TPC (fixed)")
tpc_df.to_csv("feature_level_accuracy_split_tpc_fixed.csv", index=False)

# BAT
bat_ig = np.load("bat_ig_values_full_testset_healthybaseline.npy")
bat_dangerous = np.load("bat_dangerous_mask_threshold_3_0days.npy")
bat_df = top10_by_group(bat_ig, bat_dangerous, "BAT")
bat_df.to_csv("feature_level_accuracy_split_bat.csv", index=False)

print("\nSaved: feature_level_accuracy_split_tpc_fixed.csv, feature_level_accuracy_split_bat.csv")