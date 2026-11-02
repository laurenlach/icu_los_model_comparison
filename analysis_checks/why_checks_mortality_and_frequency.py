import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

RAW = r"..\mimic-iii-clinical-database-1.4\mimic-iii-clinical-database-1.4"
adm_path = os.path.join(RAW, "ADMISSIONS.csv.gz")
icu_path = os.path.join(RAW, "ICUSTAYS.csv.gz")
for p in (adm_path, icu_path):
    if not os.path.exists(p):
        raise SystemExit(f"Cannot find {p}. Check that folder for the file and tell me what it contains.")

feature_names = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun",
    "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2",
    "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
    "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt",
    "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine", "wbc"
]
SOFA_SENSORS = {
    "map":   {"idx": 27, "worse": "low"},
    "bun":   {"idx": 9,  "worse": "high"},
    "crea":  {"idx": 15, "worse": "high"},
    "bili":  {"idx": 6,  "worse": "high"},
    "plt":   {"idx": 39, "worse": "low"},
    "fio2":  {"idx": 19, "worse": "high"},
    "o2sat": {"idx": 35, "worse": "low"},
}

def partial_correlation(x, y, controls):
    rx = x - LinearRegression().fit(controls, x).predict(controls)
    ry = y - LinearRegression().fit(controls, y).predict(controls)
    return spearmanr(rx, ry)[0]

# ---------------- Build death flags for the test patients ----------------
adm = pd.read_csv(adm_path); adm.columns = adm.columns.str.lower()
icu = pd.read_csv(icu_path); icu.columns = icu.columns.str.lower()
m = icu[["icustay_id", "hadm_id", "outtime"]].merge(
    adm[["hadm_id", "deathtime", "hospital_expire_flag"]], on="hadm_id", how="left")
m["outtime"] = pd.to_datetime(m["outtime"])
m["deathtime"] = pd.to_datetime(m["deathtime"])
m["hospital_death"] = (m["hospital_expire_flag"] == 1).astype(int)
m["icu_death"] = ((m["hospital_expire_flag"] == 1) &
                  (m["deathtime"] <= m["outtime"] + pd.Timedelta(hours=12))).astype(int)

ids_df = pd.read_csv("tpc_shap_full_testset_stay_ids.csv")
id_col = next(c for c in ids_df.columns
              if pd.api.types.is_numeric_dtype(ids_df[c]) and ids_df[c].between(200000, 300000).all())
test = ids_df[[id_col]].rename(columns={id_col: "icustay_id"}).merge(
    m[["icustay_id", "hospital_death", "icu_death"]], on="icustay_id", how="left")

X = np.load("tpc_X_scaled.npy")
rows = np.load("tpc_test_rows.npy")
y = np.load("tpc_test_targets_fixed.npy")
X_test = X[rows]
assert len(test) == len(y), f"stay-id list has {len(test)} rows but test set has {len(y)}"
print(f"Test patients with no death info found: {int(test['icu_death'].isna().sum())}")
icu_death = test["icu_death"].fillna(0).values.astype(bool)
hosp_death = test["hospital_death"].fillna(0).values.astype(bool)

print(f"\nICU deaths: {icu_death.sum()} ({100*icu_death.mean():.1f}%) | in-hospital deaths: {hosp_death.sum()} ({100*hosp_death.mean():.1f}%)")
print(f"Median true LOS: ICU deaths = {np.median(y[icu_death]):.2f} d, survivors = {np.median(y[~icu_death]):.2f} d")

# ---------------- A/B: does the backwards pattern survive removing deaths? ----------------
print("\n=== Correlation of sensor abnormality with TRUE LOS ===")
print(f"{'sensor':6s} | {'all':>7s} | {'no ICU deaths':>13s} | {'ICU deaths only':>15s} | {'no hosp deaths':>14s} | {'partial (ctrl ICU death)':>24s}")
for name, info in SOFA_SENSORS.items():
    idx = info["idx"]
    raw = X_test[:, idx, :].mean(axis=1)
    abn = -raw if info["worse"] == "low" else raw
    r_all = spearmanr(abn, y)[0]
    r_surv = spearmanr(abn[~icu_death], y[~icu_death])[0]
    r_dead = spearmanr(abn[icu_death], y[icu_death])[0]
    r_surv_h = spearmanr(abn[~hosp_death], y[~hosp_death])[0]
    r_part = partial_correlation(abn, y, icu_death.reshape(-1, 1).astype(float))
    print(f"{name:6s} | {r_all:+7.3f} | {r_surv:+13.3f} | {r_dead:+15.3f} | {r_surv_h:+14.3f} | {r_part:+24.3f}")

print("\nAbnormality vs ICU death itself:")
for name, info in SOFA_SENSORS.items():
    raw = X_test[:, info["idx"], :].mean(axis=1)
    abn = -raw if info["worse"] == "low" else raw
    print(f"{name:6s} rho={spearmanr(abn, icu_death.astype(float))[0]:+.3f}")

# ---------------- C: does feature importance follow how often a feature is measured? ----------------
print("\n=== C. Permutation importance vs how often each feature is really measured ===")
mask = np.load("bat_mask.npy")[np.load("bat_test_rows.npy")]
observed = mask.mean(axis=(0, 1))                 # fraction of patient-hours measured, per feature
for label, path in [("TPC", "permutation_importance_tpc_fixed.csv"), ("BAT", "permutation_importance_bat_full.csv")]:
    if not os.path.exists(path):
        print(f"{label}: {path} not found in this folder, skipped")
        continue
    imp = pd.read_csv(path).set_index("feature").reindex(feature_names)["mae_increase"].values
    rho, p = spearmanr(imp, observed)
    print(f"{label}: rho(importance, observed rate) = {rho:+.3f} (p={p:.3g})")