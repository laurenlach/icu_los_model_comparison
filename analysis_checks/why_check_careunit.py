import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

RAW = r"..\mimic-iii-clinical-database-1.4\mimic-iii-clinical-database-1.4"
icu = pd.read_csv(os.path.join(RAW, "ICUSTAYS.csv.gz"))
icu.columns = icu.columns.str.lower()

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

ids_df = pd.read_csv("tpc_shap_full_testset_stay_ids.csv")
id_col = next(c for c in ids_df.columns
              if pd.api.types.is_numeric_dtype(ids_df[c]) and ids_df[c].between(200000, 300000).all())
test = ids_df[[id_col]].rename(columns={id_col: "icustay_id"}).merge(
    icu[["icustay_id", "first_careunit"]], on="icustay_id", how="left")

X = np.load("tpc_X_scaled.npy")
rows = np.load("tpc_test_rows.npy")
y = np.load("tpc_test_targets_fixed.npy")
X_test = X[rows]
assert len(test) == len(y)

unit = test["first_careunit"].fillna("UNKNOWN").values
counts = pd.Series(unit).value_counts()
print("Patients and median true LOS by first care unit:")
for u, n in counts.items():
    print(f"  {u:8s} n={n:5d} | median LOS={np.median(y[unit == u]):.2f} d")

big_units = [u for u, n in counts.items() if n >= 300]
dummies = pd.get_dummies(unit).values.astype(float)

print("\nCorrelation of sensor abnormality with TRUE LOS")
header = f"{'sensor':6s} | {'pooled':>7s} | {'ctrl unit':>9s} | " + " | ".join(f"{u:>7s}" for u in big_units)
print(header)
for name, info in SOFA_SENSORS.items():
    idx = info["idx"]
    raw = X_test[:, idx, :].mean(axis=1)
    abn = -raw if info["worse"] == "low" else raw
    pooled = spearmanr(abn, y)[0]
    ctrl = partial_correlation(abn, y, dummies)
    per_unit = [spearmanr(abn[unit == u], y[unit == u])[0] for u in big_units]
    print(f"{name:6s} | {pooled:+7.3f} | {ctrl:+9.3f} | " + " | ".join(f"{r:+7.3f}" for r in per_unit))