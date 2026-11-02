import numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import Ridge

SOFA_SENSORS = {
    "map":   {"idx": 27, "worse": "low"},
    "bun":   {"idx": 9,  "worse": "high"},
    "crea":  {"idx": 15, "worse": "high"},
    "bili":  {"idx": 6,  "worse": "high"},
    "plt":   {"idx": 39, "worse": "low"},
    "fio2":  {"idx": 19, "worse": "high"},
    "o2sat": {"idx": 35, "worse": "low"},
}

# Signed ablation rhos from earlier runs (cohort-mean replacement, all patients)
TPC_ABL = {"map": -0.890, "bun": +0.878, "crea": -0.747, "bili": +0.311, "plt": +0.130, "fio2": +0.520, "o2sat": -0.723}
BAT_ABL = {"map": -0.770, "bun": +0.707, "crea": +0.231, "bili": +0.401, "plt": +0.227, "fio2": +0.104, "o2sat": -0.495}

X = np.load("tpc_X_scaled.npy")
rows = np.load("tpc_test_rows.npy")
y = np.load("tpc_test_targets_fixed.npy")
X_test = X[rows]                                   # (n, 48, 24)
y_rank = rankdata(y) / len(y)                      # rank scale, consistent with the Spearman results

def fit(features):
    m = Ridge(alpha=1.0).fit(features, y_rank)
    return m.coef_, m.score(features, y_rank)

coef_mean, r2_mean = fit(X_test.mean(axis=2))      # 24-hour average of every feature
coef_last, r2_last = fit(X_test[:, :, 23])         # last hour only (TPC leans on the last hours)
print(f"Linear model R^2 (24h mean features): {r2_mean:.3f} | (last-hour features): {r2_last:.3f}\n")

print("Positive = sicker reading goes with LONGER stay.")
print(f"{'sensor':6s} | {'marginal':>8s} | {'conditional 24h':>15s} | {'conditional last h':>18s} | {'TPC ablation':>12s} | {'BAT ablation':>12s}")
for name, info in SOFA_SENSORS.items():
    idx = info["idx"]
    sign = -1.0 if info["worse"] == "low" else 1.0
    abn = sign * X_test[:, idx, :].mean(axis=1)
    marginal = spearmanr(abn, y)[0]
    print(f"{name:6s} | {marginal:+8.3f} | {sign*coef_mean[idx]:+15.3f} | {sign*coef_last[idx]:+18.3f} | "
          f"{TPC_ABL[name]:+12.3f} | {BAT_ABL[name]:+12.3f}")