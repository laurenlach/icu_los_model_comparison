import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

feature_names = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun",
    "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2",
    "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
    "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt",
    "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine", "wbc"
]

X = np.load("tpc_X_scaled.npy")
rows = np.load("tpc_test_rows.npy")
y = np.load("tpc_test_targets_fixed.npy")
X_test = X[rows]                                    # (n, 48, 24)

# ---------- 1. Sweep: implausibly extreme standardised values in ANY feature ----------
print("=== 1. Patients with |z| > 10 at any hour, per feature ===")
sweep = []
for j, f in enumerate(feature_names):
    v = X_test[:, j, :]
    n_low, n_high = int((v.min(axis=1) < -10).sum()), int((v.max(axis=1) > 10).sum())
    if n_low or n_high:
        sweep.append((f, n_low, n_high))
for f, lo, hi in sorted(sweep, key=lambda t: -(t[1] + t[2]))[:15]:
    print(f"{f:8s} z < -10: {lo:5d} patients | z > +10: {hi:5d} patients")

# ---------- 2. The o2sat artifact ----------
o2 = X_test[:, 35, :]
art = o2.min(axis=1) < -10                          # any impossible-looking reading
art_last = o2[:, -1] < -10                          # impossible reading in the LAST hour
print(f"\n=== 2. o2sat ===")
print(f"Patients with an impossible-looking reading at any hour: {art.sum()} ({100*art.mean():.1f}%)")
print(f"Patients with it in the LAST hour: {art_last.sum()} ({100*art_last.mean():.1f}%)")
print("Most common values below z=-10 (rounded):")
print(pd.Series(np.round(o2[o2 < -10], 2)).value_counts().head(5).to_string())

# ---------- 3. What happens to these patients? ----------
RAW = r"..\mimic-iii-clinical-database-1.4\mimic-iii-clinical-database-1.4"
adm = pd.read_csv(os.path.join(RAW, "ADMISSIONS.csv.gz")); adm.columns = adm.columns.str.lower()
icu = pd.read_csv(os.path.join(RAW, "ICUSTAYS.csv.gz")); icu.columns = icu.columns.str.lower()
m = icu[["icustay_id", "hadm_id", "outtime"]].merge(
    adm[["hadm_id", "deathtime", "hospital_expire_flag"]], on="hadm_id", how="left")
m["outtime"] = pd.to_datetime(m["outtime"]); m["deathtime"] = pd.to_datetime(m["deathtime"])
m["icu_death"] = ((m["hospital_expire_flag"] == 1) &
                  (m["deathtime"] <= m["outtime"] + pd.Timedelta(hours=12))).astype(int)
ids_df = pd.read_csv("tpc_shap_full_testset_stay_ids.csv")
id_col = next(c for c in ids_df.columns
              if pd.api.types.is_numeric_dtype(ids_df[c]) and ids_df[c].between(200000, 300000).all())
test = ids_df[[id_col]].rename(columns={id_col: "icustay_id"}).merge(
    m[["icustay_id", "icu_death"]], on="icustay_id", how="left")
died = test["icu_death"].fillna(0).values.astype(bool)
print(f"\nICU death rate: artifact patients {100*died[art].mean():.1f}% vs others {100*died[~art].mean():.1f}%")
print(f"Median true LOS: artifact patients {np.median(y[art]):.2f} d vs others {np.median(y[~art]):.2f} d")
print(f"Share with LOS < 1 day: artifact {100*(y[art] < 1).mean():.1f}% vs others {100*(y[~art] < 1).mean():.1f}%")

# ---------- 4. o2sat dose-response with and without the artifact patients ----------
abn = -o2.mean(axis=1)
tpc_attr = np.abs(np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1)).mean(axis=2)[:, 35]
bat_attr = np.abs(np.load("bat_ig_values_full_testset_healthybaseline.npy")).mean(axis=2)[:, 35]
tpc_eff = np.load("tpc_ablation_signed_effects.npy")[:, 6]        # column 6 = o2sat
bat_eff = None
if os.path.exists("bat_ablation_signed_effects.npy"):
    bat_eff = np.load("bat_ablation_signed_effects.npy")[:, 6]
else:
    print("\n(bat_ablation_signed_effects.npy not found in this folder - copy it from Drive BAT_colab to include BAT's ablation)")

print("\n=== 4. o2sat dose-response: all patients vs artifact patients removed ===")
for label, keep in [("all patients", np.ones_like(art, dtype=bool)), ("artifact removed", ~art)]:
    line = (f"{label:17s} n={int(keep.sum()):5d} | true LOS rho={spearmanr(abn[keep], y[keep])[0]:+.3f} "
            f"| SHAP rho={spearmanr(abn[keep], tpc_attr[keep])[0]:+.3f} | IG rho={spearmanr(abn[keep], bat_attr[keep])[0]:+.3f} "
            f"| TPC ablation rho={spearmanr(abn[keep], tpc_eff[keep])[0]:+.3f}, mean|effect|={np.abs(tpc_eff[keep]).mean():.2f} d")
    if bat_eff is not None:
        line += f" | BAT ablation rho={spearmanr(abn[keep], bat_eff[keep])[0]:+.3f}, mean|effect|={np.abs(bat_eff[keep]).mean():.2f} d"
    print(line)
print(f"\nTPC mean |effect| of o2sat: artifact patients {np.abs(tpc_eff[art]).mean():.2f} d vs others {np.abs(tpc_eff[~art]).mean():.2f} d")