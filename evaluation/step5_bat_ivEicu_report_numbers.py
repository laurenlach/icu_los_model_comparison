"""
Step 5 of the BAT re-run: recompute every BAT number in the report (Draft 4)
for the NEW BAT (MIMIC-IV + eICU checkpoint), using the same formulas as the
report's methodology. TPC is computed alongside from its saved SHAP file, so you
can check that TPC's column still matches the report exactly. That proves the
formulas are the same ones the report used.

Run from your YAIB folder:
    python step5_bat_ivEicu_report_numbers.py

Files needed in the YAIB folder:
    bat_test_predictions_ivEicu.npy, tpc_test_predictions_fixed.npy, tpc_test_targets_fixed.npy,
    bat_test_rows.npy, bat_stay_id_order.csv,
    bat_ig_values_full_testset_healthybaseline_ivEicu.npy,
    bat_ig_convergence_deltas_healthybaseline_ivEicu.npy,
    bat_ig_full_testset_stay_ids_healthybaseline_ivEicu.csv,
    tpc_shap_values_full_testset_fixed.npy,
    sofa_24h_scores.csv, cohort_icd9_phenotype_mapping_with_fallback.csv

Nothing is overwritten. Every file it saves ends in _ivEicu.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TAG = "_ivEicu"
NAMES = ["alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun",
         "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2",
         "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
         "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt",
         "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine", "wbc"]
SOFA_IDX = [27, 9, 15, 6, 39, 19, 35]            # map, bun, crea, bili, plt, fio2, o2sat
ORGANS = {"sofa_cardio_comp": [27], "sofa_renal_comp": [9, 15], "sofa_liver_comp": [6],
          "sofa_coag_comp": [39], "sofa_resp_comp": [19, 35]}
ORGAN_LABEL = {"sofa_cardio_comp": "Cardiovascular", "sofa_renal_comp": "Renal",
               "sofa_liver_comp": "Liver", "sofa_coag_comp": "Coagulation",
               "sofa_resp_comp": "Respiratory"}

# ---------------------------------------------------------------------------
# 0. Load and check alignment. Why: every number below pairs rows across files.
# ---------------------------------------------------------------------------
bat_p = np.load(f"bat_test_predictions{TAG}.npy").ravel().astype(np.float64)
tpc_p = np.load("tpc_test_predictions_fixed.npy").ravel().astype(np.float64)
y = np.load("tpc_test_targets_fixed.npy").ravel().astype(np.float64)
test_rows = np.load("bat_test_rows.npy")
stay_ids = pd.read_csv("bat_stay_id_order.csv")["stay_id"].values[test_rows]
ig_ids = pd.read_csv(f"bat_ig_full_testset_stay_ids_healthybaseline{TAG}.csv")["stay_id"].values

bat_imp = np.abs(np.load(f"bat_ig_values_full_testset_healthybaseline{TAG}.npy").astype(np.float64))   # (n, 48, 24)
tpc_imp = np.abs(np.load("tpc_shap_values_full_testset_fixed.npy").squeeze(-1).astype(np.float64))
deltas = np.abs(np.load(f"bat_ig_convergence_deltas_healthybaseline{TAG}.npy"))
flagged = deltas > 0.1

print("=== Alignment checks (all must say True) ===")
checks = {
    "10367 stays everywhere": len(bat_p) == len(tpc_p) == len(y) == len(stay_ids) == len(bat_imp) == len(tpc_imp) == 10367,
    "IG stay IDs match bat_stay_id_order at the test rows": np.array_equal(ig_ids, stay_ids),
    "Importance arrays are (stays, 48 variables, 24 hours)": bat_imp.shape == tpc_imp.shape == (10367, 48, 24),
    "No missing values in importance arrays": bool(np.isfinite(bat_imp).all() and np.isfinite(tpc_imp).all()),
}
for k, v in checks.items():
    print(f"  {k}: {v}")
if not all(checks.values()):
    raise SystemExit("STOP: an alignment check failed. Paste this output to Claude.")

e_bat, e_tpc = bat_p - y, tpc_p - y

# ---------------------------------------------------------------------------
# 1. Section IV-B text and Discussion: paired comparison
# ---------------------------------------------------------------------------
print("\n=== Paired comparison (Section IV-B, Discussion) ===")
print(f"  BAT has the lower absolute error on {int((np.abs(e_bat) < np.abs(e_tpc)).sum())} of 10367 stays   [report: 6 465]")
print(f"  MAE advantage:  {np.abs(e_tpc).mean() - np.abs(e_bat).mean():.3f} days   [report: 0.454]")
print(f"  RMSE advantage: {np.sqrt((e_tpc**2).mean()) - np.sqrt((e_bat**2).mean()):.3f} days   [report: 0.146]")

# ---------------------------------------------------------------------------
# 2. Table IV: SOFA importance share (Equation 6) and temporal entropy
# ---------------------------------------------------------------------------
def sofa_share(imp):
    pooled = 100 * imp[:, SOFA_IDX].sum() / imp.sum()
    per = 100 * imp[:, SOFA_IDX].sum(axis=(1, 2)) / imp.sum(axis=(1, 2))
    return pooled, per.mean(), per.std()

def temporal_entropy(imp):
    h = imp.sum(axis=(0, 1))            # importance per hour, summed over stays and variables
    p = h / h.sum()
    p = p[p > 0]
    return -(p * np.log(p)).sum() / np.log(24)

print("\n=== Table IV ===")
print(f"  {'':<26}{'Overall':>10}{'Per stay':>18}{'Temporal entropy':>20}")
for name, imp, rep in [("TPC  [report: 18.95%, 19.91% (5.24), 0.637]", tpc_imp, None),
                       ("BAT new", bat_imp, None),
                       ("BAT new, flagged excluded", bat_imp[~flagged], None)]:
    pooled, mean, sd = sofa_share(imp)
    print(f"  {name}\n  {'':<26}{pooled:>9.2f}%{mean:>10.2f}% ({sd:.2f}){temporal_entropy(imp):>20.3f}")
print(f"  BAT new is {sofa_share(bat_imp)[0] - 100 * 7 / 48:.1f} points above the 14.58% chance baseline   "
      f"[report: 21.3 for the original BAT]")
print(f"  TPC hours with exactly zero importance: {int((tpc_imp.sum(axis=(0, 1)) == 0).sum())}   [report: 9, hours 0 to 8]")
print(f"  Stays with IG convergence error above 0.1: {int(flagged.sum())} ({100 * flagged.mean():.2f}%)")

# ---------------------------------------------------------------------------
# 3. Table V: Spearman alignment per organ, plus the flagged-stays check
# ---------------------------------------------------------------------------
sofa = pd.read_csv("sofa_24h_scores.csv").rename(columns={"icustay_id": "stay_id"})
print("\n=== Table V ===")
print(f"  {'Organ':<16}{'Stays':>8}{'TPC':>9}{'BAT new':>10}{'BAT excl. flagged':>20}")
table5 = []
align_files = {}
for organ, idx in ORGANS.items():
    row = {"organ": ORGAN_LABEL[organ]}
    for model, imp in [("TPC", tpc_imp), ("BAT", bat_imp)]:
        df = pd.DataFrame({"stay_id": stay_ids, "attribution_mass": imp[:, idx, :].sum(axis=(1, 2)), "flagged": flagged})
        df = df.merge(sofa[["stay_id", organ]], on="stay_id", how="inner").dropna(subset=[organ])
        row["stays"] = len(df)
        row[model] = spearmanr(df[organ], df["attribution_mass"])[0]
        if model == "BAT":
            k = ~df["flagged"]
            row["BAT_excl"] = spearmanr(df.loc[k, organ], df.loc[k, "attribution_mass"])[0]
        align_files[(organ, model)] = df[["stay_id", organ, "attribution_mass"]]
    table5.append(row)
    print(f"  {row['organ']:<16}{row['stays']:>8}{row['TPC']:>9.3f}{row['BAT']:>10.3f}{row['BAT_excl']:>20.3f}")
print("  [report TPC: -0.343, 0.181, 0.740, 0.167, 0.058]")
for organ in ORGANS:
    align_files[(organ, "BAT")].to_csv(f"alignment_{organ}_BAT_healthybaseline{TAG}.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Comparative summary: mean alignment score per disease group
#    (same method as phenotype_alignment_matrix_v2.py: Spearman per organ
#     within the group if n >= 10, then the mean over organs)
# ---------------------------------------------------------------------------
pheno = pd.read_csv("cohort_icd9_phenotype_mapping_with_fallback.csv")
rows = []
for (organ, model), df in align_files.items():
    for ph, g in df.merge(pheno, on="stay_id", how="inner").groupby("phenotype"):
        rho = spearmanr(g[organ], g["attribution_mass"])[0] if len(g) >= 10 else np.nan
        rows.append({"phenotype": ph, "model": model, "organ": organ, "rho": rho})
grp = pd.DataFrame(rows).groupby(["phenotype", "model"])["rho"].mean().unstack()
grp["BAT_higher"] = grp["BAT"] > grp["TPC"]
print("\n=== Comparative summary: mean alignment score per disease group ===")
print(grp.round(4).to_string())
print(f"\n  BAT has the higher mean alignment score in {int(grp['BAT_higher'].sum())}/{len(grp)} groups   [report: 25/25]")
grp.to_csv(f"phenotype_alignment_aggregate{TAG}.csv")

# ---------------------------------------------------------------------------
# 5. Figure 3 and Section IV-C: mean signed error by true length of stay
# ---------------------------------------------------------------------------
edges = [0, 3, 5, 7, 10, 14, 20, np.inf]
labels = ["0-3", "3-5", "5-7", "7-10", "10-14", "14-20", "20+"]
bins = pd.cut(y, edges, labels=labels, right=False)
fig3 = pd.DataFrame({"bin": bins, "TPC": e_tpc, "BAT": e_bat}).groupby("bin", observed=False).agg(
    n=("TPC", "size"), TPC_mean_error=("TPC", "mean"), BAT_mean_error=("BAT", "mean"))
fig3["gap_TPC_minus_BAT"] = fig3["TPC_mean_error"] - fig3["BAT_mean_error"]
print("\n=== Figure 3: mean signed error by true length of stay ===")
print(fig3.round(2).to_string())
print("  [report: 0-3 n=6574, TPC +1.79, original BAT +0.82, gap 0.97; 20+ n=293, TPC -24.30, original BAT -24.20; 14-20 gap 0.04]")
fig3.to_csv(f"figure3_error_by_true_los{TAG}.csv")

# ---------------------------------------------------------------------------
# 6. Figure 4: the two illustrative stays, located from the report's values
#    (TPC 7.5 / true 7.6, and TPC 10.2 / true 9.2), then the new BAT's prediction
# ---------------------------------------------------------------------------
print("\n=== Figure 4: the two illustrative stays ===")
for label, true_v, tpc_v in [("Both accurate", 7.6, 7.5), ("BAT under-predicts", 9.2, 10.2)]:
    cand = np.where((np.abs(y - true_v) < 0.05) & (np.abs(tpc_p - tpc_v) < 0.05))[0]
    for i in cand:
        print(f"  {label}: stay {stay_ids[i]}  true {y[i]:.2f}  TPC {tpc_p[i]:.2f}  BAT new {bat_p[i]:.2f}  "
              f"(BAT error {e_bat[i]:+.2f}, dangerous: {e_bat[i] <= -3})")
    if len(cand) == 0:
        print(f"  {label}: no stay matched, tell Claude")

# ---------------------------------------------------------------------------
# 7. Appendix C: mean importance and rank for all 48 variables
#    (mean of absolute importance over stays and hours)
# ---------------------------------------------------------------------------
app = pd.DataFrame({"variable": NAMES,
                    "TPC": tpc_imp.mean(axis=(0, 2)), "BAT_new": bat_imp.mean(axis=(0, 2))})
app["TPC_rank"] = app["TPC"].rank(ascending=False, method="first").astype(int)
app["BAT_new_rank"] = app["BAT_new"].rank(ascending=False, method="first").astype(int)
app = app.sort_values("TPC_rank")
print("\n=== Appendix C (ordered by TPC rank; TPC column must match the report) ===")
print(app.round(4).to_string(index=False))
print("  [report TPC top 3: temp 0.1204, ph 0.1079, map 0.0687]")
app.to_csv(f"appendix_c_feature_importance{TAG}.csv", index=False)
print("\n  New BAT top 10:", ", ".join(app.sort_values("BAT_new_rank")["variable"].head(10)))

print(f"\nSaved: alignment_*_BAT_healthybaseline{TAG}.csv (5 files), phenotype_alignment_aggregate{TAG}.csv, "
      f"figure3_error_by_true_los{TAG}.csv, appendix_c_feature_importance{TAG}.csv")
print("Done.")