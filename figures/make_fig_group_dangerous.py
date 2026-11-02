import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOT_COLOR = "#e2635f"
INK, MUTED, LINE = "#0b0b0b", "#52514e", "#b9b8b2"
SIZE = lambda n: n / 9.0

tpc_error = np.load("tpc_error_fixed.npy")
bat_error = np.load("bat_error.npy")
test_rows = np.load("bat_test_rows.npy")

stay_id_order = pd.read_csv("bat_stay_id_order.csv")
phenotype_map = pd.read_csv("cohort_icd9_phenotype_mapping_with_fallback.csv")

df = pd.DataFrame({
    "stay_id": stay_id_order["stay_id"].iloc[test_rows].values,
    "tpc_err": tpc_error,
    "bat_err": bat_error,
}).merge(phenotype_map, on="stay_id", how="left").dropna(subset=["phenotype"])

g = df.groupby("phenotype")
summary = pd.DataFrame({
    "n": g.size(),
    "tpc_dang": g["tpc_err"].apply(lambda e: 100 * (e <= -3).mean()),
    "bat_dang": g["bat_err"].apply(lambda e: 100 * (e <= -3).mean()),
}).reset_index()
summary.to_csv("group_dangerous_values.csv", index=False)
print(summary.to_string(index=False))
print(f"\nBAT higher in {(summary.bat_dang > summary.tpc_dang).sum()} of {len(summary)} groups")

LIM = 33
fig, ax = plt.subplots(figsize=(3.3, 2.7), dpi=400)

ax.plot([0, LIM], [0, LIM], ls=(0, (4, 3)), color=LINE, lw=1.0, zorder=1)
ax.text(LIM * 0.87, LIM * 0.93, "equal rate", rotation=42, fontsize=6.2,
        color=MUTED, ha="center", va="center")

for _, r in summary.iterrows():
    ax.scatter(r["tpc_dang"], r["bat_dang"], s=SIZE(r["n"]), color=DOT_COLOR,
               alpha=.75, edgecolor="white", lw=.4, zorder=3)

KX, KY, KSTEP = 2.4, 29.0, 2.4
ax.text(KX, KY + 2.3, "stays", fontsize=5.2, color=MUTED,
        ha="center", va="center")
for i, (nn, lab) in enumerate([(100, "100"), (500, "500"), (1000, "1000")]):
    ax.scatter(KX, KY - i * KSTEP, s=SIZE(nn), color=DOT_COLOR, alpha=.45,
               edgecolor="white", lw=.4, zorder=3)
    ax.text(KX + 2.3, KY - i * KSTEP, lab, fontsize=5.0,
            color=MUTED, va="center")

ax.set_xlim(0, LIM); ax.set_ylim(0, LIM)
ax.set_xlabel("TPC dangerous-case rate (\\%)".replace("\\", ""),
              fontsize=7, color=INK)
ax.set_ylabel("BAT dangerous-case rate (%)", fontsize=7, color=INK)
ax.tick_params(labelsize=6.3, colors=MUTED, length=2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#d6d5d0")

plt.tight_layout(pad=0.2)
plt.savefig("fig_group_dangerous.pdf", bbox_inches="tight")
print("saved fig_group_dangerous.pdf")