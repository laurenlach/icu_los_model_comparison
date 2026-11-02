import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOT_COLOR = "#e2635f"          # change to match your existing figure if different
INK, MUTED, LINE = "#0b0b0b", "#52514e", "#b9b8b2"
SIZE = lambda n: n / 9.0       # the one place marker scaling is defined

rows = []
for r in csv.DictReader(open("significance_test_by_phenotype_fixed.csv")):
    rows.append((int(r["n_patients"]),
                 float(r["TPC_MAE_fixed"]),
                 float(r["BAT_MAE"])))
print(f"{len(rows)} groups loaded")

LIM = 5.8
fig, ax = plt.subplots(figsize=(3.3, 2.7), dpi=400)

ax.plot([0, LIM], [0, LIM], ls=(0, (4, 3)), color=LINE, lw=1.0, zorder=1)
ax.text(LIM * 0.87, LIM * 0.93, "equal error", rotation=42, fontsize=6.2,
        color=MUTED, ha="center", va="center")

for n, t, b in rows:
    ax.scatter(t, b, s=SIZE(n), color=DOT_COLOR, alpha=.75,
               edgecolor="white", lw=.4, zorder=3)

# size key — same SIZE() as the dots, so the circles cannot lie about the scale
KX, KY, KSTEP = 0.42, 5.10, 0.42
ax.text(KX, KY + 0.40, "stays", fontsize=5.2, color=MUTED,
        ha="center", va="center")
for i, (nn, lab) in enumerate([(100, "100"), (500, "500"), (1000, "1000")]):
    ax.scatter(KX, KY - i * KSTEP, s=SIZE(nn), color=DOT_COLOR, alpha=.45,
               edgecolor="white", lw=.4, zorder=3)
    ax.text(KX + 0.40, KY - i * KSTEP, lab, fontsize=5.0,
            color=MUTED, va="center")

ax.set_xlim(0, LIM); ax.set_ylim(0, LIM)
ax.set_xlabel("TPC mean absolute error (days)", fontsize=7, color=INK)
ax.set_ylabel("BAT mean absolute error (days)", fontsize=7, color=INK)
ax.tick_params(labelsize=6.3, colors=MUTED, length=2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#d6d5d0")

plt.tight_layout(pad=0.2)
plt.savefig("fig_phenotype_mae_sizekey.pdf", bbox_inches="tight")
print("saved fig_phenotype_mae_sizekey.pdf")