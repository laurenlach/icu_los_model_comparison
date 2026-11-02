"""
fig_phenotype_mae.py
Per-phenotype mean absolute error, BAT against TPC.
Run from the folder holding reliability_matrix_fixed.csv:
    python fig_phenotype_mae.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv("reliability_matrix_fixed.csv")
p = d.pivot(index="phenotype", columns="model")

tpc = p[("mae", "TPC")].values
bat = p[("mae", "BAT")].values
n = p[("n_patients", "BAT")].values

print(f"Phenotypes: {len(tpc)}")
print(f"BAT lower MAE in {(bat < tpc).sum()}/{len(tpc)} phenotypes")

# ---------------------------------------------------------------------------
# Scatter against the line of equality. A point below the diagonal means BAT
# had the lower error for that diagnosis group. Marker area scales with the
# number of patients, so small, noisy groups are visibly small.
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(3.4, 3.0))

lims = [0, max(tpc.max(), bat.max()) * 1.08]
ax.plot(lims, lims, color="grey", linewidth=0.8, linestyle="--", zorder=1)
ax.text(lims[1] * 0.97, lims[1] * 0.97, "equal error", fontsize=6,
        color="grey", ha="right", va="bottom", rotation=45,
        rotation_mode="anchor")

ax.scatter(tpc, bat, s=np.sqrt(n) * 2.2, color="#d62728", alpha=0.65,
           edgecolors="white", linewidths=0.5, zorder=2)

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal")
ax.set_xlabel("TPC mean absolute error (days)")
ax.set_ylabel("BAT mean absolute error (days)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("fig_phenotype_mae.pdf", bbox_inches="tight")
fig.savefig("fig_phenotype_mae.png", dpi=300, bbox_inches="tight")
print("Saved: fig_phenotype_mae.pdf and fig_phenotype_mae.png")