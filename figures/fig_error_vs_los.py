"""
fig_error_vs_los_zoom.py
Mean signed prediction error against true length of stay, TPC vs BAT.
Panel (a) is the full range. Panel (b) zooms on the range where the two
curves overlap.

Run from the folder holding the .npy files:
    python fig_error_vs_los_zoom.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Load the saved test-set predictions and targets.
#    TPC uses the FIXED model files. Do NOT use the _200epoch (pre-fix) ones.
# ---------------------------------------------------------------------------
tpc_pred = np.load("tpc_test_predictions_fixed.npy").ravel()
tpc_true = np.load("tpc_test_targets_fixed.npy").ravel()
bat_pred = np.load("bat_test_predictions.npy").ravel()
bat_true = np.load("bat_test_targets.npy").ravel()

assert len(tpc_pred) == len(bat_pred), "Test sets differ in size - stop and check."
assert np.allclose(tpc_true, bat_true), "Targets differ - models are not on the same split."

tpc_err = tpc_pred - tpc_true
bat_err = bat_pred - bat_true
true_los = tpc_true

# ---------------------------------------------------------------------------
# 2. Bucket by true length of stay, average the signed error in each.
# ---------------------------------------------------------------------------
edges = [0, 3, 5, 7, 10, 14, 20, np.inf]
labels = ["0-3", "3-5", "5-7", "7-10", "10-14", "14-20", "20+"]

tpc_means, bat_means, counts = [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (true_los >= lo) & (true_los < hi)
    tpc_means.append(tpc_err[sel].mean())
    bat_means.append(bat_err[sel].mean())
    counts.append(int(sel.sum()))

print(f"{'LOS (days)':>11} {'n':>6} {'TPC mean err':>13} {'BAT mean err':>13} {'gap':>7}")
for lab, n, t, b in zip(labels, counts, tpc_means, bat_means):
    print(f"{lab:>11} {n:>6} {t:>13.2f} {b:>13.2f} {abs(t - b):>7.2f}")

x = np.arange(len(labels))
tpc_means = np.array(tpc_means)
bat_means = np.array(bat_means)

ZOOM = slice(2, 6)   # 5-7, 7-10, 10-14, 14-20: where the curves overlap

# ---------------------------------------------------------------------------
# 3. Two stacked panels, sized for one IEEE column.
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, (ax, axz) = plt.subplots(
    2, 1, figsize=(3.4, 4.2), gridspec_kw={"height_ratios": [1.35, 1]}
)

# --- Panel (a): full range -------------------------------------------------
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(-3, color="grey", linewidth=0.8, linestyle=":")
ax.text(-0.15, -3.8, "dangerous-case threshold", fontsize=6,
        color="grey", ha="left", va="top")

ax.plot(x, tpc_means, marker="o", markersize=4, linewidth=1.4,
        color="#1f77b4", label="TPC")
ax.plot(x, bat_means, marker="s", markersize=4, linewidth=1.4,
        color="#d62728", label="BAT")

ax.axvspan(x[ZOOM][0] - 0.4, x[ZOOM][-1] + 0.4, color="grey", alpha=0.10,
           zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Mean signed error (days)")
ax.legend(frameon=False, loc="lower left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("(a) full range", fontsize=7, loc="left", pad=4)

# --- Panel (b): zoom on the overlapping range ------------------------------
zx = x[ZOOM]
axz.axhline(-3, color="grey", linewidth=0.8, linestyle=":")
axz.plot(zx, tpc_means[ZOOM], marker="o", markersize=4, linewidth=1.4,
         color="#1f77b4", label="TPC")
axz.plot(zx, bat_means[ZOOM], marker="s", markersize=4, linewidth=1.4,
         color="#d62728", label="BAT")

axz.set_xticks(zx)
axz.set_xticklabels([labels[i] for i in range(*ZOOM.indices(len(labels)))])
axz.set_xlim(zx[0] - 0.4, zx[-1] + 0.4)
axz.set_xlabel("True length of stay (days)")
axz.set_ylabel("Mean signed error (days)")
axz.spines["top"].set_visible(False)
axz.spines["right"].set_visible(False)
axz.set_facecolor("#f7f7f7")
axz.set_title("(b) overlapping range", fontsize=7, loc="left", pad=4)

fig.tight_layout()
fig.savefig("fig_error_vs_los_zoom.pdf", bbox_inches="tight")
fig.savefig("fig_error_vs_los_zoom.png", dpi=300, bbox_inches="tight")
print("\nSaved: fig_error_vs_los_zoom.pdf and fig_error_vs_los_zoom.png")