import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

tp = np.load("tpc_test_predictions_fixed.npy").ravel().astype(float)
tt = np.load("tpc_test_targets_fixed.npy").ravel().astype(float)
bp = np.load("bat_test_predictions.npy").ravel().astype(float)

TPC_C, BAT_C, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"

# Row 8723: both models accurate. Row 7300: BAT dangerous, TPC safe.
cases = [(8723, "Both models accurate"), (7300, "BAT under-predicts")]

fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.35), dpi=400)

for ax, (row, title) in zip(axes, cases):
    true = tt[row]
    vals = [tp[row], bp[row]]

    ax.barh([1, 0], vals, height=0.5, color=[TPC_C, BAT_C], zorder=3)
    ax.axvline(true, color=INK, lw=1.1, zorder=4)
    ax.axvline(true - 3, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=4)

    for y, v in zip([1, 0], vals):
        ax.text(v + 0.25, y, f"{v:.1f}", va="center", ha="left",
                fontsize=5.4, color=INK, zorder=5)

    ax.set_yticks([1, 0])
    ax.set_yticklabels(["TPC", "BAT"], fontsize=6)
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.5, 1.5)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_title(f"{title}, true stay {true:.1f} days",
                 fontsize=5.8, color=INK, pad=3)
    ax.tick_params(axis="x", labelsize=5.4, length=2, colors=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("days", fontsize=5.6, color=MUTED, labelpad=1)

    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d6d5d0")
    ax.grid(axis="x", color="#eceae5", lw=0.5, zorder=0)

plt.tight_layout(pad=0.25)
plt.savefig("fig_cases.pdf", bbox_inches="tight")

print(f"Case 1 row 8723: true {tt[8723]:.2f}  TPC {tp[8723]:.2f}  BAT {bp[8723]:.2f}")
print(f"Case 2 row 7300: true {tt[7300]:.2f}  TPC {tp[7300]:.2f}  BAT {bp[7300]:.2f}")
print("saved fig_cases.pdf")