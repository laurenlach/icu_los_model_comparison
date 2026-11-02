
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = "_ivEicu"
TPC_COLOUR, BAT_COLOUR = "#1f77b4", "#d62728"


y = np.load("tpc_test_targets_fixed.npy").ravel().astype(np.float64)
tpc = np.load("tpc_test_predictions_fixed.npy").ravel().astype(np.float64)
bat = np.load(f"bat_test_predictions{TAG}.npy").ravel().astype(np.float64)
stay_ids = pd.read_csv("bat_stay_id_order.csv")["stay_id"].values[np.load("bat_test_rows.npy")]
assert len(y) == len(tpc) == len(bat) == len(stay_ids) == 10367, "STOP: wrong number of stays"
e_tpc, e_bat = tpc - y, bat - y

def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def size_key(ax):
    # "stays" key in the top-left, same marker scale as the points (area = stays / 8)
    handles = [ax.scatter([], [], s=n / 8, color=BAT_COLOUR, alpha=0.4, edgecolor="none") for n in (100, 500, 1000)]
    ax.legend(handles, ["100", "500", "1000"], title="stays", loc="upper left", frameon=False,
              fontsize=5, title_fontsize=5, labelspacing=0.9, borderpad=0.6, handletextpad=1.2)


# MAE per disease group, BAT against TPC
df = pd.DataFrame({"stay_id": stay_ids, "tpc": np.abs(e_tpc), "bat": np.abs(e_bat)})
df = df.merge(pd.read_csv("cohort_icd9_phenotype_mapping_with_fallback.csv"),
              on="stay_id", how="inner").dropna(subset=["phenotype"])
g = df.groupby("phenotype").agg(n=("tpc", "size"), tpc=("tpc", "mean"), bat=("bat", "mean"))
print(f"Figure 2: {len(g)} groups, {int(g['n'].sum())} stays, BAT lower MAE in {int((g['bat'] < g['tpc']).sum())}")

plt.rcParams.update({"font.size": 7, "font.family": "sans-serif"})
fig, ax = plt.subplots(figsize=(3.4, 3.0))
lim = 5.5
ax.plot([0, lim], [0, lim], linestyle="--", color="#aaaaaa", linewidth=1.0, zorder=1)
ax.text(5.0, 5.1, "equal error", rotation=45, fontsize=6, color="grey", ha="center", va="bottom")
ax.scatter(g["tpc"], g["bat"], s=g["n"] / 8, color=BAT_COLOUR, alpha=0.6,
           edgecolor="white", linewidth=0.5, zorder=2)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("TPC mean absolute error (days)")
ax.set_ylabel("BAT mean absolute error (days)")
size_key(ax)
clean(ax)
fig.tight_layout()
fig.savefig(f"fig2_mae_by_group{TAG}.pdf", bbox_inches="tight")
fig.savefig(f"fig2_mae_by_group{TAG}.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# dangerous-case rate D(-3) per disease group, BAT against TPC.
# Same style as the MAE panel. Points ABOVE the line are groups where BAT
# under-predicts by 3+ days more often than TPC.

dd = pd.DataFrame({"stay_id": stay_ids, "tpc": e_tpc <= -3, "bat": e_bat <= -3})
dd = dd.merge(pd.read_csv("cohort_icd9_phenotype_mapping_with_fallback.csv"),
              on="stay_id", how="inner").dropna(subset=["phenotype"])
gd = dd.groupby("phenotype").agg(n=("tpc", "size"), tpc=("tpc", "mean"), bat=("bat", "mean"))
gd[["tpc", "bat"]] *= 100
print(f"Figure 2(b): TPC lower rate in {int((gd['tpc'] < gd['bat']).sum())}, "
      f"equal in {int((gd['tpc'] == gd['bat']).sum())}, BAT lower in {int((gd['bat'] < gd['tpc']).sum())}")

fig, ax = plt.subplots(figsize=(3.4, 3.0))
lim = 32
ax.plot([0, lim], [0, lim], linestyle="--", color="#aaaaaa", linewidth=1.0, zorder=1)
ax.text(29.5, 30.3, "equal rate", rotation=45, fontsize=6, color="grey", ha="center", va="bottom")
ax.scatter(gd["tpc"], gd["bat"], s=gd["n"] / 8, color=BAT_COLOUR, alpha=0.6,
           edgecolor="white", linewidth=0.5, zorder=2)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("TPC dangerous-case rate (%)")
ax.set_ylabel("BAT dangerous-case rate (%)")
size_key(ax)
clean(ax)
fig.tight_layout()
fig.savefig(f"fig2b_dangerous_by_group{TAG}.pdf", bbox_inches="tight")
fig.savefig(f"fig2b_dangerous_by_group{TAG}.png", dpi=300, bbox_inches="tight")
plt.close(fig)

#  mean signed error against true length of stay

edges = [0, 3, 5, 7, 10, 14, 20, np.inf]
labels = ["0-3", "3-5", "5-7", "7-10", "10-14", "14-20", "20+"]
tpc_m, bat_m = [], []
print(f"\nFigure 3:\n{'LOS':>7} {'n':>6} {'TPC':>8} {'BAT':>8}")
for lab, lo, hi in zip(labels, edges[:-1], edges[1:]):
    sel = (y >= lo) & (y < hi)
    tpc_m.append(e_tpc[sel].mean()); bat_m.append(e_bat[sel].mean())
    print(f"{lab:>7} {int(sel.sum()):>6} {tpc_m[-1]:>8.2f} {bat_m[-1]:>8.2f}")
x = np.arange(len(labels))

plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(-3, color="grey", linewidth=0.8, linestyle=":")
ax.text(-0.15, -3.5, "dangerous-case threshold", fontsize=6, color="grey", ha="left", va="top")
ax.plot(x, tpc_m, marker="o", markersize=4, linewidth=1.4, color=TPC_COLOUR, label="TPC")
ax.plot(x, bat_m, marker="s", markersize=4, linewidth=1.4, color=BAT_COLOUR, label="BAT")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel("True length of stay (days)")
ax.set_ylabel("Mean signed error (days)")
ax.legend(frameon=False, loc="lower left")
clean(ax)
fig.tight_layout()
fig.savefig(f"fig3_error_vs_los{TAG}.pdf", bbox_inches="tight")
fig.savefig(f"fig3_error_vs_los{TAG}.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# illustrative stays 

examples = [(284464, "Both models accurate"), (271079, "BAT under-predicts")]
plt.rcParams.update({"font.family": "sans-serif", "font.size": 5.5})
fig, axes = plt.subplots(1, 2, figsize=(4.0, 1.5))
print("\nFigure 4:")
for ax, (sid, title) in zip(axes, examples):
    i = int(np.where(stay_ids == sid)[0][0])
    t, p_tpc, p_bat = y[i], tpc[i], bat[i]
    print(f"  stay {sid}: true {t:.2f}  TPC {p_tpc:.2f}  BAT {p_bat:.2f}  BAT error {p_bat - t:+.2f}")
    ax.barh([1, 0], [p_tpc, p_bat], color=["#2f7fd6", "#ef6c3c"], height=0.55)
    for yy, v in [(1, p_tpc), (0, p_bat)]:
        ax.text(max(v, t) + 0.2, yy, f"{v:.1f}", va="center", fontsize=5)   # label clear of the true-stay line
    ax.axvline(t, color="black", linewidth=1.2)
    ax.axvline(t - 3, color="black", linewidth=0.9, linestyle="--")
    ax.set_yticks([1, 0]); ax.set_yticklabels(["TPC", "BAT"])
    ax.set_xlim(0, max(11, p_tpc + 1.5, p_bat + 1.5))
    ax.set_xlabel("days")
    ax.set_title(f"{title}, true stay {t:.1f} days", fontsize=5.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#dddddd", linewidth=0.5); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(f"fig4_example_stays{TAG}.pdf", bbox_inches="tight")
fig.savefig(f"fig4_example_stays{TAG}.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved: fig2_mae_by_group{TAG}, fig2b_dangerous_by_group{TAG}, fig3_error_vs_los{TAG}, fig4_example_stays{TAG} (.pdf and .png)")