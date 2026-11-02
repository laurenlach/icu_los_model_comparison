import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

order = pd.read_csv("bat_stay_id_order.csv")
test_rows = np.load("bat_test_rows.npy")
t  = np.load("tpc_test_targets_fixed.npy").ravel().astype(float)
tp = np.load("tpc_test_predictions_fixed.npy").ravel().astype(float)
bp = np.load("bat_test_predictions.npy").ravel().astype(float)

df = pd.DataFrame({
    "stay_id": order.stay_id.values[test_rows],
    "adv": np.abs(tp - t) - np.abs(bp - t),   # per stay, + means BAT better
})
src = pd.read_csv("icustay_dbsource.csv").rename(
    columns={"icustay_id": "stay_id"})
df = df.merge(src, on="stay_id", how="left")

cv = df.loc[df.dbsource == "carevue",    "adv"]
mv = df.loc[df.dbsource == "metavision", "adv"]

u, p = mannwhitneyu(cv, mv)
print(f"carevue    n={len(cv)}  mean advantage {cv.mean():+.4f}")
print(f"metavision n={len(mv)}  mean advantage {mv.mean():+.4f}")
print(f"difference {cv.mean()-mv.mean():+.4f} days,  p = {p:.4f}")
print("\nsame test on BAT's advantage as a fraction of TPC's error:")
for nm, g in [("carevue", "carevue"), ("metavision", "metavision")]:
    s = df[df.dbsource == g]
    print(f"  {nm}: {s.adv.mean() / np.abs(tp - t)[df.dbsource.values == g].mean()*100:.2f}%")