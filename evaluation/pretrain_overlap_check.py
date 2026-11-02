import numpy as np, pandas as pd

order = pd.read_csv("bat_stay_id_order.csv")
test_rows = np.load("bat_test_rows.npy")

t  = np.load("tpc_test_targets_fixed.npy").ravel().astype(float)
tp = np.load("tpc_test_predictions_fixed.npy").ravel().astype(float)
bp = np.load("bat_test_predictions.npy").ravel().astype(float)

df = pd.DataFrame({
    "stay_id": order.stay_id.values[test_rows],
    "tpc_e": tp - t,
    "bat_e": bp - t,
})

src = pd.read_csv("icustay_dbsource.csv").rename(
    columns={"icustay_id": "stay_id"})
df = df.merge(src, on="stay_id", how="left")

print("merged:", len(df), " unmatched:", df.dbsource.isna().sum())
print(df.dbsource.value_counts(dropna=False), "\n")

for name, g in df.groupby("dbsource"):
    tm = g.tpc_e.abs().mean()
    bm = g.bat_e.abs().mean()
    print(f"{name}  (n={len(g)})")
    print(f"   TPC MAE {tm:.4f}   D(-3) {(g.tpc_e<=-3).mean()*100:.2f}%")
    print(f"   BAT MAE {bm:.4f}   D(-3) {(g.bat_e<=-3).mean()*100:.2f}%")
    print(f"   BAT advantage: {tm-bm:+.4f} days\n")