import numpy as np

t  = np.load("tpc_test_targets_fixed.npy").ravel().astype(float)
tp = np.load("tpc_test_predictions_fixed.npy").ravel().astype(float)
bp = np.load("bat_test_predictions.npy").ravel().astype(float)

short = t < 1.0          # stay under 24h, so its input was padded
full  = ~short

print(f"test stays: {len(t)}")
print(f"  under 24h (padded): {short.sum()} ({short.mean()*100:.2f}%)")
print(f"  24h or longer:      {full.sum()} ({full.mean()*100:.2f}%)")
print(f"  shortest stay: {t.min()*24:.2f} hours\n")

def report(mask, label):
    te = tp[mask] - t[mask]
    be = bp[mask] - t[mask]
    print(label)
    for name, e in [("TPC", te), ("BAT", be)]:
        print(f"   {name}  MAE {np.mean(np.abs(e)):.4f}  "
              f"RMSE {np.sqrt(np.mean(e**2)):.4f}  "
              f"mean {np.mean(e):+.4f}  "
              f"D(-3) {np.mean(e <= -3)*100:.2f}%")
    print(f"   BAT MAE advantage: {np.mean(np.abs(te)) - np.mean(np.abs(be)):+.4f} days")

report(np.ones_like(short, bool), "FULL TEST SET (Table II)")
print()
report(full, "EXCLUDING STAYS UNDER 24h")
print()
report(short, "ONLY padded stays under 24h")

print("\n=== are the models reading the padding? ===")
print(f"  mean true LOS on padded stays: {t[short].mean():.3f} d "
      f"({t[short].mean()*24:.1f} h)")
print(f"  mean TPC prediction there:     {tp[short].mean():.3f} d")
print(f"  mean BAT prediction there:     {bp[short].mean():.3f} d")
print(f"  TPC error on padded stays: {np.mean(tp[short]-t[short]):+.3f} d")
print(f"  BAT error on padded stays: {np.mean(bp[short]-t[short]):+.3f} d")