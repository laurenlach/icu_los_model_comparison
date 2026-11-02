import numpy as np

feature_names = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun",
    "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2",
    "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
    "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt",
    "po2", "ptt", "resp", "sbp", "temp", "tnt", "urine", "wbc"
]

mask = np.load("bat_mask.npy")            # (51835, 24, 48) - 1=observed, 0=filled/padded
test_rows = np.load("bat_test_rows.npy")
mask_test = mask[test_rows]               # (10367, 24, 48)

features_to_check = ["methb", "crp", "bili_dir"]

# Compute observed-rate for EVERY feature, so we can rank these three
# against the full cohort of 48, not just look at them in isolation
observed_pct_all = mask_test.mean(axis=(0, 1)) * 100  # % of values genuinely observed, per feature
rank_of_missingness = np.argsort(observed_pct_all)     # ascending - most missing first

print("=== Observed-value rate (% of all patient-hours with a real reading) ===\n")
for f in features_to_check:
    idx = feature_names.index(f)
    pct = observed_pct_all[idx]
    rank = int(np.where(rank_of_missingness == idx)[0][0]) + 1  # 1 = most missing feature overall
    print(f"{f}: {pct:.2f}% observed "
          f"(this is the {rank}th most sparsely-measured feature out of 48)")

print(f"\nFor comparison, median observed-rate across all 48 features: "
      f"{np.median(observed_pct_all):.2f}%")
print(f"Most complete feature: {feature_names[np.argmax(observed_pct_all)]} "
      f"({observed_pct_all.max():.2f}% observed)")
print(f"Most sparse feature: {feature_names[np.argmin(observed_pct_all)]} "
      f"({observed_pct_all.min():.2f}% observed)")