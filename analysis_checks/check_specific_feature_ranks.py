import pandas as pd

df = pd.read_csv("attribution_vs_raw_correlation.csv")

features_to_check = ["methb", "crp", "bili_dir"]

print("=== Where do the top-3 raw-LOS-correlated features actually rank in each model? ===\n")
for f in features_to_check:
    row = df[df["feature"] == f].iloc[0]
    print(f"{f}:")
    print(f"  Raw correlation rank: {int(row['raw_rank'])} / 48")
    print(f"  TPC (SHAP) rank:      {int(row['tpc_rank'])} / 48")
    print(f"  BAT (IG) rank:        {int(row['bat_rank_healthybaseline'])} / 48")
    print()