import pandas as pd

los = pd.read_csv("los_per_stay.csv")
sta = pd.read_parquet(r"..\YAIB-cohorts-main\YAIB-cohorts-main\data\los\mimic\sta.parquet")

merged = sta.merge(los, on="stay_id", how="left")

print(merged.shape)
print("Missing LOS values:", merged["los"].isna().sum())
print(merged.head())

merged.to_parquet("bat_sta_with_los.parquet", index=False)