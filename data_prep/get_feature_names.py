import pandas as pd

dyn = pd.read_parquet(r"..\YAIB-cohorts-main\YAIB-cohorts-main\data\los\mimic\dyn.parquet")

sensor_cols = [c for c in dyn.columns if c not in ("stay_id", "time")]

print("Number of sensor columns:", len(sensor_cols))
print("\nOrdered feature list (index: name):")
for i, name in enumerate(sensor_cols):
    print(f"{i}: {name}")