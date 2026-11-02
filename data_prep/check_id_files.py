import pandas as pd

tpc_ids = pd.read_csv("tpc_shap_full_testset_stay_ids.csv")
bat_ids = pd.read_csv("bat_ig_full_testset_stay_ids.csv")

print("TPC ids columns:", tpc_ids.columns.tolist(), "| n =", len(tpc_ids))
print("BAT ids columns:", bat_ids.columns.tolist(), "| n =", len(bat_ids))

print("\nTPC head:")
print(tpc_ids.head())
print("\nBAT head:")
print(bat_ids.head())