import numpy as np

tpc_shap = np.load("tpc_shap_values_full_testset.npy")
bat_ig = np.load("bat_ig_values_full_testset.npy")

print("TPC SHAP shape:", tpc_shap.shape)
print("BAT IG shape:", bat_ig.shape)