import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SOFA_SENSORS = {
    "map":   {"idx": 27, "worse": "low"},
    "bun":   {"idx": 9,  "worse": "high"},
    "crea":  {"idx": 15, "worse": "high"},
    "bili":  {"idx": 6,  "worse": "high"},
    "plt":   {"idx": 39, "worse": "low"},
    "fio2":  {"idx": 19, "worse": "high"},
    "o2sat": {"idx": 35, "worse": "low"},
}

# 1. Load raw (Z-scored) feature values and attribution, for both models
X = np.load("bat_X_scaled.npy")                     # (51835, 24, 48)
test_rows = np.load("bat_test_rows.npy")
X_test = X[test_rows].mean(axis=1)                  # (10367, 48) - mean over window

tpc_shap = np.load("tpc_shap_values_full_testset.npy").squeeze(-1)  # (10367, 48, 24)
bat_ig = np.load("bat_ig_values_full_testset_healthybaseline.npy")  # (10367, 48, 24)

tpc_attr_mean = np.abs(tpc_shap).mean(axis=2)       # (10367, 48) - mean |attribution| over window
bat_attr_mean = np.abs(bat_ig).mean(axis=2)         # (10367, 48)

# 2. For each SOFA sensor, convert raw value into "abnormality in the worse
#    direction" (higher = more abnormal, regardless of whether worse means
#    higher or lower for that specific sensor), then correlate against
#    attribution for that same sensor
results = []
for name, info in SOFA_SENSORS.items():
    idx = info["idx"]
    raw_values = X_test[:, idx]

    # Flip sign for "lower is worse" sensors, so higher always = more abnormal
    abnormality = -raw_values if info["worse"] == "low" else raw_values

    tpc_attr = tpc_attr_mean[:, idx]
    bat_attr = bat_attr_mean[:, idx]

    rho_tpc, p_tpc = spearmanr(abnormality, tpc_attr)
    rho_bat, p_bat = spearmanr(abnormality, bat_attr)

    results.append({
        "sensor": name, "worse_direction": info["worse"],
        "tpc_dose_response_rho": rho_tpc, "tpc_p": p_tpc,
        "bat_dose_response_rho": rho_bat, "bat_p": p_bat,
    })

results_df = pd.DataFrame(results)
print("=== Dose-response: does attribution rise as the sensor gets more abnormal? ===")
print("(Positive rho = attribution rises with severity, as expected for real reasoning)\n")
print(results_df.to_string(index=False))

# 3. Decile view for one sensor per model, to see the SHAPE, not just the
#    correlation strength - is it a smooth rise, or a step, or flat then
#    jumping only at the extreme?
def decile_shape(abnormality, attribution, sensor_name, model_name):
    df = pd.DataFrame({"abnormality": abnormality, "attribution": attribution})
    df["decile"] = pd.qcut(df["abnormality"], 10, labels=False, duplicates="drop")
    shape = df.groupby("decile")["attribution"].mean()
    print(f"\n{model_name} - {sensor_name} - mean attribution by abnormality decile "
          f"(0=least abnormal, 9=most abnormal):")
    print(shape.round(4).to_string())

# Show the shape for creatinine (crea) as a representative, clinically-familiar example
crea_idx = SOFA_SENSORS["crea"]["idx"]
decile_shape(X_test[:, crea_idx], tpc_attr_mean[:, crea_idx], "crea", "TPC")
decile_shape(X_test[:, crea_idx], bat_attr_mean[:, crea_idx], "crea", "BAT")

results_df.to_csv("dose_response_check.csv", index=False)
print("\nSaved: dose_response_check.csv")