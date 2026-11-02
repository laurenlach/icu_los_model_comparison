import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ORGAN_COLS = ["sofa_cardio_comp", "sofa_renal_comp", "sofa_liver_comp",
              "sofa_coag_comp", "sofa_resp_comp"]

phenotype_map = pd.read_csv("cohort_icd9_phenotype_mapping_with_fallback.csv")

def alignment_filename(organ, model):
    if model == "BAT":
        return f"alignment_{organ}_BAT_healthybaseline.csv"
    return f"alignment_{organ}_TPC_fixed.csv"

detailed_rows = []
for organ in ORGAN_COLS:
    for model in ["TPC", "BAT"]:
        df = pd.read_csv(alignment_filename(organ, model))
        df = df.merge(phenotype_map, on="stay_id", how="inner")
        for phenotype, group in df.groupby("phenotype"):
            if len(group) < 10:
                rho, n = np.nan, len(group)
            else:
                rho, _ = spearmanr(group[organ], group["attribution_mass"])
                n = len(group)
            detailed_rows.append({
                "phenotype": phenotype, "model": model,
                "organ_system": organ.replace("sofa_", "").replace("_comp", ""),
                "spearman_rho": rho, "n_patients": n,
            })

detailed = pd.DataFrame(detailed_rows)
aggregate = (
    detailed.groupby(["phenotype", "model"])
    .agg(mean_alignment_score=("spearman_rho", "mean"),
         organs_with_data=("spearman_rho", lambda x: x.notna().sum()),
         min_n_across_organs=("n_patients", "min"))
    .reset_index()
)
print(aggregate.sort_values(["phenotype", "model"]).to_string(index=False))
aggregate.to_csv("phenotype_alignment_aggregate_fixed.csv", index=False)
print("\nSaved: phenotype_alignment_aggregate_fixed.csv")