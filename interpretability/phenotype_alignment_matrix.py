import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ORGAN_COLS = ["sofa_cardio_comp", "sofa_renal_comp", "sofa_liver_comp",
              "sofa_coag_comp", "sofa_resp_comp"]
MODELS = ["TPC", "BAT"]

phenotype_map = pd.read_csv("cohort_icd9_phenotype_mapping_with_fallback.csv")

# --- Step 1: per-organ, per-model, per-phenotype correlation ---
detailed_rows = []

for organ in ORGAN_COLS:
    for model in MODELS:
        df = pd.read_csv(f"alignment_{organ}_{model}.csv")
        df = df.merge(phenotype_map, on="stay_id", how="inner")

        for phenotype, group in df.groupby("phenotype"):
            if len(group) < 10:
                # Too few patients for a meaningful correlation; skip rather
                # than report a rho computed on a handful of points.
                rho, n = np.nan, len(group)
            else:
                rho, _ = spearmanr(group[organ], group["attribution_mass"])
                n = len(group)

            detailed_rows.append({
                "phenotype": phenotype,
                "model": model,
                "organ_system": organ.replace("sofa_", "").replace("_comp", ""),
                "spearman_rho": rho,
                "n_patients": n,
            })

detailed = pd.DataFrame(detailed_rows)
detailed.to_csv("phenotype_organ_alignment_detailed.csv", index=False)

# --- Step 2: aggregate across organ systems -> one alignment score per phenotype/model ---
aggregate = (
    detailed.groupby(["phenotype", "model"])
    .agg(
        mean_alignment_score=("spearman_rho", "mean"),
        organs_with_data=("spearman_rho", lambda x: x.notna().sum()),
        min_n_across_organs=("n_patients", "min"),
        total_n_phenotype=("n_patients", "max"),
    )
    .reset_index()
)

aggregate.to_csv("phenotype_alignment_aggregate.csv", index=False)

print(f"Detailed (per-organ) rows: {len(detailed)}")
print(f"Aggregate (per-phenotype/model) rows: {len(aggregate)}")
print("\nAggregate preview:")
print(aggregate.sort_values(["phenotype", "model"]).to_string(index=False))