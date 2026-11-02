import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# 1. Load Harutyunyan et al.'s phenotype definitions
# ---------------------------------------------------------------------------
with open("hcup_ccs_2015_definitions.yaml", "r") as f:
    phenotype_defs = yaml.safe_load(f)

print(f"Total phenotype categories in file: {len(phenotype_defs)}")

# ---------------------------------------------------------------------------
# 2. Keep only phenotypes marked use_in_benchmark: True - this is the
#    standard, published 25-phenotype scheme used in their actual benchmark,
#    not the full raw HCUP CCS category list (which includes many
#    non-ICU-relevant or overly granular categories).
# ---------------------------------------------------------------------------
benchmark_phenotypes = {
    name: info for name, info in phenotype_defs.items()
    if info.get("use_in_benchmark") is True
}
print(f"Phenotypes used in benchmark (use_in_benchmark=True): {len(benchmark_phenotypes)}")

# ---------------------------------------------------------------------------
# 3. Build a direct lookup: ICD-9 code -> phenotype name
#    Each phenotype has a list of codes; invert this into one big dictionary
#    for fast lookup per patient.
# ---------------------------------------------------------------------------
code_to_phenotype = {}
for phenotype_name, info in benchmark_phenotypes.items():
    for code in info["codes"]:
        code_to_phenotype[code] = phenotype_name

print(f"Total ICD-9 codes mapped: {len(code_to_phenotype)}")

# ---------------------------------------------------------------------------
# 4. Load our cohort's primary diagnosis codes and map each patient
# ---------------------------------------------------------------------------
icd_data = pd.read_csv("icd9_primary_diagnosis.csv")
icd_data["icd9_code"] = icd_data["icd9_code"].astype(str)

icd_data["phenotype"] = icd_data["icd9_code"].map(code_to_phenotype)

n_mapped = icd_data["phenotype"].notna().sum()
n_total = len(icd_data)
print(f"\nPatients successfully mapped to a phenotype: {n_mapped} / {n_total} "
      f"({100*n_mapped/n_total:.1f}%)")

# ---------------------------------------------------------------------------
# 5. Show the distribution across phenotype groups
# ---------------------------------------------------------------------------
print("\nPatient counts per phenotype group:")
print(icd_data["phenotype"].value_counts())

# ---------------------------------------------------------------------------
# 6. Restrict to our actual final cohort (51,835 patients), not all 61,525
#    raw ICU stays - since our models only ever saw the filtered cohort.
# ---------------------------------------------------------------------------
stay_id_order = pd.read_csv("bat_stay_id_order.csv")
cohort_icd = stay_id_order.merge(icd_data, on="stay_id", how="left")

n_cohort_mapped = cohort_icd["phenotype"].notna().sum()
print(f"\nWithin final cohort (51,835 patients): {n_cohort_mapped} mapped "
      f"({100*n_cohort_mapped/len(cohort_icd):.1f}%)")

# ---------------------------------------------------------------------------
# 7. Save the final mapping, aligned with the cohort's row order - so it
#    can be indexed the same way as bat_X.npy / tpc_X_scaled.npy etc.
# ---------------------------------------------------------------------------
cohort_icd.to_csv("cohort_icd9_phenotype_mapping.csv", index=False)
print("\nSaved: cohort_icd9_phenotype_mapping.csv")
print("Done.")