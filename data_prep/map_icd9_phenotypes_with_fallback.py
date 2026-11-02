import pandas as pd
import yaml

# 1. Load Harutyunyan et al.'s STANDARD 25-phenotype scheme 
with open("hcup_ccs_2015_definitions.yaml", "r") as f:
    phenotype_defs = yaml.safe_load(f)

benchmark_phenotypes = {
    name: info for name, info in phenotype_defs.items()
    if info.get("use_in_benchmark") is True
}
print(f"Using standard {len(benchmark_phenotypes)}-phenotype benchmark scheme")

code_to_phenotype = {}
for phenotype_name, info in benchmark_phenotypes.items():
    for code in info["codes"]:
        code_to_phenotype[code] = phenotype_name

# 2. Pull all diagnoses per stay 
all_diag = pd.read_csv("icd9_all_diagnoses.csv")
all_diag["icd9_code"] = all_diag["icd9_code"].astype(str)
all_diag = all_diag.sort_values(["stay_id", "seq_num"])

print(f"Total diagnosis rows (all seq_nums): {len(all_diag)}")
print(f"Unique patients with at least one diagnosis: {all_diag['stay_id'].nunique()}")


# 3. For each patient, walk through their diagnoses in seq_num order and
#    take the first one that matches one of the 25 benchmark phenotypes 
def find_first_match(group):
    for code in group["icd9_code"]:
        if code in code_to_phenotype:
            return code_to_phenotype[code]
    return None  # no diagnosis for this patient matched any of the 25

print("\nSearching each patient's full diagnosis list for a match...")
phenotype_with_fallback = all_diag.groupby("stay_id").apply(find_first_match)
phenotype_with_fallback = phenotype_with_fallback.reset_index()
phenotype_with_fallback.columns = ["stay_id", "phenotype"]


# 4. Restrict to our actual final cohort and report coverage
stay_id_order = pd.read_csv("bat_stay_id_order.csv")
cohort_icd_fallback = stay_id_order.merge(phenotype_with_fallback, on="stay_id", how="left")

n_mapped = cohort_icd_fallback["phenotype"].notna().sum()
n_total = len(cohort_icd_fallback)
print(f"\nWith secondary-diagnosis fallback: {n_mapped} / {n_total} mapped "
      f"({100*n_mapped/n_total:.1f}%)")
print("(Compare to primary-only result: 25,783 / 51,835 = 49.7%)")

print("\nPatient counts per phenotype group (with fallback):")
print(cohort_icd_fallback["phenotype"].value_counts())


# 5. Save 
cohort_icd_fallback.to_csv("cohort_icd9_phenotype_mapping_with_fallback.csv", index=False)
print("\nSaved: cohort_icd9_phenotype_mapping_with_fallback.csv")
print("Done.")
