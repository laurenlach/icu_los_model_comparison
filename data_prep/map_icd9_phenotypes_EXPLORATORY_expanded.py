import pandas as pd
import yaml

# EXPLORATORY TEST: use ALL 283 HCUP CCS categories, not just the 25
# Harutyunyan et al. selected for their benchmark. This is a supplementary check to see how much
# coverage improves if we accept a larger, less-established set of categories.
#

with open("hcup_ccs_2015_definitions.yaml", "r") as f:
    phenotype_defs = yaml.safe_load(f)

print(f"Using ALL {len(phenotype_defs)} categories (not just the 25 benchmark ones)")

code_to_phenotype_all = {}
for phenotype_name, info in phenotype_defs.items():
    for code in info["codes"]:
        code_to_phenotype_all[code] = phenotype_name

icd_data = pd.read_csv("icd9_primary_diagnosis.csv")
icd_data["icd9_code"] = icd_data["icd9_code"].astype(str)
icd_data["phenotype_expanded"] = icd_data["icd9_code"].map(code_to_phenotype_all)

stay_id_order = pd.read_csv("bat_stay_id_order.csv")
cohort_icd_expanded = stay_id_order.merge(icd_data, on="stay_id", how="left")

n_mapped = cohort_icd_expanded["phenotype_expanded"].notna().sum()
n_total = len(cohort_icd_expanded)
print(f"\nWith all {len(phenotype_defs)} categories (primary diagnosis only): "
      f"{n_mapped} / {n_total} mapped ({100*n_mapped/n_total:.1f}%)")
print("(Compare to standard 25-category, primary-only result: 25,783 / 51,835 = 49.7%)")

counts = cohort_icd_expanded["phenotype_expanded"].value_counts()
print(f"\nTotal distinct phenotype groups populated: {len(counts)}")
print(f"Groups with fewer than 30 patients: {(counts < 30).sum()}")
print(f"Groups with fewer than 10 patients: {(counts < 10).sum()}")

print("\nTop 25 groups by patient count:")
print(counts.head(25))

print("\nBottom 10 groups by patient count (smallest, most statistically fragile):")
print(counts.tail(10))

cohort_icd_expanded.to_csv("cohort_icd9_phenotype_mapping_EXPLORATORY_all_categories.csv", index=False)
print("\nSaved: cohort_icd9_phenotype_mapping_EXPLORATORY_all_categories.csv")
print("Done.")
