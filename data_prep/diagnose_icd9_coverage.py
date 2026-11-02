import pandas as pd
import yaml

with open("hcup_ccs_2015_definitions.yaml", "r") as f:
    phenotype_defs = yaml.safe_load(f)

# Build lookup from the FULL 283 categories (not just the 25 benchmark ones)
code_to_any_phenotype = {}
for phenotype_name, info in phenotype_defs.items():
    for code in info["codes"]:
        code_to_any_phenotype[code] = (phenotype_name, info.get("use_in_benchmark"))

icd_data = pd.read_csv("icd9_primary_diagnosis.csv")
icd_data["icd9_code"] = icd_data["icd9_code"].astype(str)

stay_id_order = pd.read_csv("bat_stay_id_order.csv")
cohort_icd = stay_id_order.merge(icd_data, on="stay_id", how="left")

# Check: does the code exist SOMEWHERE in the full scheme (any of 283 categories)?
def check_match(code):
    if pd.isna(code):
        return "no_code_at_all"
    if code in code_to_any_phenotype:
        _, in_benchmark = code_to_any_phenotype[code]
        return "matched_benchmark_25" if in_benchmark else "matched_but_not_in_25"
    return "code_not_in_scheme_at_all"

cohort_icd["match_status"] = cohort_icd["icd9_code"].apply(check_match)

print("Breakdown of match status across full cohort:")
print(cohort_icd["match_status"].value_counts())
print()
print("As percentages:")
print((cohort_icd["match_status"].value_counts(normalize=True) * 100).round(1))

# Show the most common UNMATCHED codes, to see what's actually being missed
print("\nTop 20 most common codes that don't match ANY of the 283 categories:")
unmatched = cohort_icd[cohort_icd["match_status"] == "code_not_in_scheme_at_all"]
print(unmatched["icd9_code"].value_counts().head(20))
