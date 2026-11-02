# Length of Stay Prediction in the ICU: A Comparative Evaluation of Temporal Pointwise Convolution and the Bi-Axial Transformer

BSc Honours research project, School of Computer Science and Applied Mathematics, University of the Witwatersrand, 2026.

- **Author:** Lauren Lachman (Student No. 2702445)
- **Supervisor:** Dr. Hairong Bau

## What this project does

This project compares two models for predicting ICU length of stay (LOS) on MIMIC-III.

- **BAT** (Bi-Axial Transformer, Jagd et al.), a time-series foundation model. A pretrained checkpoint is fine-tuned for single-value LOS regression.
- **TPC** (Temporal Pointwise Convolution, Rocheteau et al.), a specialised model. It is reimplemented from scratch and simplified to single-task LOS regression.

Both models use the same cohort, the same 48 dynamic sensors, the same 4 static features (age, sex, height, weight) and the same 60:20:20 train/validation/test split. Beyond overall accuracy, the project examines the direction of prediction errors (under-prediction is the clinically dangerous direction), which inputs each model relies on (SHAP for TPC, Integrated Gradients for BAT, permutation importance, alignment with SOFA organ scores) and how results vary across 25 diagnosis groups. The full analysis and results are in the written report.

## Repository layout

| Folder | Contents |
|---|---|
| `data_prep/` | Dataset construction for both models, feature scaling, healthy-reference baseline, ICD-9 phenotype mapping |
| `data_prep/r_cohort/` | The modified R cohort-extraction scripts (YAIB-cohorts / ricu): `base_cohort.R`, `los.R`, `.load_ricu.R`, `src/obs_time.R`, `ricu-extensions/configs/output/concept-dict.json` |
| `training/` | TPC model tests and training scripts, BAT debug training, and the Colab notebooks used to train and analyse BAT on a GPU |
| `evaluation/` | Test-set metrics, threshold sensitivity, per-phenotype performance, significance tests, re-run of BAT on the second pretrained checkpoint |
| `interpretability/` | SHAP (TPC), Integrated Gradients (BAT), permutation importance, SOFA alignment, phenotype-level rankings, comparative summary |
| `analysis_checks/` | Follow-up checks and hypothesis tests used to explain the main findings |
| `figures/` | Scripts that produce the report figures |
| `YAIB/` | The YAIB framework (see below), with the files changed for this project |

Scripts whose names end in `_fixed` were run after the TPC bug fix (the temporal branch of the original TPC `forward()` was disconnected from the prediction path) and are the versions used for the reported TPC results. Earlier versions are kept for the record.

## What is original and what is not

The `YAIB/` folder is the YAIB framework as modified by Jagd et al. (https://github.com/Katja-Jagd/YAIB, `development` branch). Almost all of it is unchanged. The files changed for this project are:

- `YAIB/icu_benchmarks/models/dl_models/tpc.py`: the TPC model (added, then corrected after the bug fix)
- `YAIB/configs/prediction_models/BAT_LOS.gin`: BAT configuration for LOS
- `YAIB/configs/prediction_models/common/DLCommon_nodsa.gin`: edited common configuration

In `data_prep/r_cohort/`, the files above are edits of the YAIB-cohorts R scripts (https://github.com/rvandewater/YAIB-cohorts). The other YAIB-cohorts files are unchanged and are not included.

## Data (not included)

No patient data is included, because the MIMIC-III licence does not allow redistribution.

To reproduce this work you need credentialed access to **MIMIC-III v1.4** from PhysioNet (https://physionet.org/content/mimiciii/1.4/). The pretrained BAT weights are not included either:

- MIMIC-III/IV checkpoint: `Katja-Jagd/bat-pretrained-mimiciii-mimiciv-base` on Hugging Face
- MIMIC-IV + eICU checkpoint: `Katja-Jagd/bat-pretrained-eicu-mimiciv-base` on Hugging Face

## Running the pipeline

1. **Cohort extraction (R).** Install `ricu` and convert MIMIC-III for it. Create a `config.yaml` one level above the `R/` folder containing `ricu_data_path` (where the converted MIMIC data lives) and `out_dir`. Run `base_cohort.R`, then `los.R`, from inside the `R` folder.
2. **Dataset construction.** Run the scripts in `data_prep/` (merge LOS, build the BAT and TPC arrays, compute scaling statistics, build the healthy-reference baseline, map ICD-9 phenotypes).
3. **Training.** TPC trains on a laptop CPU with the scripts in `training/`. BAT is fine-tuned on a GPU (Google Colab) with the notebooks in `training/`.
4. **Evaluation and interpretability.** Run the scripts in `evaluation/`, then `interpretability/`, then `analysis_checks/` and `figures/`.

SOFA scores were pulled with ricu's `load_concepts` in an interactive R session, not saved as a script. The analysis scripts read the resulting tables.

## Environment notes

- Python 3 with PyTorch, `shap`, `captum`, `torchmetrics`, `pytorch-lightning`, `gin-config`, `pytorch-ignite==0.5.0.post2`, `einops==0.6.1`.
- **`x-transformers` must be pinned to `1.16.23`.** Newer versions rename internal layers and make the pretrained checkpoint load incorrectly.
- Several scripts contain absolute Windows paths (for example `C:\Users\Laure\...`). Edit these to match your own folders before running.
- Hardware used: TPC training and SHAP on a laptop CPU; BAT fine-tuning and Integrated Gradients on a Colab GPU (T4 or L4).

## Acknowledgements

- Jagd et al., Bi-Axial Transformer and its pretrained checkpoints, and the YAIB fork this work builds on
- van de Water et al., YAIB (Yet Another ICU Benchmark) and YAIB-cohorts
- Rocheteau et al., Temporal Pointwise Convolution
- Bennett et al., `ricu`
- Johnson et al., MIMIC-III
