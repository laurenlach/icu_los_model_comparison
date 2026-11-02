# Length of Stay Prediction in the ICU: A Comparative Evaluation of Temporal Pointwise Convolution and the Bi-Axial Transformer

BSc Honours research project, School of Computer Science and Applied Mathematics, University of the Witwatersrand, 2026.

- **Supervisor:** Dr. Hairong Bau

## What this project does

This project compares two models for predicting ICU length of stay (LOS) on MIMIC-III.

- **Bi-Axial Transformer** (BAT, Jagd et al.), a clinical time series foundation model.
- **Temporal Pointwise Convolution** (TPC, Rocheteau et al.), a specialised model designed for ICU LOS prediction. 

The aim is to compare a clinical time series foundation model with a specialised model on the same task, using the same cohort, features and data split.

Both models use the same 48 dynamic sensors, the same 4 static features (age, sex, height, weight) and the same 60:20:20 train, validation and test split. Beyond overall accuracy, the project examines the direction of prediction errors (under-prediction is the dangerous direction), which inputs each model relies on (SHAP for TPC, Integrated Gradients for BAT) and how results vary across 25 diagnosis groups. The full analysis and results are in the written report.

## Repository layout

| Folder | What is in it |
|---|---|
| `data_prep/` | Builds the datasets for both models, including scaling, the healthy reference baseline and the diagnosis groups (ICD-9) |
| `data_prep/r_cohort/` | The R scripts that pick the ICU cohort from MIMIC-III |
| `training/` | Code to train TPC, and the Colab notebooks used to train and analyse BAT |
| `evaluation/` | Scripts that measure how accurate each model is and test whether the differences are real |
| `interpretability/` | Scripts that show which inputs each model relies on and how that lines up with SOFA scores |
| `analysis_checks/` | Extra checks I ran to understand the main findings |
| `figures/` | Scripts that make the figures in the report |
| `YAIB/` | The YAIB framework this work builds on, with the few files I changed |

Scripts ending in `_fixed` are the final versions. They were run after a correction to the TPC model, in which part of the model was not connected to its output. The older versions of these scripts are kept for the record.

## Changes to YAIB

The `YAIB/` folder is the YAIB framework as modified by Jagd et al. (https://github.com/Katja-Jagd/YAIB, `development` branch). Almost all of it is unchanged. The files changed for this project are:

- `YAIB/icu_benchmarks/models/dl_models/tpc.py`: the TPC model (added, then corrected after the bug fix)
- `YAIB/configs/prediction_models/BAT_LOS.gin`: BAT configuration for LOS
- `YAIB/configs/prediction_models/common/DLCommon_nodsa.gin`: edited common configuration

In `data_prep/r_cohort/`, the files above are edits of the YAIB-cohorts R scripts (https://github.com/rvandewater/YAIB-cohorts). The other YAIB-cohorts files are unchanged and are not included.

## Data (not included)

No patient data is included, because the MIMIC-III licence does not allow redistribution.

To reproduce this work you need credentialed access to **MIMIC-III v1.4** from PhysioNet (https://physionet.org/content/mimiciii/1.4/). The pretrained BAT weights are not included either:

- MIMIC-III and MIMIC-IV checkpoint: `Katja-Jagd/bat-pretrained-mimiciii-mimiciv-base` on Hugging Face
- MIMIC-IV and eICU checkpoint: `Katja-Jagd/bat-pretrained-eicu-mimiciv-base` on Hugging Face


## Running the pipeline

Run the steps in this order.

1. **Pick the cohort (R).** Install `ricu` and convert MIMIC-III for it. Next to the `R/` folder, create a `config.yaml` with two lines: `ricu_data_path` (where the converted MIMIC data is) and `out_dir` (where the output should go). Then run `base_cohort.R` followed by `los.R` from inside the `R` folder.
2. **Build the datasets.** Run the scripts in `data_prep/`. They create the input arrays for BAT and TPC, scale the features, build the healthy reference baseline and map each patient to a diagnosis group.
3. **Train the models.** TPC is small enough to train on a laptop CPU using the scripts in `training/`. BAT needs a GPU, so it was fine-tuned it on Google Colab using the notebooks in `training/`.
4. **Evaluate and explain.** Run the scripts in `evaluation/` to measure accuracy, then `interpretability/` to see which inputs each model relies on. After that, run `analysis_checks/` and `figures/`.

SOFA scores were pulled with ricu's `load_concepts` in an interactive R session, not saved as a script. The analysis scripts read the resulting tables.
 

## Environment notes

- Python 3 with PyTorch, `shap`, `captum`, `torchmetrics`, `pytorch-lightning`, `gin-config`, `pytorch-ignite==0.5.0.post2` and `einops==0.6.1`.
- **`x-transformers` must be pinned to `1.16.23`.** Newer versions rename internal layers, which makes the pretrained checkpoint load incorrectly.
- Several scripts contain absolute Windows paths (for example `C:\Users\<name>\...`). Edit these to match your own folders before running.
- Hardware used: TPC training and SHAP on a laptop CPU, BAT fine-tuning and Integrated Gradients on a Colab GPU (T4 or L4).

