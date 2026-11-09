# Length of Stay Prediction in the ICU: A Comparative Evaluation of Temporal Pointwise Convolution and the Bi-Axial Transformer

BSc Honours research project, School of Computer Science and Applied Mathematics, University of the Witwatersrand, 2026.

- **Supervisor:** Dr. Hairong Bau

## What this project does

This project compares two models for predicting ICU length of stay on MIMIC-III.

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

The `YAIB/` folder is the YAIB framework as modified by Jagd et al. (https://github.com/Katja-Jagd/YAIB). The R scripts in `data_prep/r_cohort/` are edits of the YAIB-cohorts scripts (https://github.com/rvandewater/YAIB-cohorts).

## Data (not included)

No patient data is included, because the MIMIC-III licence does not allow redistribution.

To reproduce this work you need credentialed access to **MIMIC-III v1.4** from PhysioNet (https://physionet.org/content/mimiciii/1.4/). The pretrained BAT weights are not included either.
