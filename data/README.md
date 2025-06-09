# Data Directory

This directory is created by the pipeline and holds all datasets used during training. The main subfolders are:

- `0_raw/` – original data dumps.
- `1_preprocess/` and `1_preprocess_ts/` – cleaned and time-series specific files.
- `2_processed/` and `2_model_input/` – feature engineered datasets for modelling.
- `2_trainingdata_ts/` – prepared time-series training splits.
- `3_trainingdata/` – final supervised learning matrices.
- `4_results/` – model outputs and predictions.
- `5_metrics/` – evaluation metrics and charts.

Only minimal sample files live under `data/samples/` in the repository. All other folders are created during execution and are excluded from version control.
