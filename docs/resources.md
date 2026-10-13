# Resource map

The Git repository contains research code, original-ID records needed for the small curation tables, selected development results, aggregate external metrics and ten candidate hypotheses. Large resources are handled separately so that ordinary Git history remains readable.

| Resource | Original recorded location | Needed for | Git status |
|---|---|---|---|
| Three fixed fine-tuned model states, `model_20261001.pt` through `model_20261003.pt` | `screening_full_fit_2026-10-01_v1` | Archived model inference | [v0.1.0 model release](https://github.com/tail258/TransLock-GPR84/releases/tag/v0.1.0); absent from Git history |
| CheMeleon pretrained checkpoint, `chemeleon_mp.pt` | [Zenodo 15460715](https://doi.org/10.5281/zenodo.15460715) | Model architecture construction and retraining | Download from the fixed upstream record; absent from Git |
| Full ChEMBL 37 screening catalogue and request responses | Recorded 2026-10-01 catalogue snapshot | Replaying the 100,000-record screen | Absent from Git; current API responses are a new data version |
| Full scored-record tables | Recorded catalogue run of 2026-10-01 | Rechecking all 82,883 graph ranks | Absent from Git; summary and ten nominations are included |
| Patent row-level transcriptions and predictions | WO2024028364A1 Table 10 | Recomputing the external AUC from individual labels and scores | Held outside Git pending redistribution review; aggregate metrics and methods are included |
| Conditional docking poses and nine MD trajectories | 2026-10-02–03 recorded runs | Full structural reanalysis | Absent from Git; selected analysis code and results are documented |

For the original model loader, download the three files from the [v0.1.0 model release](https://github.com/tail258/TransLock-GPR84/releases/tag/v0.1.0) into `models/`. Retrieve the author's checkpoint from [Zenodo 15460715](https://doi.org/10.5281/zenodo.15460715) into `models/pretrained/`; `scripts/fetch_chemeleon_weights.py --output models/pretrained` performs that download when the output directory does not yet exist. These paths are intentionally ignored by Git. Keep the associated provenance records and upstream license when distributing or using these resources.

The original full-screening commands use the archived fixed catalogue at `data/catalogue/structures.csv` together with its `manifest.json`. A newly downloaded catalogue cannot silently substitute for this historical snapshot. Only use `--model-dir` with a directory containing all three seeds and the matching training configuration and members.

See [reproduction](reproduction.md) for exact entry points and current validation levels.
