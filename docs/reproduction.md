# Reproduction guide

The recorded production environment was WSL2 Ubuntu 22.04, uv and Python 3.11. The root `uv.lock` pins a Linux x86_64 CUDA-based CheMeleon environment. New installations can require substantial downloads. A successful dependency installation alone does not prove a research result has been recomputed.

| Entry point | Input | Output | Current public-checkout status |
|---|---|---|---|
| `govern.py --from-raw` | Included ChEMBL source records, curation configuration and the optional data environment | New curation audit and training-unit comparison | Historical package has a completed data replay; a fresh public-checkout run is separately recorded when performed. |
| `demo.py` | Small CSV, fixed author checkpoint and three full-fit model states | All input statuses and raw logits for admitted graphs | New public example. It requires model resources and is validated separately from full catalogue screening. |
| `train.py` | Included training units, fixed author checkpoint and root model environment | New three-seed full-fit directory | Historical three-seed training is recorded. A new fit is a new version, not the 2026-10-01 archived weights. |
| `predict.py` / `screen.py` | Exact historical 100,000-record catalogue and its manifest, training data and matching model directory | Full graph scores and candidate table | Historical fixed run exists. The full historical catalogue is not in Git. |
| `main.py --workflow full` | Same fixed catalogue plus training resources | Training then screening under a new output directory | The unified connection exists and has interface/fixture checks; the formal ZIP states it has not been run as one complete full-scale process. |
| `scripts/run_external_patent_validation.py` | Patent transcription and model states | Row-level external predictions and metrics | Historical single-series result exists. Row-level transcription is outside Git pending separate publication review. |

All output paths should be new directories. For the complete historical inputs, use the corresponding archived snapshot; downloading today's database records creates a different dataset.

## Small inference example

Place the three full-fit states as `models/model_20261001.pt`, `model_20261002.pt`, `model_20261003.pt`, and the fixed CheMeleon author checkpoint as `models/pretrained/chemeleon_mp.pt`. The three seed files are required by the formal model-directory contract even though the example reports the primary seed's scores.

```bash
uv sync --frozen
uv run --frozen --no-sync python demo.py \
  --input examples/demo_molecules.csv \
  --output outputs/demo --device cpu
```

This example applies the same recorded structure standardization and scope checks, then runs the fixed primary model on admitted graphs. The output also records excluded inputs. A small-sample logit has no full-catalogue rank or calibrated probability interpretation.

## Fixed full-catalogue run

The archived `data/catalogue/structures.csv` and its `manifest.json` are required at their documented relative location. After all resources are in place:

```bash
uv run --frozen --no-sync python main.py \
  --workflow full \
  --input data/catalogue/structures.csv \
  --output outputs/full-review --device cuda
```

This command **trains new states** and then scores the full catalogue; do not use it merely to inspect existing results. To use the archived fixed states, call `screen.py` with `--model-dir models` and a new output directory. Full training and screening have significant compute and resource requirements, and this public edition does not claim a fresh end-to-end execution of `main.py --workflow full`.

## Verification levels

The original study preserved fixed configurations, source and unit IDs, prediction and metric files, and independent readback records. The historical submission package reported 37 relevant tests passing. New public-edition checks and their exact scope are recorded in [`verification.md`](verification.md). Do not treat a small example or a fixture check as a complete rerun of training, the 82,883-graph screen or MD.
