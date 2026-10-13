# Public-edition verification

Recorded on 2026-10-13 (Asia/Shanghai). These checks ran in the existing WSL2 Ubuntu 22.04 CheMeleon environment with Python 3.11.16, Chemprop 2.2.0, RDKit 2026.03.5 and PyTorch 2.11.0+cu128. The root lockfile was used with `uv run --frozen --no-sync`; pytest 9.1.1 was provided as an ephemeral `uv run --with` tool dependency. No training, full-catalogue scoring or new MD was run.

| Check | Outcome |
|---|---|
| Focused public example, screening, external-metric, benchmark and measurement tests | 30 passed |
| Four-row real CPU example, using the fixed 2026-10-01 states | Two admitted graph scores; known-target and out-of-scope exclusions as expected |
| CHEMBL120691 primary logit | 6.3348917961120605, identical to archived candidate table |
| CHEMBL1086046 primary logit | 5.689266204833984, identical to archived candidate table |
| Public source scan | No personal absolute paths or obvious credential-token patterns in selected text files |

The local validation loaded original weight files outside Git. The release and fresh-install path have not been equated with this reuse of an existing environment. The full `main.py --workflow full` run has not been executed as one new full-scale run in this public edition. Some original-package tests require intentionally omitted resources such as full docking pose files and the 100,000-record catalogue; the 30-test result above is the exact tested scope.

## Equivalent commands

```bash
UV_PROJECT_ENVIRONMENT=/path/to/existing/chemeleon-env \
  uv run --with pytest==9.1.1 --frozen --no-sync python -m pytest -q \
  tests/test_public_demo.py tests/test_screening_application.py \
  tests/test_external_validation.py tests/test_benchmark.py tests/test_measurements.py

UV_PROJECT_ENVIRONMENT=/path/to/existing/chemeleon-env \
  uv run --frozen --no-sync python demo.py \
  --input examples/demo_molecules.csv \
  --output outputs/demo-real --device cpu
```

These commands normalize private local paths. Use a new output directory on each run. The original code package reported 37 relevant checks, a separate historical statement with a different file set and verification scope.
