# Third-party materials and provenance

| Component | Role | Source and terms |
|---|---|---|
| CheMeleon message-passing adaptation | Molecular encoder interface in `src/translock_screening/chemeleon_encoder.py` | Adapted from the authors' [standalone fingerprint implementation](https://github.com/JacksonBurns/chemeleon/blob/b5e5bd88a7070c3f278e7e65e85b792617bf125c/chemeleon_fingerprint.py); MIT notice in [`licenses/CheMeleon-MIT.txt`](licenses/CheMeleon-MIT.txt). Copyright (c) 2025 Jackson Burns. |
| Chemprop 2.2.0 | Molecular graph features and message-passing implementation | [Official source](https://github.com/chemprop/chemprop/tree/v2.2.0); MIT notice in [`licenses/chemprop_v2.2.0_LICENSE.txt`](licenses/chemprop_v2.2.0_LICENSE.txt). |
| CheMeleon pretrained encoder | Initialization for project fine-tuning | [Fixed Zenodo record 15460715](https://doi.org/10.5281/zenodo.15460715), 34,859,448-byte checkpoint; record metadata states MIT. The author pretraining corpus was not copied or rerun here. |
| ChEMBL 37 | GPR84 activity records and screening-catalogue source | [ChEMBL API](https://www.ebi.ac.uk/chembl/api/data/) and [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/legalcode). Original IDs, publication DOIs and retrieval context are retained in `data/`. Cite ChEMBL and the underlying experimental papers. |
| WO2024028364A1 | Single-series external retrospective evaluation | [Public patent record](https://patents.google.com/patent/WO2024028364A1/en). The aggregate metrics are included here; manually transcribed row-level patent material is held outside Git pending a separate redistribution review. |
| PDB 9WIJ | Source structure for conditional pose work | [RCSB entry](https://www.rcsb.org/structure/9WIJ); derived pose coordinates and MD trajectories are outside Git. |

Project-authored code and documentation are licensed under the [project MIT License](LICENSE). This license does not replace the separate terms of the data, model checkpoints or third-party material above. Installed Python packages have their own terms; exact versions are recorded in `uv.lock` and the optional environments.
