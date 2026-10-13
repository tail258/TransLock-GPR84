# Methods and study stages

## Data units and admission

The recorded ChEMBL 37 query produced 2,070 GPR84 activity records. The curation pipeline admitted 285 measurements, aggregated them into 277 non-conflicting assay-level molecular units, and encoded 248 distinct molecular graphs. They came from six publications and 17 assay systems. The steps, exclusions and original IDs are stored in `data/source/` and `data/training/`; the original unit of observation must not be silently replaced by a graph count.

The task is human GPR84 functional antagonism, with an IC50 threshold of 1 µM. Only a point or censored interval wholly on one side of the threshold receives a class. Missing or unmeasured molecules are not assigned the weaker class. Within an assay, compatible measurements of the same standardized graph are combined and original measurement IDs remain traceable. Cross-assay occurrences receive graph-frequency weighting for training.

## Internal development

Each eligible publication was held out in turn. Identical non-stereo connections appearing in both its test portion and the remaining training portion were removed from training. Five suitable assay groups contributed top-20% enrichment metrics, weighted equally across groups and averaged over the fixed three model seeds. Fine-tuning, a matched frozen-encoder setting and a fingerprint similarity reference were compared. These comparisons were used during method development.

The main model is a CheMeleon-initialized Chemprop 2.2.0 bond-message-passing encoder with mean aggregation, a non-affine layer normalization and a linear binary head. The fixed full-fit stage uses all 277 admitted assay units, 30 epochs and seeds 20261001, 20261002 and 20261003. This full fit has no held-out validation set or test-selected epoch. Configurations are in `configs/` and the detailed architecture is in the [model card](../models/Model_Card.md).

## Catalogue application

The fixed ChEMBL 37 catalogue input consists of 100,000 records in ascending source-ID order. Parent/charge standardization, scope filtering, known-target identity exclusion and graph deduplication produced 82,883 scored graphs. The primary 20261001 raw-logit ranking forms a top-1% pool, capped at 1,000 graphs; the first distinct non-stereo Bemis–Murcko scaffolds yield up to ten nominations. The other two seeds describe ranking sensitivity and do not choose the primary results.

The model's heavy-atom and element limits come from observed training coverage. They are screening rules, not a proven applicability domain. The ten nominated structures have unknown GPR84 functional activity and direction.

## External retrospective evaluation

The external set comprises 61 patent-reported compounds from WO2024028364A1 Table 10, measured in a human GPR84 Gαi-BRET antagonist assay. Patent bins A and B are strong at the study's 1 µM cutoff, C is weaker: 53 strong and 8 weaker. They were checked against the project's recorded supervised-training identities. The model, primary seed, ranking budget and similarity reference were fixed before scoring; the patent labels had already been read. Evaluation included a planned 60-connection collapse sensitivity analysis.

The external comparison supports a positive result within this single patent series. It does not validate the activity of the ten nominated catalogue molecules. Full definitions, random expectation and numerical results are in [evaluation](evaluation.md).

## Structural interpretation

Conditional docking poses and short membrane MD runs add structural context for two nominations and GLPG1205. Each of the three systems has three 5 ns replicas. The ensemble is too short to establish equilibrium, binding free energy or functional direction. Original docking poses and MD trajectories remain in the project archive; the code and selected derived summaries are included or described here.
