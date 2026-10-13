# Evaluation and interpretation

## Internal development comparison

The internal protocol used publication holdout, a non-stereo connection-overlap exclusion, five suitable assay groups, a top-20% selection budget and three fixed seeds. The recorded macro-mean EF values were:

| Method | Mean EF |
|---|---:|
| CheMeleon fine-tuned | 1.112 |
| Same architecture, frozen encoder | 1.016 |
| Nearest strong training-graph fingerprint similarity | 1.083 |

These comparisons informed method selection. They are development evidence, not independent validation. Assay-level and seed-level details are in [`results/internal_development/`](../results/internal_development/) and [`results/internal_cross/`](../results/internal_cross/).

## Single-series external retrospective result

The external source was [WO2024028364A1](https://patents.google.com/patent/WO2024028364A1/en), Table 10. It reported 61 human GPR84 antagonist measurements grouped into 53 strong (A/B) and 8 weaker (C) compounds. At the fixed top-13 budget, random selection would include 13 × 53 / 61 = **11.295** strong compounds on average. This unusually high strong fraction limits the headroom of top-budget enrichment.

| Ranking method | Strong in top 13 | EF | ROC-AUC |
|---|---:|---:|---:|
| Primary fine-tuned model, seed 20261001 | 12 | 1.062 | 0.646 |
| Predetermined sensitivity seed 20261002 | 13 | 1.151 | 0.745 |
| Predetermined sensitivity seed 20261003 | 13 | 1.151 | 0.757 |
| Original nearest-strong similarity reference | 10 | 0.885 | 0.401 |

The two sensitivity seeds are not two independent external sources. After collapsing one stereochemical pair to 60 connection graphs, the primary model selected 11 strong compounds at a fixed budget of 12 and the reference selected 10. Aggregate [metrics](../results/external/metrics.csv) are included. The row-level patent transcription and individual predictions are not in this Git repository while their separate redistribution status is being reviewed. The [public patent](https://patents.google.com/patent/WO2024028364A1/en) and the fixed evaluation code remain available for independent reconstruction.

The comparison supports prioritization in this patent series. The labels were known before the one-time evaluation, although the model and rules were fixed before scoring. It is neither a prospective blind test nor a measurement of the ten catalogue nominations. Unlabeled overlap between the original CheMeleon pretraining corpus and these patent molecules has not been audited molecule by molecule.

## Screened catalogue and short MD

The 100,000-record application scored 82,883 distinct graphs and nominated ten. This is an implementation and traceability result; it is not a measured hit rate. The main seed's top-1% pool contained 829 graphs. The selected ten lay in each predetermined seed's top 829, but the complete top-829 pools differed, so this observation does not imply stable whole-catalogue ordering.

Two nominations and GLPG1205 were used in nine 5 ns membrane MD replicas. The results constrain short-term pose and contact interpretation. They cannot establish antagonist activity, potency, binding free energy or long-time stability.

The raw-logit output is not calibrated to IC50 or probability. Consult the [model card](../models/Model_Card.md) before using individual scores.
