# Small inference input

`demo_molecules.csv` uses the same `molecule_chembl_id` and `canonical_smiles` column names as the fixed catalogue. Two rows are previously nominated, activity-unknown catalogue molecules; one is a previously recorded target identity and one is outside the training heavy-atom range. The example lets you see both model scores and explicit exclusion reasons.

Run the command in [reproduction](../docs/reproduction.md) after placing the model resources in `models/`. Scores from this small input are raw logits. They are not full-catalogue ranks, activity measurements, or new candidate nominations.
