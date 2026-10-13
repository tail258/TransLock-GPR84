"""固定候选与有出处参照的结构比较；不赋予药理结论。"""

from itertools import combinations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from translock_screening.screening_application import connection, normalize


def describe(smiles):
    mol = normalize(smiles)
    Chem.RemoveStereochemistry(mol)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return {
        "connection": connection(mol),
        "scaffold": Chem.MolToSmiles(scaffold, isomericSmiles=False),
        "generic_framework": Chem.MolToSmiles(generic, isomericSmiles=False),
        "has_ring_scaffold": scaffold.GetNumAtoms() > 0,
    }


def count_fp(smiles):
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=3, includeChirality=False
    ).GetSparseCountFingerprint(Chem.MolFromSmiles(smiles))


def prepare_reference(frame, scope, id_column, smiles_column):
    rows, failures = [], []
    for row in frame.to_dict("records"):
        entry = {"scope": scope, "reference_id": str(row[id_column])}
        try:
            rows.append(
                {**entry, "source_smiles": row[smiles_column], **describe(row[smiles_column])}
            )
        except (ValueError, RuntimeError) as error:
            failures.append({**entry, "error": str(error), "source_smiles": row[smiles_column]})
    if not rows:
        raise ValueError("参照没有有效结构")
    return pd.DataFrame(rows), failures


def compare(candidates, reference):
    # 同图多来源保留原表，近邻计算只用同范围内的首个图代表。
    unique = reference.drop_duplicates("connection")
    fps = [count_fp(s) for s in unique.connection]
    sfps = [count_fp(s) for s in unique.scaffold]
    results = []
    for row in candidates.to_dict("records"):
        desc = describe(row["smiles"])
        graph_sims = np.asarray(
            DataStructs.BulkTanimotoSimilarity(count_fp(desc["connection"]), fps)
        )
        scaffold_sims = np.asarray(
            DataStructs.BulkTanimotoSimilarity(count_fp(desc["scaffold"]), sfps)
        )
        graph_neighbor = unique.iloc[int(graph_sims.argmax())]
        scaffold_neighbor = unique.iloc[int(scaffold_sims.argmax())]
        result = {
            "candidate_id": row["candidate_id"],
            "scope": reference.iloc[0].scope,
            **desc,
            "reference_rows": len(reference),
            "reference_graphs": len(unique),
        }
        for kind in ("connection", "scaffold", "generic_framework"):
            # 无环分子间的空骨架不当作有信息的相同骨架证据。
            matches = reference[reference[kind] == desc[kind]] if desc[kind] else reference.iloc[:0]
            result[f"{kind}_match_rows"] = len(matches)
            result[f"{kind}_match_ids"] = "||".join(matches.reference_id)
        for name, neighbor, sims in (
            ("graph", graph_neighbor, graph_sims),
            ("scaffold", scaffold_neighbor, scaffold_sims),
        ):
            result[f"nearest_{name}_similarity"] = float(sims.max())
            result[f"nearest_{name}_reference_id"] = neighbor.reference_id
            result[f"nearest_{name}_smiles"] = neighbor.connection
        results.append(result)
    return pd.DataFrame(results)


def candidate_pairs(candidates):
    descriptions = [(r.candidate_id, describe(r.smiles)) for r in candidates.itertuples()]
    result = []
    for (left, a), (right, b) in combinations(descriptions, 2):
        row = {"left": left, "right": right}
        for kind in ("connection", "scaffold", "generic_framework"):
            row[f"same_{kind}"] = bool(a[kind]) and a[kind] == b[kind]
        for kind in ("connection", "scaffold"):
            row[f"{kind}_similarity"] = DataStructs.TanimotoSimilarity(
                count_fp(a[kind]), count_fp(b[kind])
            )
        result.append(row)
    return pd.DataFrame(result)
