"""公开目录的结构准入、独立近邻参照与固定候选提名规则。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold


def validate_source(manifest, expected_records):
    selected = manifest["selection"]
    if (
        manifest["status"] != "complete"
        or manifest["database_version"] != "ChEMBL_37"
        or selected["molecule_type"] != "Small molecule"
        or selected["order_by"] != "molecule_chembl_id"
        or selected["requested_limit"] != expected_records
        or manifest["statistics"]["downloaded_records"] != expected_records
        or manifest["known_target_identity"]["target_chembl_id"] != "CHEMBL3714079"
    ):
        raise ValueError("目录来源、数量或靶点与协议不同")


def validate_training_scope(units, scope):
    molecules = [Chem.MolFromSmiles(value) for value in units.smiles.unique()]
    sizes = [mol.GetNumHeavyAtoms() for mol in molecules]
    elements = {atom.GetSymbol() for mol in molecules for atom in mol.GetAtoms()}
    if (min(sizes), max(sizes)) != (scope["min_heavy_atoms"], scope["max_heavy_atoms"]):
        raise ValueError("训练图大小范围与配置不一致")
    if elements != set(scope["allowed_elements"]) - {"H"}:
        raise ValueError("训练图元素范围与配置不一致")


def normalize(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("missing_structure")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid_structure")
    mol = rdMolStandardize.Cleanup(mol)
    mol = rdMolStandardize.FragmentParent(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.SanitizeMol(mol)
    return mol


def connection(mol):
    copy = Chem.Mol(mol)
    Chem.RemoveStereochemistry(copy)
    return Chem.MolToSmiles(copy, isomericSmiles=False)


def properties(mol):
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    Chem.RemoveStereochemistry(scaffold)
    return {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "molecular_weight": Descriptors.MolWt(mol),
        "clogp": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
        "unspecified_tetrahedral_centers": sum(
            label == "?"
            for _, label in Chem.FindMolChiralCenters(
                mol, includeUnassigned=True, useLegacyImplementation=False
            )
        ),
        "scaffold": Chem.MolToSmiles(scaffold, isomericSmiles=False),
    }


def prepare_catalogue(raw, known_ids, known_connections, scope, progress=None):
    if raw.molecule_chembl_id.duplicated().any():
        raise ValueError("目录来源ID重复")
    rows, seen = [], {}
    for index, source in enumerate(raw.to_dict("records")):
        identifier = source["molecule_chembl_id"]
        original = source.get("canonical_smiles")
        row = {
            "input_order": index,
            "candidate_id": identifier,
            "source_url": f"https://www.ebi.ac.uk/chembl/explore/compound/{identifier}",
            "source_smiles": original,
            "smiles": None,
            "connection_id": None,
            "status": "pending",
            "reasons": "",
            "duplicate_of": None,
            "gpr84_activity": "unknown_in_checked_scope",
            "functional_direction": "unestablished",
        }
        reasons = []
        if identifier in known_ids:
            reasons.append("known_target_id")
        try:
            mol = normalize(original)
            row["smiles"] = Chem.MolToSmiles(mol, isomericSmiles=True)
            row["connection_id"] = connection(mol)
            row.update(properties(mol))
            if row["connection_id"] in known_connections:
                reasons.append("known_or_quarantined_connection")
            if not scope["min_heavy_atoms"] <= mol.GetNumHeavyAtoms() <= scope["max_heavy_atoms"]:
                reasons.append("outside_training_size_scope")
            if any(atom.GetSymbol() not in scope["allowed_elements"] for atom in mol.GetAtoms()):
                reasons.append("outside_training_element_scope")
            if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
                reasons.append("radical_structure")
            if len(Chem.GetMolFrags(mol)) != 1:
                reasons.append("multiple_parent_fragments")
            if not reasons and row["smiles"] in seen:
                row["duplicate_of"] = seen[row["smiles"]]
                reasons.append("duplicate_eligible_graph")
            if not reasons:
                seen[row["smiles"]] = identifier
        except (ValueError, RuntimeError) as error:
            reasons.append(str(error))
        row["reasons"] = ";".join(reasons)
        if any(reason.startswith("known") for reason in reasons):
            row["status"] = "known_target_excluded"
            row["gpr84_activity"] = "recorded_or_quarantined_not_reinterpreted"
        elif row["duplicate_of"] is not None:
            row["status"] = "duplicate_excluded"
        elif reasons:
            row["status"] = "structure_or_scope_excluded"
        else:
            row["status"] = "ready_for_scoring"
        rows.append(row)
        if progress is not None and (index + 1) % 1000 == 0:
            progress(index + 1, len(raw))
    return pd.DataFrame(rows)


def fingerprint(mol):
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=3, includeChirality=True
    ).GetSparseCountFingerprint(mol)


def nearest_support(smiles, units):
    """对每个assay图单位找近邻；并列取原单位顺序，保留双类图的两侧证据。"""
    train_fps = [fingerprint(Chem.MolFromSmiles(value)) for value in units.smiles]
    labels = units.label.to_numpy()
    if set(labels) != {0, 1}:
        raise ValueError("近邻参照要求训练两类齐全")
    result = []
    for value in smiles:
        fp = fingerprint(Chem.MolFromSmiles(value))
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
        row = {}
        for name, indices in [
            ("any", np.arange(len(units))),
            ("strong", np.flatnonzero(labels == 1)),
            ("weak", np.flatnonzero(labels == 0)),
        ]:
            index = int(indices[np.argmax(similarities[indices])])
            row[f"nearest_{name}_similarity"] = float(similarities[index])
            row[f"nearest_{name}_unit_id"] = units.iloc[index].unit_id
        result.append(row)
    return pd.DataFrame(result)


def add_rankings(scored, seeds):
    frame = scored.copy()
    if frame.candidate_id.duplicated().any():
        raise ValueError("评分候选ID重复")
    for seed in seeds:
        col = f"score_{seed}"
        if not np.isfinite(frame[col].to_numpy()).all():
            raise ValueError("非有限评分")
        ordered = frame.sort_values([col, "candidate_id"], ascending=[False, True])
        frame.loc[ordered.index, f"rank_{seed}"] = np.arange(1, len(frame) + 1)
        frame[f"rank_{seed}"] = frame[f"rank_{seed}"].astype(int)
    ordered = frame.sort_values(
        ["nearest_strong_similarity", "candidate_id"], ascending=[False, True]
    )
    frame.loc[ordered.index, "similarity_reference_rank"] = np.arange(1, len(frame) + 1)
    frame["similarity_reference_rank"] = frame.similarity_reference_rank.astype(int)
    rank_columns = [f"rank_{seed}" for seed in seeds]
    frame["seed_rank_min"] = frame[rank_columns].min(axis=1)
    frame["seed_rank_max"] = frame[rank_columns].max(axis=1)
    frame["seed_rank_span"] = frame.seed_rank_max - frame.seed_rank_min
    return frame


def nominate(scored, ranking):
    primary_rank = f"rank_{ranking['primary_seed']}"
    budget = min(ranking["pool_cap"], math.ceil(len(scored) * ranking["pool_fraction"]))
    pool = scored[scored[primary_rank] <= budget].sort_values(primary_rank)
    keep, seen = [], set()
    for index, row in pool.iterrows():
        scaffold = row.scaffold if pd.notna(row.scaffold) else ""
        if scaffold in seen:
            continue
        seen.add(scaffold)
        keep.append(index)
        if len(keep) == ranking["maximum_candidates"]:
            break
    result = scored.loc[keep].copy()
    result.insert(0, "nomination_order", np.arange(1, len(result) + 1))
    return result, budget
