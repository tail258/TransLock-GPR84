"""独立回读骨架及连续近邻比较，生成内部结构证据页。"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold


def manual_similarity(a, b):
    keys = a.keys() | b.keys()
    denominator = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return sum(min(a.get(k, 0), b.get(k, 0)) for k in keys) / denominator if denominator else 0.0


def keys_from_smiles(smi):
    mol = rdMolStandardize.Uncharger().uncharge(
        rdMolStandardize.FragmentParent(rdMolStandardize.Cleanup(Chem.MolFromSmiles(smi)))
    )
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.SanitizeMol(mol)
    Chem.RemoveStereochemistry(mol)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return [
        Chem.MolToSmiles(mol, isomericSmiles=False),
        MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False),
        Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaffold), isomericSmiles=False),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    candidates = pd.read_csv(args.run / "input_candidates.csv")
    comparisons = pd.read_csv(args.run / "candidate_comparisons.csv", keep_default_na=False)
    reference = pd.read_csv(args.run / "reference_structures.csv", keep_default_na=False)
    fields = ["connection", "scaffold", "generic_framework"]
    source_root = Path(manifest["inputs"][1]["path"]).parents[3]
    # 按原始参照重新生成身份和骨架，不调用生产描述/比较函数。
    rebuilt = []
    for scope, spec in manifest["config"]["references"].items():
        frame = pd.read_csv(source_root / spec["path"], dtype=str, keep_default_na=False)
        for row in frame.to_dict("records"):
            rebuilt.append(
                {
                    "scope": scope,
                    "reference_id": row[spec["id_column"]],
                    **dict(zip(fields, keys_from_smiles(row[spec["smiles_column"]]), strict=True)),
                }
            )
    check_frame = pd.DataFrame(rebuilt)
    assert check_frame.to_dict("records") == reference[["scope", "reference_id", *fields]].to_dict(
        "records"
    )
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, includeChirality=False)
    fingerprints = {}

    def fp(smi):
        if smi not in fingerprints:
            fingerprints[smi] = generator.GetSparseCountFingerprint(
                Chem.MolFromSmiles(smi)
            ).GetNonzeroElements()
        return fingerprints[smi]

    for candidate in candidates.itertuples():
        descriptors = keys_from_smiles(candidate.smiles)
        rows = comparisons[comparisons.candidate_id == candidate.candidate_id]
        assert len(rows) == len(manifest["config"]["references"])
        for row in rows.itertuples():
            ref = check_frame[check_frame.scope == row.scope]
            for field, value in zip(fields, descriptors, strict=True):
                matching = ref[ref[field] == value] if value else ref.iloc[:0]
                assert len(matching) == getattr(row, field + "_match_rows")
                assert "||".join(matching.reference_id) == getattr(row, field + "_match_ids")
            unique = ref.drop_duplicates("connection")
            for name, field, smi in (
                ("graph", "connection", descriptors[0]),
                ("scaffold", "scaffold", descriptors[1]),
            ):
                values = [manual_similarity(fp(smi), fp(v)) for v in unique[field]]
                index = int(np.argmax(values))
                assert unique.iloc[index].reference_id == getattr(
                    row, "nearest_" + name + "_reference_id"
                )
                assert abs(values[index] - getattr(row, "nearest_" + name + "_similarity")) < 1e-12
    pairs = pd.read_csv(args.run / "candidate_pairs.csv")
    assert len(pairs) == 45
    by_id = {r.candidate_id: keys_from_smiles(r.smiles) for r in candidates.itertuples()}
    for row in pairs.itertuples():
        for index, field in enumerate(fields):
            a, b = by_id[row.left][index], by_id[row.right][index]
            assert bool(a) and (a == b) == bool(getattr(row, "same_" + field))
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), output / "source_snapshot.py")
    verification = {
        "status": "passed",
        "reference_rows": len(reference),
        "comparison_rows": len(comparisons),
        "pair_rows": len(pairs),
        "checks": [
            "原始结构重新标准化与骨架提取",
            "匹配成员与数量",
            "手写计数Tanimoto及全部近邻最大值",
            "45对候选骨架关系",
        ],
        "scope": "不重训、不重评分、不做哈希",
    }
    (output / "independent_readback.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    view = comparisons[comparisons.scope == "admitted_antagonist_context"].copy()
    mols, legends = [], []
    for row in view.itertuples():
        mols.extend(
            [Chem.MolFromSmiles(row.connection), Chem.MolFromSmiles(row.nearest_graph_smiles)]
        )
        legends.extend(
            [
                row.candidate_id + " | candidate",
                f"known antagonist-context neighbor | r3 T={row.nearest_graph_similarity:.3f}",
            ]
        )
    Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(500, 320), legends=legends).save(
        output / "candidate_known_neighbors.png"
    )
    (output / "candidate_known_neighbors.svg").write_text(
        Draw.MolsToGridImage(
            mols, molsPerRow=2, subImgSize=(500, 320), legends=legends, useSVG=True
        ),
        encoding="utf-8",
    )
    summary = pd.DataFrame(manifest["scope_summary"])
    view = view[
        [
            "candidate_id",
            "nearest_graph_similarity",
            "nearest_graph_reference_id",
            "nearest_scaffold_similarity",
            "nearest_scaffold_reference_id",
        ]
    ]
    view.to_csv(output / "admitted_neighbors.csv", index=False)
    html = (
        """<!doctype html><html lang="zh"><meta charset="utf-8"><title>候选骨架检查</title>
    <style>body{max-width:1200px;margin:40px auto;padding:0 24px;font:17px/1.6 system-ui;
    color:#23334a;background:#f5f7fa}table{font-size:13px;border-collapse:collapse;background:white;
    width:100%;overflow-wrap:anywhere}td,th{padding:8px;border:1px solid #dde4ed}
    img{width:100%;background:white}h1{font-size:30px}section{margin:32px 0}</style>
    <h1>十项目录候选：与已取得GPR84结构的比较</h1>
    <p>2026-10-02 · 结构证据，活性与功能方向未知。</p>
    <p>所有参照范围均无连接图、精确Murcko骨架或通用框架匹配。通用框架将原子改为C、
    键改单键。此结论限于当前结构登记，不表示全球查新或已证明筛选有效。</p>
    <p>未指定立体不影响本次无立体比较。不同精确骨架或通用框架仍可含有明显共享核心；
    候选自身已有其他靶点实验背景。ChEMBL、BindingDB及旧登记不自动视为拮抗剂集合。</p>
    <section><h2>分范围清点</h2>"""
        + summary.to_html(index=False)
        + """</section>
    <section><h2>准入拮抗实验中的最近邻</h2>"""
        + view.to_html(index=False)
        + """</section>
    <section><h2>左：目录候选；右：最近准入结构</h2>
    <img src="candidate_known_neighbors.png"></section>
    <p><a href="../candidate_comparisons.csv">四范围完整比较</a> ·
    <a href="../candidate_pairs.csv">45对候选关系</a> ·
    <a href="independent_readback.json">独立回读</a></p></html>"""
    )
    (output / "index.html").write_text(html, encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == "__main__":
    main()
