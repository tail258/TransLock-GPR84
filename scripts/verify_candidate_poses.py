"""从磁盘独立核对复合物身份、坐标、接触与姿势重复差异。"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.spatial import cKDTree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load = lambda name: json.loads((args.run / name).read_text())  # noqa: E731
    manifest = load("manifest.json")
    if manifest["status"] not in {"complete", "completed_with_recorded_failures"}:
        raise ValueError("尚未完成的运行不能验收")
    jobs, states, exported = load("jobs.json"), load("states.json"), load("exported_complexes.json")
    assert len(jobs) == manifest["actual_jobs"]
    assert len({j["job_id"] for j in jobs}) == len(jobs)
    expected_jobs = {
        f"{s['state_id']}_{seed}"
        for s in states
        if s["status"] == "prepared"
        for seed in manifest["config"]["docking_seeds"]
    }
    assert {j["job_id"] for j in jobs} == expected_jobs
    by_job = {j["job_id"]: j for j in jobs}
    receptor_lines = [
        line
        for line in (args.run / "receptor.pdb").read_text().splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
    ]
    heavy_lines = [line for line in receptor_lines if line[76:78].strip() not in {"H", "D"}]
    xyz = np.array(
        [[float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))] for line in heavy_lines]
    )
    labels = [f"{line[21]}:{line[17:20].strip()}{int(line[22:26])}" for line in heavy_lines]
    tree = cKDTree(xyz)
    rows = []
    for record in exported:
        mol = Chem.SDMolSupplier(str(args.run / record["sdf"]), removeHs=False)[0]
        expected_key = Chem.MolToInchiKey(Chem.MolFromSmiles(record["smiles"]))
        assert mol is not None and mol.GetConformer().Is3D()
        assert np.isfinite(mol.GetConformer().GetPositions()).all()
        assert Chem.MolToInchiKey(Chem.RemoveHs(mol)) == expected_key
        stereo = Chem.Mol(mol)
        Chem.RemoveStereochemistry(stereo)
        Chem.AssignStereochemistryFrom3D(stereo, replaceExistingTags=True)
        assert Chem.MolToInchiKey(Chem.RemoveHs(stereo)) == expected_key
        pdb_lines = [
            line
            for line in (args.run / record["pdb"]).read_text().splitlines()
            if line.startswith(("ATOM  ", "HETATM"))
        ]
        assert pdb_lines[: len(receptor_lines)] == receptor_lines
        ligand_lines = pdb_lines[len(receptor_lines) :]
        ligand_xyz = np.array(
            [[float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))] for line in ligand_lines]
        )
        assert np.allclose(ligand_xyz, mol.GetConformer().GetPositions(), atol=0.000501, rtol=0)
        assert len(ligand_lines) == mol.GetNumAtoms()
        heavy = Chem.RemoveHs(mol).GetConformer().GetPositions()
        distances, _ = tree.query(heavy)
        minimum = float(distances.min())
        contacts = {}
        for point in heavy:
            for index in tree.query_ball_point(point, r=4.0):
                distance = float(np.linalg.norm(point - xyz[index]))
                contacts[labels[index]] = min(distance, contacts.get(labels[index], float("inf")))
        assert set(contacts) == set(record["contacts_4A"])
        assert all(
            abs(value - record["contacts_4A"][key]) < 2e-5 for key, value in contacts.items()
        )
        job = by_job[record["job_id"]]
        assert job["selected_rank"] == record["pose_rank"]
        audit = job["audits"][record["pose_rank"] - 1]
        assert abs(minimum - audit["minimum_receptor_distance_A"]) < 2e-5
        assert minimum >= manifest["config"]["minimum_receptor_heavy_distance_A"]
        rows.append(
            {
                "job_id": record["job_id"],
                "candidate_id": record["candidate_id"],
                "state_id": record["state_id"],
                "seed": record["seed"],
                "selected_rank": record["pose_rank"],
                "minimum_receptor_distance_A": minimum,
                "contact_count": len(contacts),
                "contacts": ";".join(sorted(contacts)),
                "sdf": record["sdf"],
                "pdb": record["pdb"],
            }
        )
    variations = load("seed_variations.json")
    for variation in variations:
        pair = [e for e in exported if e["state_id"] == variation["state_id"]]
        a, b = [Chem.SDMolSupplier(str(args.run / e["sdf"]), removeHs=True)[0] for e in pair]
        xa, xb = a.GetConformer().GetPositions(), b.GetConformer().GetPositions()
        mappings = b.GetSubstructMatches(a, uniquify=False, useChirality=False, maxMatches=1000000)
        rmsd = min(float(np.sqrt(np.square(xa - xb[list(m)]).sum(axis=1).mean())) for m in mappings)
        assert abs(rmsd - variation["seed_pose_rmsd_A"]) < 1e-6
        sa, sb = set(pair[0]["contacts_4A"]), set(pair[1]["contacts_4A"])
        assert abs(len(sa & sb) / len(sa | sb) - variation["contact_jaccard"]) < 1e-12
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), args.output / "source_snapshot.py")
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "complex_checks.csv", index=False)
    variation_frame = pd.DataFrame(variations)
    variation_frame.to_csv(args.output / "seed_variations.csv", index=False)
    summary = {
        "status": "passed",
        "jobs": len(jobs),
        "complexes": len(exported),
        "candidate_complexes": sum(e["candidate_id"] != "GLPG1205" for e in exported),
        "state_pairs_checked": len(variations),
        "checks": [
            "固定状态×种子成员",
            "记录及坐标推断身份",
            "受体记录与配体坐标保持",
            "KDTree独立残基接触及距离",
            "原子图映射直接坐标RMSD（不叠合）",
        ],
        "limits": "姿势文件的技术验收；不验证亲和力或功能方向",
    }
    (args.output / "independent_readback.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.4, 1]})
    order = variation_frame.state_id.tolist()
    axes[0].barh(order, variation_frame.seed_pose_rmsd_A, color="#406e8e")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Symmetry RMSD in receptor frame (Angstrom)")
    axes[0].set_title("Two seeds: position and conformation differences")
    axes[1].barh(order, variation_frame.contact_jaccard, color="#55a891")
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[1].set_xlabel("Jaccard of residues within 4 Angstrom")
    axes[1].set_title("Two seeds: contact overlap")
    fig.suptitle(
        "Directory candidate pose hypotheses | no activity or direction inference", fontsize=15
    )
    fig.tight_layout()
    for extension in ("png", "svg", "pdf"):
        fig.savefig(args.output / ("pose_seed_variation." + extension), dpi=200)
    plt.close(fig)
    html = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>候选复合物假设</title>
    <style>body{max-width:1300px;margin:40px auto;padding:0 24px;font:17px/1.6 system-ui;
    color:#23334a;background:#f5f7fa}table{font-size:13px;border-collapse:collapse;background:white;
    width:100%}td,th{padding:8px;border:1px solid #dde4ed}img{width:100%}a{color:#215f95}</style>
    <h1>十项目录候选的GPR84复合物假设</h1><p>2026-10-02 · 9WIJ来源受体 · 条件性中性输入</p>
    <p>GLPG1205两种子控制先通过，随后运行固定名单。每状态保留两个种子的首个合格姿势。
    四项未指定中心分别生成两个条件性构型，实验构型与混合比例未知。</p>
    <p>下图的RMSD与接触重叠描述对接姿势的重复性。
    没有按对接分数重排候选，也没有运行膜MD。</p><img src="pose_seed_variation.png">
    <h2>结构文件与技术检查</h2>"""
    display = frame[
        [
            "candidate_id",
            "state_id",
            "seed",
            "selected_rank",
            "minimum_receptor_distance_A",
            "contact_count",
        ]
    ].copy()
    display["PDB"] = [f'<a href="../{p}">PDB</a>' for p in frame.pdb]
    display["SDF"] = [f'<a href="../{p}">SDF</a>' for p in frame.sdf]
    html += (
        display.to_html(index=False, escape=False)
        + """<p>
    <a href="../control_checks.json">GLPG控制</a> · <a href="complex_checks.csv">接触和距离表</a> ·
    <a href="independent_readback.json">独立回读</a></p></html>"""
    )
    (args.output / "index.html").write_text(html, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
