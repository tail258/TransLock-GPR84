"""连接既有十项提名、结构/药理和MD证据；保持原顺序，不生成综合评分。"""

import argparse
import html
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value):
    return html.escape(str(value))


def source_entry(path):
    s = path.stat()
    return {"path": str(path), "bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--md-analysis", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root / "artifacts/track3_rebuild"
    screen = root / "screening_application_2026-10-01_v1"
    public = root / "candidate_public_validation_2026-10-02_v1"
    scaffold = root / "candidate_scaffold_audit_2026-10-02_v1"
    poses = root / "candidate_pose_hypotheses_2026-10-02_v1"
    names = {
        "candidates": screen / "candidate_hypotheses.csv",
        "neighbors": screen / "candidate_support_measurements.csv",
        "scaffolds": scaffold / "candidate_comparisons.csv",
        "poses": poses / "analysis_v1/seed_variations.csv",
        "public": public / "candidate_public_evidence.csv",
        "public_full": public / "candidates_evidence_v2.json",
        "measurements": public / "chembl_candidate_measurements.csv",
        "md": args.md_analysis / "summaries.json",
        "config": args.config,
        "contact_summary": args.md_analysis / "figures/contact_summary.json",
    }
    snapshots = {k: source_entry(p) for k, p in names.items()}
    candidates = pd.read_csv(names["candidates"]).sort_values("nomination_order")
    structures = pd.read_csv(names["scaffolds"])
    variations = pd.read_csv(names["poses"])
    pub = pd.read_csv(names["public"]).set_index("candidate_id")
    public_full = {r["candidate_id"]: r for r in read_json(names["public_full"])}
    neighbors = pd.read_csv(names["neighbors"])
    md = read_json(names["md"])
    contact_summary = read_json(names["contact_summary"])
    config = read_json(names["config"])
    ids = candidates.candidate_id.tolist()
    assert len(ids) == len(set(ids)) == 10 and candidates.nomination_order.tolist() == list(
        range(1, 11)
    )
    assert set(ids) == set(pub.index) == set(public_full) == set(config["candidate_notes"])
    assert len(md) == 9 and len({(r["ligand_id"], r["seed"]) for r in md}) == 9
    assert set(r["ligand_id"] for r in md) == {"GLPG1205", "CHEMBL120691", "CHEMBL1086046"}
    args.output.mkdir(parents=True, exist_ok=False)
    records, details, state_rows = [], [], []
    for _, candidate in candidates.iterrows():
        cid = candidate.candidate_id
        scope = structures[structures.candidate_id == cid]
        assert len(scope) == scope.scope.nunique() == 4
        assert all(
            scope[field].sum() == 0
            for field in [
                "connection_match_rows",
                "scaffold_match_rows",
                "generic_framework_match_rows",
            ]
        )
        pose = variations[variations.candidate_id == cid].sort_values("state_id")
        stereo_count = int(candidate.unspecified_tetrahedral_centers)
        assert len(pose) == (2 if stereo_count else 1)
        assert (
            pub.loc[
                cid,
                ["gpr84_chembl_rows", "gpr84_pubchem_rows", "bindingdb_gpr84_connection_matches"],
            ].sum()
            == 0
        )
        supporting = neighbors[neighbors.candidate_id == cid]
        assert len(supporting) == 3 and set(supporting.neighbor_role) == {"any", "strong", "weak"}
        repeats = [r for r in md if r["ligand_id"] == cid]
        assert len(repeats) in [0, 3]
        note = config["candidate_notes"][cid]
        record = {
            "nomination_order": int(candidate.nomination_order),
            "candidate_id": cid,
            "smiles": candidate.smiles,
            "source_url": candidate.source_url,
            **{k: int(candidate[k]) for k in ["rank_20261001", "rank_20261002", "rank_20261003"]},
            "nearest_training_similarity": float(candidate.nearest_any_similarity),
            "known_reference_scopes": 4,
            "exact_Murcko_matches_in_each_scope": 0,
            "exact_generic_framework_matches_in_each_scope": 0,
            "unspecified_tetrahedral_centers": stereo_count,
            "conditional_pose_states": len(pose),
            "state_pose_RMSDs_A": "; ".join(
                f"{r.state_id}:{r.seed_pose_rmsd_A:.6f}" for r in pose.itertuples()
            ),
            "state_contact_Jaccards": "; ".join(
                f"{r.state_id}:{r.contact_jaccard:.6f}" for r in pose.itertuples()
            ),
            "public_background": note["background"],
            "chembl_measurement_rows": int(pub.loc[cid, "chembl_measurement_rows"]),
            "pubchem_measurement_rows": int(pub.loc[cid, "pubchem_measurement_rows"]),
            "public_GPR84_status": "在已核查范围未找到实测匹配；活性及功能方向未知",
            "public_identity_note": pub.loc[cid, "pubchem_identity_note"],
            "MD_status": "已完成3×5ns" if repeats else "未做此次MD",
            "MD_replicates": len(repeats),
            "MD_total_ns": len(repeats) * 5,
            "next_work": note["next_work"],
            "specific_limit": note["scope"],
        }
        if repeats:
            for label, metric in [
                ("MD_original_RMSD", "original_total_A"),
                ("MD_start_RMSD", "replica_start_total_A"),
                ("MD_centroid_shift", "original_centroid_shift_A"),
                ("MD_core_self_fit", "original_core_self_fit_internal_A"),
            ]:
                vals = [r["metrics"][metric]["median"] for r in repeats]
                record[label + "_median_min_A"] = min(vals)
                record[label + "_median_max_A"] = max(vals)
            record["MD_common_80pct_contacts"] = ";".join(
                contact_summary[cid]["common_80pct_all_three"]
            )
        records.append(record)
        details.append(
            {
                "candidate": record,
                "pose_states": pose.to_dict("records"),
                "reference_scopes": scope.to_dict("records"),
                "public_source": public_full[cid],
                "training_neighbors": supporting.to_dict("records"),
                "md_replicate_sources": [
                    {"seed": r["seed"], "source_run": r["source_run"]} for r in repeats
                ],
            }
        )
        state_rows.extend(pose.to_dict("records"))
    table = pd.DataFrame(records)
    table.to_csv(args.output / "candidate_evidence.csv", index=False)
    pd.DataFrame(state_rows).to_csv(args.output / "all_candidate_pose_states.csv", index=False)
    (args.output / "candidate_evidence.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copyfile(names["measurements"], args.output / "public_measurements_source.csv")
    shutil.copyfile(names["neighbors"], args.output / "training_neighbor_measurements.csv")
    shutil.copyfile(args.config, args.output / "interpretation_config.json")
    mols = [Chem.MolFromSmiles(r["smiles"]) for r in records]
    assert all(m is not None for m in mols)
    legends = [
        f"{r['nomination_order']}. {r['candidate_id']}\n"
        f"ML rank {r['rank_20261001']} | MD repeats {r['MD_replicates']}\n"
        f"Unassigned tetrahedral centers: {r['unspecified_tetrahedral_centers']}"
        for r in records
    ]
    grid = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(650, 390), legends=legends)
    grid.save(str(args.output / "ten_candidate_structures.png"))
    svg = Draw.MolsToGridImage(
        mols, molsPerRow=2, subImgSize=(650, 390), legends=legends, useSVG=True
    )
    (args.output / "ten_candidate_structures.svg").write_text(svg, encoding="utf-8")
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 7), sharey=True, gridspec_kw={"width_ratios": [1.6, 1.5, 1]}
    )
    y = np.arange(10)
    for index, (seed, color) in enumerate(
        [(20261001, "#2878b5"), (20261002, "#d77a2b"), (20261003, "#239474")]
    ):
        axes[0].scatter(
            table[f"rank_{seed}"], y + (index - 1) * 0.15, label=str(seed), color=color, s=34
        )
    axes[0].set_yticks(y, [f"{r['nomination_order']:02d}  {r['candidate_id']}" for r in records])
    axes[0].set(
        xlabel="Rank among 82,883 scored graphs",
        title="Model-seed ranks (lower is earlier)",
        xlim=(-10, 590),
    )
    axes[0].legend(fontsize=8, loc="lower right")
    for i, cid in enumerate(ids):
        v = variations[variations.candidate_id == cid].sort_values("state_id")
        for j, row in enumerate(v.itertuples()):
            axes[1].scatter(
                row.seed_pose_rmsd_A,
                i + (j - (len(v) - 1) / 2) * 0.18,
                color=["#2878b5", "#8058a5"][j],
                marker=["o", "s"][j],
                s=42,
            )
            axes[1].annotate(
                f"{row.seed_pose_rmsd_A:.2f}",
                (row.seed_pose_rmsd_A, i + (j - (len(v) - 1) / 2) * 0.18),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
        axes[2].text(
            0.05,
            i,
            "3 × 5 ns" if records[i]["MD_replicates"] else "Not simulated",
            va="center",
            fontsize=10,
        )
        if records[i]["unspecified_tetrahedral_centers"]:
            for ax in axes:
                ax.axhspan(i - 0.42, i + 0.42, color="#f5ead7", zorder=-1)
    axes[1].set(
        xlabel="Two docking seeds: pose RMSD (Å)",
        title="Each conditional stereochemical state",
        xlim=(-0.4, 12.6),
    )
    axes[2].set(title="Existing MD coverage", xticks=[], xlim=(0, 1))
    for ax in axes:
        ax.set_ylim(9.6, -0.6)
        ax.grid(axis="y", alpha=0.12)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Ten nominated candidates | original nomination order retained", fontsize=15)
    fig.text(
        0.5,
        0.012,
        "Shaded rows: one unassigned tetrahedral center. Circles/squares: s01/s02. "
        "No measured GPR84 activity match in checked scope; no composite score.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(args.output / f"candidate_evidence_map.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    compact = pd.DataFrame(
        [
            {
                "提名顺序": r["nomination_order"],
                "候选": r["candidate_id"],
                "三种子排名": f"{r['rank_20261001']} / {r['rank_20261002']} / {r['rank_20261003']}",
                "最近训练相似度": f"{r['nearest_training_similarity']:.3f}",
                "未指定手性中心": r["unspecified_tetrahedral_centers"],
                "MD": r["MD_status"],
                "已有药理背景": r["public_background"],
                "建议补充工作": r["next_work"],
            }
            for r in records
        ]
    )
    articles = []
    for r, detail in zip(records, details, strict=True):
        state_table = pd.DataFrame(detail["pose_states"])[
            ["state_id", "seed_pose_rmsd_A", "contact_jaccard"]
        ]
        articles.append(
            f"<details id='{r['candidate_id']}'><summary>{r['nomination_order']}. "
            f"{r['candidate_id']} · {esc(r['MD_status'])}</summary>"
            f"<p><b>药理背景：</b>{esc(r['public_background'])}。"
            f"ChEMBL {r['chembl_measurement_rows']}条、PubChem {r['pubchem_measurement_rows']}条；"
            "有转录重叠，不相加为独立实验数。</p>"
            f"<p><b>身份：</b>{esc(r['public_identity_note'])}。<b>具体限制：</b>{esc(r['specific_limit'])}。</p>"
            f"<p><b>下一步：</b>{esc(r['next_work'])}。</p>"
            + state_table.to_html(index=False, border=0, float_format=lambda x: f"{x:.3f}")
            + f"<p><a href='{esc(r['source_url'])}'>ChEMBL原目录</a> · "
            "<a href='https://www.ebi.ac.uk/chembl/api/data/activity.json?"
            f"molecule_chembl_id={r['candidate_id']}&amp;limit=1000'>"
            "ChEMBL测量出处</a></p></details>"
        )
    relative_md = Path(
        os.path.relpath(args.md_analysis / "figures/index.html", args.output)
    ).as_posix()
    page = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>十项候选证据总览</title>
    <style>body{max-width:1320px;margin:36px auto;padding:0 24px;font:16px/1.65 system-ui;
    color:#243649;background:#fafbfc}
    h1,h2{color:#163a54}a{color:#166598}img{max-width:100%;background:white}table{border-collapse:collapse;width:100%;font-size:14px;background:white}
    th,td{padding:10px;border-bottom:1px solid #d9e3ea;text-align:left;vertical-align:top}
    th{background:#eaf1f6}details{padding:16px;margin:12px 0;border:1px solid #d9e3ea;
    background:white}
    summary{font-weight:650;cursor:pointer}.note{background:#eaf2f7;padding:18px}.scroll{overflow-x:auto}</style>
    <h1>十项候选：筛选、结构、公开药理与MD证据</h1>
    <p>2026-10-03 · 保留原提名顺序 · 只整合既有结果和已完成轨迹</p>
    <p class="note">10项候选来自82,883个评分分子图，活性与功能方向待实验验证。
    MD覆盖2项候选和1个GLPG1205已知对照，每个3次，共9条；其余候选未开展MD。</p>"""
    page += (
        f"<p><a href='{relative_md}'>打开九条MD结构解释与三组结果图</a> · "
        "<a href='candidate_evidence.csv'>下载完整十项证据CSV</a> · "
        "<a href='candidate_evidence.json'>逐项来源JSON</a></p>"
    )
    page += (
        "<h2>证据地图</h2><p>左侧为三次模型训练种子的目录排名；"
        "中间为两次对接的姿势差异，各立体状态单独显示；右侧为MD覆盖。"
        "浅色背景提示未指定手性。</p><img src='candidate_evidence_map.png'>"
    )
    page += (
        "<p><a href='candidate_evidence_map.pdf'>PDF</a> · "
        "<a href='candidate_evidence_map.svg'>SVG</a></p>"
        "<h2>十项并列比较</h2><div class='scroll'>"
        + compact.to_html(index=False, border=0)
        + "</div>"
    )
    page += (
        "<p>四类已取得GPR84参照中，十项均无相同Murcko骨架/通用框架匹配；"
        "比较范围为上述四类参照。"
        "最近训练相似度仅为连续描述，未经可靠性校准。</p>"
    )
    page += "<h2>每项证据与待解决问题</h2>" + "".join(articles)
    page += (
        "<h2>分子结构</h2><img src='ten_candidate_structures.png'>"
        "<p><a href='ten_candidate_structures.svg'>分子结构SVG</a></p>"
    )
    page += (
        "<h2>候选后续研究</h2>"
        "<p>CHEMBL120691与CHEMBL1086046已有条件性短程姿势证据；"
        "后续分析重点为位移和接触变化。"
        "CHEMBL121380可作为扩展备选，但与已有代表有相关背景，新增信息量需说明。"
        "其他项先按各自的立体/姿势缺口处理。</p>"
    )
    page += (
        "<p>原主种子的回顾EF=1.062，相似性参照=1.083；"
        "三种子微调平均EF为1.112。"
        "模型回顾评价与MD结构分析分别报告。"
        "本页整合已有评分、对接及MD记录。</p>"
    )
    page += (
        "<p><a href='public_measurements_source.csv'>41条其他终点原测量</a> · "
        "<a href='training_neighbor_measurements.csv'>30行训练近邻及论文DOI</a> · "
        "<a href='source_manifest.json'>来源文件与处理记录</a></p></html>"
    )
    (args.output / "index.html").write_text(page, encoding="utf-8")
    for key, path in names.items():
        assert source_entry(path) == snapshots[key]
    manifest = {
        "status": "complete_evidence_integration",
        "created_utc": datetime.now(UTC).isoformat(),
        "candidates": 10,
        "preserved_nomination_order": ids,
        "MD_candidate_count": 2,
        "unassigned_stereo_candidates": int((table.unspecified_tetrahedral_centers > 0).sum()),
        "pose_states": len(state_rows),
        "public_GPR84_matches_in_checked_scope": 0,
        "composite_score_created": False,
        "sources": snapshots,
        "hashes": False,
    }
    (args.output / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copyfile(__file__, args.output / "source_snapshot.py")
    print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
