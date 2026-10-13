"""独立核对筛选清点、排名、候选身份与近邻；生成应用图和分子结构图。"""

from __future__ import annotations

import argparse
import html
import importlib.metadata
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem, DataStructs  # noqa: E402
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors  # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D  # noqa: E402
from rdkit.Chem.MolStandardize import rdMolStandardize  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402

SEEDS = [20261001, 20261002, 20261003]


def verify(args):
    raw = pd.read_csv(args.catalogue / "structures.csv")
    all_rows = pd.read_csv(args.run / "all_records.csv", float_precision="round_trip")
    scored = pd.read_csv(args.run / "scored_records.csv", float_precision="round_trip")
    candidates = pd.read_csv(args.run / "candidate_hypotheses.csv", float_precision="round_trip")
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    assert len(raw) == len(all_rows) == summary["input_records"] == 100000
    assert raw.molecule_chembl_id.tolist() == all_rows.candidate_id.tolist()
    assert not all_rows.candidate_id.duplicated().any()
    assert not scored.smiles.duplicated().any() and not scored.candidate_id.duplicated().any()
    assert len(scored) == summary["scored_graphs"]
    assert all_rows.status.value_counts().to_dict() == summary["status_counts"]
    included = all_rows[all_rows.status.eq("scored_candidate_hypothesis_only")]
    assert included.candidate_id.tolist() == scored.candidate_id.tolist()
    known_ids = set(pd.read_csv(args.run / "excluded_known_ids.csv").known_id)
    known_connections = set(pd.read_csv(args.run / "excluded_connections.csv").connection_id)
    assert not set(scored.candidate_id) & known_ids
    assert not set(scored.connection_id) & known_connections
    assert scored.heavy_atoms.between(15, 43).all()
    assert scored.gpr84_activity.eq("unknown_in_checked_scope").all()
    assert scored.functional_direction.eq("unestablished").all()
    for seed in SEEDS:
        scores = scored[f"score_{seed}"].to_numpy()
        assert np.isfinite(scores).all()
        order = np.lexsort((scored.candidate_id.to_numpy(), -scores))
        ranks = np.empty(len(scored), dtype=int)
        ranks[order] = np.arange(1, len(scored) + 1)
        np.testing.assert_array_equal(ranks, scored[f"rank_{seed}"])
        np.testing.assert_array_equal(included[f"score_{seed}"], scores)
    order = np.lexsort(
        (scored.candidate_id.to_numpy(), -scored.nearest_strong_similarity.to_numpy())
    )
    ranks = np.empty(len(scored), dtype=int)
    ranks[order] = np.arange(1, len(scored) + 1)
    np.testing.assert_array_equal(ranks, scored.similarity_reference_rank)
    matrix = scored[[f"rank_{seed}" for seed in SEEDS]].to_numpy()
    np.testing.assert_array_equal(scored.seed_rank_min, matrix.min(axis=1))
    np.testing.assert_array_equal(scored.seed_rank_max, matrix.max(axis=1))
    budget = min(1000, math.ceil(0.01 * len(scored)))
    pool = scored[scored.rank_20261001 <= budget].sort_values("rank_20261001")
    expected = pool.assign(scaffold=pool.scaffold.fillna("")).drop_duplicates("scaffold").head(10)
    assert expected.candidate_id.tolist() == candidates.candidate_id.tolist()
    assert (
        len(candidates) == summary["candidate_hypotheses"] and budget == summary["candidate_pool"]
    )
    units = pd.read_csv(args.models / "training_members.csv", dtype={"activity_ids": "string"})
    bridge_path = next(
        Path(path) for path in summary["input_stats"] if Path(path).name == "legacy_identity.csv"
    )
    policy_path = next(
        Path(path)
        for path in summary["input_stats"]
        if Path(path).name == "manifest.json" and Path(path).parent.name.startswith("pilot_")
    )
    bridge = pd.read_csv(bridge_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))["expanded_policy"]
    current = pd.read_csv(args.catalogue / "known_target_molecules.csv")
    activity_ids = set(
        pd.read_csv(args.catalogue / "known_target_activities.csv").molecule_chembl_id.dropna()
    )
    expected_ids = (
        set(bridge.molecule_chembl_id.dropna())
        | set(policy["excluded_molecule_ids"])
        | set(current.molecule_chembl_id.dropna())
        | activity_ids
    )
    expected_connections = set()
    source_smiles = [
        *bridge.standardized_smiles.dropna(),
        *policy["excluded_alias_smiles"],
        *units.smiles,
        *current.canonical_smiles.dropna(),
    ]
    for smiles in dict.fromkeys(source_smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        Chem.SanitizeMol(mol)
        Chem.RemoveStereochemistry(mol)
        expected_connections.add(Chem.MolToSmiles(mol, isomericSmiles=False))
    assert expected_ids == known_ids and expected_connections == known_connections
    registry = json.loads((args.run / "known_registry_scope.json").read_text(encoding="utf-8"))
    assert (
        registry["current_missing_structure_ids"]
        == current.loc[current.canonical_smiles.isna(), "molecule_chembl_id"].tolist()
    )
    assert registry["activity_ids_without_structure_row"] == sorted(
        activity_ids - set(current.molecule_chembl_id)
    )
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, includeChirality=True)
    fps = [generator.GetSparseCountFingerprint(Chem.MolFromSmiles(s)) for s in units.smiles]
    sdf = list(Chem.SDMolSupplier(str(args.run / "candidate_hypotheses_2d.sdf"), removeHs=True))
    assert len(sdf) == len(candidates)
    expected_support = []
    for (_, row), stored in zip(candidates.iterrows(), sdf, strict=True):
        assert stored is not None and stored.GetProp("_Name") == row.candidate_id
        assert not stored.GetConformer().Is3D()
        mol = Chem.MolFromSmiles(row.source_smiles)
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        Chem.SanitizeMol(mol)
        assert Chem.MolToSmiles(mol, isomericSmiles=True) == row.smiles
        assert Chem.MolToSmiles(stored, isomericSmiles=True) == row.smiles
        bare = Chem.Mol(mol)
        Chem.RemoveStereochemistry(bare)
        assert Chem.MolToSmiles(bare, isomericSmiles=False) == row.connection_id
        assert mol.GetNumHeavyAtoms() == row.heavy_atoms
        assert {a.GetSymbol() for a in mol.GetAtoms()} <= {
            "H",
            "Br",
            "C",
            "Cl",
            "F",
            "N",
            "O",
            "P",
            "S",
        }
        assert all(a.GetNumRadicalElectrons() == 0 for a in mol.GetAtoms())
        for actual, expected_value in [
            (Descriptors.MolWt(mol), row.molecular_weight),
            (Crippen.MolLogP(mol), row.clogp),
            (rdMolDescriptors.CalcTPSA(mol), row.tpsa),
        ]:
            assert abs(actual - expected_value) < 1e-10
        scaffold = MurckoScaffold.GetScaffoldForMol(bare)
        assert Chem.MolToSmiles(scaffold, isomericSmiles=False) == (
            row.scaffold if pd.notna(row.scaffold) else ""
        )
        distances = np.array(
            [
                DataStructs.TanimotoSimilarity(generator.GetSparseCountFingerprint(mol), fp)
                for fp in fps
            ]
        )
        for name, indices in [
            ("any", np.arange(len(units))),
            ("strong", np.flatnonzero(units.label.eq(1))),
            ("weak", np.flatnonzero(units.label.eq(0))),
        ]:
            index = int(indices[np.argmax(distances[indices])])
            assert abs(distances[index] - row[f"nearest_{name}_similarity"]) < 1e-12
            assert units.iloc[index].unit_id == row[f"nearest_{name}_unit_id"]
            expected_support.append(
                {
                    "candidate_id": row.candidate_id,
                    "neighbor_role": name,
                    "similarity": distances[index],
                    **units.iloc[index].to_dict(),
                }
            )
    exported_support = pd.read_csv(
        args.run / "candidate_support_measurements.csv",
        dtype={"activity_ids": "string"},
        float_precision="round_trip",
    )
    pd.testing.assert_frame_equal(
        exported_support,
        pd.DataFrame(expected_support),
        check_dtype=False,
        rtol=0,
        atol=1e-12,
    )
    verified = {
        "status": "pass",
        "input_records": len(all_rows),
        "scored_graphs": len(scored),
        "all_three_seed_rankings_recomputed": True,
        "similarity_ranking_recomputed": True,
        "nomination_rule_rebuilt": True,
        "candidate_2d_sdf_identity_checked": len(sdf),
        "candidate_properties_and_nearest_neighbors_recomputed": len(candidates),
        "exported_support_rows_checked": len(exported_support),
        "exclusion_keys_rebuilt_from_bridge_policy_and_current_target": True,
        "scope": "全表清点/排除键/排序核验；候选逐项结构和近邻重算；"
        "不冒称独立重算全部模型预测或全部原始图标准化",
    }
    return summary, all_rows, scored, candidates, verified


def draw_structures(candidates, output):
    molecules = [Chem.MolFromSmiles(s) for s in candidates.smiles]
    legends = [
        f"{row.nomination_order}. {row.candidate_id}\nPrimary rank {row.rank_20261001}; "
        f"nearest training similarity {row.nearest_any_similarity:.2f}\n"
        "Activity and functional direction unknown"
        for row in candidates.itertuples()
    ]
    rows = max(1, math.ceil(len(molecules) / 2))
    for backend, extension in [
        (rdMolDraw2D.MolDraw2DSVG, "svg"),
        (rdMolDraw2D.MolDraw2DCairo, "png"),
    ]:
        drawer = backend(2400, rows * 600, 1200, 600)
        drawer.drawOptions().legendFontSize = 30
        drawer.drawOptions().legendFraction = 0.18
        drawer.drawOptions().padding = 0.06
        drawer.DrawMolecules(molecules, legends=legends)
        drawer.FinishDrawing()
        data = drawer.GetDrawingText()
        path = output / f"candidate_structures.{extension}"
        if isinstance(data, bytes):
            path.write_bytes(data)
        else:
            path.write_text(data, encoding="utf-8")


def make_reader(args, summary, candidates):
    """把固定名单与两类公开测量近邻并排展示，页面不再筛选或重新排序。"""
    support = pd.read_csv(
        args.run / "candidate_support_measurements.csv", dtype={"activity_ids": "string"}
    )

    def safe(value):
        return html.escape(str(value), quote=True)

    def molecule_svg(smiles):
        drawer = rdMolDraw2D.MolDraw2DSVG(540, 300)
        drawer.drawOptions().padding = 0.08
        drawer.DrawMolecule(Chem.MolFromSmiles(smiles))
        drawer.FinishDrawing()
        return drawer.GetDrawingText().split("?>")[-1]

    def neighbor(row, role):
        item = support[
            support.candidate_id.eq(row.candidate_id) & support.neighbor_role.eq(role)
        ].iloc[0]
        title = "强效类近邻" if role == "strong" else "弱效类近邻"
        return (
            f'<section class="neighbor"><h3>{title}</h3>{molecule_svg(item.smiles)}'
            f"<p>r3 计数相似度 <b>{item.similarity:.3f}</b></p>"
            f"<p>原分子 ID：{safe(item.molecule_ids)}<br>"
            f"测量 ID：{safe(item.activity_ids)}<br>"
            f'<a href="https://www.ebi.ac.uk/chembl/explore/assay/{safe(item.assay_id)}">'
            f"{safe(item.assay_id)}</a><br>"
            f'<a href="https://doi.org/{safe(item.document_doi)}">'
            f"{safe(item.document_doi)}</a></p></section>"
        )

    cards = []
    for row in candidates.itertuples():
        cards.append(
            f"<article><header><h2>{row.nomination_order:02d} · "
            f'<a href="{safe(row.source_url)}">{safe(row.candidate_id)}</a></h2>'
            f'<span class="badge">候选假设 · 活性及功能方向未知</span></header>'
            '<div class="molecules"><section><h3>库内候选结构</h3>'
            f"{molecule_svg(row.smiles)}"
            f"<p>主排名 <b>{row.rank_20261001:,}</b> / {summary['scored_graphs']:,}<br>"
            f"其他种子排名：{row.rank_20261002:,}、{row.rank_20261003:,}<br>"
            f"分子量 {row.molecular_weight:.1f} · cLogP {row.clogp:.2f} · "
            f"TPSA {row.tpsa:.1f} Å²<br>"
            f"未指定四面体中心：{row.unspecified_tetrahedral_centers}</p></section>"
            f"{neighbor(row, 'strong')}{neighbor(row, 'weak')}</div></article>"
        )
    document = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TransLock-GPR84 · 筛选应用</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f4f3;color:#24323a;
font:16px/1.7 'Segoe UI','Microsoft YaHei',sans-serif}main{max-width:1500px;
margin:auto;padding:40px 30px}h1{font-size:38px;line-height:1.25;margin:12px 0}
h2{font-size:24px;margin:0}h3{font-size:16px;margin:0}a{color:#2c667a;
text-decoration:none}a:hover{text-decoration:underline}.eyebrow{letter-spacing:2px;
color:#537783}.intro{max-width:1000px}.metrics{display:flex;gap:35px;margin:28px 0}
.metrics b{display:block;font-size:34px;color:#2c667a}.metrics span{color:#60717a}
.links{display:flex;gap:22px;flex-wrap:wrap;margin:20px 0}article{background:white;
border:1px solid #dae1e1;border-radius:12px;margin:24px 0;padding:25px}
article header{display:flex;align-items:center;justify-content:space-between;
gap:15px;flex-wrap:wrap;margin-bottom:22px}.badge{font-size:13px;background:#f6eadd;
color:#866239;padding:5px 10px;border-radius:6px}.molecules{display:grid;
grid-template-columns:repeat(3,minmax(0,1fr));gap:24px}.molecules section{min-width:0}
.molecules svg{width:100%;height:auto}.molecules p{font-size:14px;overflow-wrap:anywhere}
.neighbor{border-left:1px solid #e2e8e8;padding-left:22px}.overview{width:100%;
background:white;border:1px solid #dae1e1;border-radius:12px}.note{color:#586a74;
font-size:14px}footer{padding:20px 0;color:#586a74;font-size:13px}
@media(max-width:850px){main{padding:24px 16px}h1{font-size:28px}.molecules{
grid-template-columns:1fr}.neighbor{border-left:0;border-top:1px solid #e2e8e8;
padding:18px 0 0}.metrics{gap:20px}.metrics b{font-size:26px}}
</style><main><div class="eyebrow">TRANSLOCK–GPR84 / 2026-10-01</div>
<h1>从公开目录到可追溯候选</h1>
<p class="intro">本页展示固定规则产生的应用结果。模型在已有拮抗实验的效力标签上训练；
输出用于提出候选假设，未知分子的 GPR84 活性与功能方向待实验验证。</p>
"""
    document += (
        '<div class="metrics">'
        f"<div><b>{summary['input_records']:,}</b><span>公开目录记录</span></div>"
        f"<div><b>{summary['scored_graphs']:,}</b><span>完成评分的独立图</span></div>"
        f"<div><b>{len(candidates)}</b><span>不同骨架候选假设</span></div></div>"
        '<nav class="links"><a href="../candidate_hypotheses.csv">候选 CSV</a>'
        '<a href="../candidate_hypotheses_2d.sdf">2D 结构 SDF</a>'
        '<a href="../candidate_support_measurements.csv">近邻测量来源</a>'
        '<a href="model_card.md">模型说明</a>'
        '<a href="screening_application.pdf">应用图 PDF</a>'
        '<a href="independent_readback.json">独立核验记录</a></nav>'
        '<img class="overview" src="screening_application.svg" alt="输入清点、分数与相似度、'
        '种子排名及两类近邻相似度">'
        f'<p class="note">提名规则：首种子前 {summary["candidate_pool"]:,} 项'
        "（可评分图的前 1%，上限 1,000），依主排名取不同无立体 Bemis–Murcko 骨架，最多 10 项。"
        "其他种子和近邻只供观察，不改变名单。固定 ID 顺序截取的目录并非随机化学空间样本。</p>"
        '<p class="note">下方近邻保留所属 assay：强效类为该任务准入的 IC50 ≤ 1 μM，'
        "弱效类为对应阈值另一侧的准入测量；弱效不等于无活性。相似度是结构支持描述，"
        "没有经过可靠性阈值校准。每类仅显示按原训练顺序取到的一个最近邻。</p>"
        + "".join(cards)
        + "<footer>数据：ChEMBL 37（CC BY-SA 3.0）。计算：TransLock-GPR84。"
        "已知排除范围涵盖旧审计、隔离别名及当前 GPR84 靶点记录。"
        "三种子、结构性质和排除状态完整保存在同级原始结果表。</footer>"
        "</main></html>"
    )
    (args.output / "index.html").write_text(document, encoding="utf-8")
    card = f"""# 筛选应用模型说明

2026-10-01，Asia/Shanghai。用途：在公开分子目录中生成可追溯候选假设。

- 输入：ChEMBL 37，Small molecule，按 molecule_chembl_id 升序的前
  {summary["input_records"]:,} 条；固定非随机子集。
- 训练：285条准入测量形成277个assay图单位、248个编码图，保留assay效力语境。
  IC50≤1µM为强类，未测分子从未充当阴性。BINOP、50c/50d和37e别名继续隔离。
- 模型：CheMeleon公开预训练编码器 + LayerNorm线性头，1e-4步长、30epoch、batch16、
  图重复降权；三原种子各全量拟合一次，权重保存后预测精确回读。
- 主排序：种子20261001的原始logit降序，同分按ID升序；不是校准活性概率。
  另两模型仅描述随机种子敏感性。独立r3计数Morgan近邻排名完整保留。
- 既有验证：五主要assay、整篇留出且剔除训练连接图重叠的回顾评价。
  主种子EF约1.062，相似性参照1.083；三种子均值1.112单独报告。
  未知目录命中率待实验评价。
- 准入：标准化父体、15–43重原子、训练覆盖的元素、无自由基、单片段；
  此范围按训练结构覆盖设定，可靠性尚未校准。已知/隔离ID及连接图排除，规范立体图去重。
- 输出：{summary["scored_graphs"]:,}个评分图，固定前{summary["candidate_pool"]:,}项中按
  无立体Murcko骨架取首项，共{len(candidates)}项候选假设。
  活性、激动/拮抗方向、选择性和合成可行性未知。
- 适用限制：公开已知系列、assay条件与未知库存在分布差异；无标签预训练图重叠未核准；
  相似度和种子排名差异用于描述结构支持及种子敏感性，尚未作可靠性校准。
- 结构：SDF提供2D化学身份坐标。结构未全局查新，目录不保证可购买。
- 测速：分块构图与三模型评分{summary["inference_seconds_three_models"]:.1f}秒，处理
  {summary["scored_graphs_per_second_three_models"]:.1f}图/秒，为三模型合计速度，
  不含网络下载或结构准入。
- 来源/许可：ChEMBL数据CC BY-SA 3.0；预训练权重MIT，
  来源 https://doi.org/10.5281/zenodo.15460715 。
  依赖版本、输入来源与运行快照见父目录manifest.json、summary.json和code_snapshot。
- 验收：全表清点/排除键/三种子排序/提名独立核对；候选结构、性质及近邻逐项重算。
  核验范围见independent_readback.json。

目录候选尚无新增实测标签，前瞻命中率和筛选收益待实验评价。
"""
    (args.output / "model_card.md").write_text(card, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    for name in ["run", "models", "catalogue", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.output / Path(__file__).name)
    summary, all_rows, scored, candidates, checked = verify(args)
    checked["versions"] = {
        name: importlib.metadata.version(name) for name in ["rdkit", "pandas", "matplotlib"]
    }
    fractions = scored[
        [
            "nearest_any_similarity",
            "nearest_strong_similarity",
            "nearest_weak_similarity",
            "seed_rank_span",
        ]
    ].quantile([0, 0.1, 0.5, 0.9, 1])
    fractions.to_csv(args.output / "support_distribution.csv")
    agreement = []
    for seed in SEEDS:
        first = set(scored[scored.rank_20261001 <= summary["candidate_pool"]].candidate_id)
        other = set(scored[scored[f"rank_{seed}"] <= summary["candidate_pool"]].candidate_id)
        agreement.append(
            {
                "seed": seed,
                "pool_size": len(other),
                "overlap_with_primary": len(first & other),
                "candidate_count_in_own_top_pool": int(
                    candidates[f"rank_{seed}"].le(summary["candidate_pool"]).sum()
                ),
            }
        )
    pd.DataFrame(agreement).to_csv(args.output / "rank_stability.csv", index=False)
    candidates[
        [
            "candidate_id",
            "rank_20261001",
            "rank_20261002",
            "rank_20261003",
            "nearest_any_similarity",
            "nearest_strong_similarity",
            "nearest_weak_similarity",
            "unspecified_tetrahedral_centers",
        ]
    ].to_csv(args.output / "candidate_overview.csv", index=False)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(left=0.20, right=0.96, top=0.85, bottom=0.12, wspace=0.50, hspace=0.45)
    fig.suptitle(
        "Public-library screening: application and support",
        x=0.06,
        y=0.98,
        ha="left",
        fontsize=20,
        weight="bold",
    )
    fig.text(
        0.06,
        0.915,
        "100,000 ChEMBL records | fixed ID-order subset | 3 full-data refits | "
        "first seed defines the primary ranking\n"
        "Potency model from known antagonist assays; "
        "candidate activity and functional direction remain unknown.",
        color="#4c5660",
    )
    ax = axs[0, 0]
    counts = [
        summary["input_records"],
        summary["status_counts"].get("known_target_excluded", 0),
        summary["status_counts"].get("structure_or_scope_excluded", 0),
        summary["status_counts"].get("duplicate_excluded", 0),
        summary["scored_graphs"],
        summary["candidate_hypotheses"],
    ]
    labels = [
        "Input records",
        "Known target excluded",
        "Structure / scope excluded",
        "Duplicate graph excluded",
        "Scored unique graphs",
        "Candidate hypotheses",
    ]
    ax.barh(
        range(6), counts, color=["#a0b6c6", "#d3b4a1", "#d3b4a1", "#d3b4a1", "#457c9b", "#bd6845"]
    )
    for i, count in enumerate(counts):
        ax.text(count + 1500, i, f"{count:,}", va="center")
    ax.set_xlim(0, 120000)
    ax.set_yticks(range(6), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of records / graphs")
    ax.set_title("A  Complete input accounting", loc="left", weight="bold")
    ax = axs[0, 1]
    h = ax.hexbin(
        scored.nearest_any_similarity,
        scored.score_20261001,
        gridsize=40,
        mincnt=1,
        bins="log",
        cmap="Blues",
    )
    ax.scatter(
        candidates.nearest_any_similarity,
        candidates.score_20261001,
        s=30,
        facecolors="none",
        edgecolors="#bc563d",
    )
    ax.set_xlabel("Nearest training r3-count Tanimoto")
    ax.set_ylabel("Primary model raw logit")
    ax.set_title("B  Score versus structural support", loc="left", weight="bold")
    fig.colorbar(h, ax=ax, fraction=0.045, label="Records per bin (log scale)")
    ax = axs[1, 0]
    for offset, seed, color in zip(
        [-0.16, 0, 0.16], SEEDS, ["#bd6845", "#457c9b", "#6b9b86"], strict=True
    ):
        ax.scatter(
            candidates[f"rank_{seed}"],
            np.arange(len(candidates)) + offset,
            color=color,
            s=24,
            label=str(seed),
        )
    ax.set_yticks(range(len(candidates)), candidates.candidate_id, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.7, max(10, len(scored) * 1.2))
    ax.axvline(summary["candidate_pool"], color="#777777", ls="--", lw=1)
    ax.set_xlabel("Rank among scored graphs (log scale)")
    ax.set_title("C  Candidate ranks under each seed", loc="left", weight="bold")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax = axs[1, 1]
    for i, row in enumerate(candidates.itertuples()):
        ax.plot(
            [row.nearest_strong_similarity, row.nearest_weak_similarity], [i, i], color="#d3dadd"
        )
    ax.scatter(
        candidates.nearest_strong_similarity,
        range(len(candidates)),
        label="Nearest measured strong",
        color="#457c9b",
        s=25,
    )
    ax.scatter(
        candidates.nearest_weak_similarity,
        range(len(candidates)),
        label="Nearest measured weak",
        color="#bd6845",
        s=25,
    )
    ax.set_yticks(range(len(candidates)), candidates.candidate_id, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("r3-count Tanimoto similarity")
    ax.set_title("D  Neighbors from both potency classes", loc="left", weight="bold")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.text(
        0.06,
        0.025,
        "Selection: first-seed top 1% (cap 1,000), then first entry per "
        "non-stereo Bemis–Murcko scaffold, at most 10.\n"
        "Similarity and seed ranks describe support/sensitivity; no calibrated reliability "
        "threshold, activity probability or prospective hit rate is implied.",
        fontsize=9,
        color="#4c5660",
    )
    for extension in ["png", "svg", "pdf"]:
        fig.savefig(args.output / f"screening_application.{extension}", dpi=300)
    plt.close(fig)
    draw_structures(candidates, args.output)
    make_reader(args, summary, candidates)
    (args.output / "independent_readback.json").write_text(
        json.dumps(checked, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(checked, ensure_ascii=False))
    print(
        candidates[
            [
                "candidate_id",
                "rank_20261001",
                "rank_20261002",
                "rank_20261003",
                "nearest_any_similarity",
            ]
        ].to_string(index=False)
    )
    print(pd.DataFrame(agreement).to_string(index=False))


if __name__ == "__main__":
    main()
