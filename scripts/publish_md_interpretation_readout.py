"""将中文解读及首末窗口统计加入分析网页，并保存原网页。"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    notes = args.analysis / "reading_notes_v1"
    notes.mkdir(exist_ok=False)
    summaries = json.loads((args.analysis / "summaries.json").read_text(encoding="utf-8"))
    assert len(summaries) == 9
    records = []
    for row in summaries:
        stats = row["metrics"]["replica_start_total_A"]
        records.append(
            {
                "分子": row["ligand_id"],
                "种子": row["seed"],
                "首100帧均值 / Å": stats["first_100_mean"],
                "末100帧均值 / Å": stats["last_100_mean"],
                "最后一帧 / Å": stats["final"],
            }
        )
    table = pd.DataFrame(records).sort_values(["分子", "种子"])
    table.to_csv(notes / "late_window_comparison.csv", index=False)
    prose = (
        '<section id="final-readout"><h2>主要发现</h2>'
        "<p><strong>CHEMBL120691主要是整体位置调整，核心也随之移动；"
        "相对原对接姿势的变化涉及整个分子。</strong>"
        "其三个重复在末段的代表姿势差异较小。</p>"
        "<p><strong>CHEMBL1086046同时存在位置和内部形状变化，"
        "20261011重复的后段偏移更明显。</strong>"
        "三条重复的后段变化不同，需分别分析；解离及活性状态待进一步验证。</p>"
        "<p>两候选在生产起点前已经发生约3Å的整体RMSD调整。"
        "相对准备后起点与原始对接姿势的变化应分别报告。"
        "其余8项的后续研究按各自姿势与身份问题确定。</p>"
        "</section>"
    )
    window = (
        "<h2>后段是否与前段相同</h2><p>以下均以各重复生产起点为参照。"
        "首、末100帧各约1ns；每行是一条5ns重复。</p>"
        + table.to_html(index=False, border=0, float_format=lambda v: f"{v:.2f}")
        + '<p><a href="../reading_notes_v1/late_window_comparison.csv">完整窗口统计</a> · '
        '<a href="../representatives_v2/manifest.json">18份当前代表姿势SDF及来源</a></p>'
        "<p>重原子展示SDF已移除不适用的含氢电荷数组属性；原始参数及轨迹保持。"
        "这些展示文件不作为完整MD重启体系。</p>"
    )
    pages = [
        (args.analysis / "figures/index.html", "md_index_before_readout.html", prose + window),
        (args.evidence / "index.html", "evidence_index_before_readout.html", prose),
    ]
    for path, backup, snippet in pages:
        original = path.read_text(encoding="utf-8")
        assert 'id="final-readout"' not in original
        assert "<h2>" in original
        (notes / backup).write_text(original, encoding="utf-8")
        updated = original.replace("<h2>", snippet + "<h2>", 1)
        path.write_text(updated, encoding="utf-8")
    shutil.copyfile(__file__, notes / "source_snapshot.py")
    print(json.dumps({"pages_updated": 2, "window_rows": len(table), "source_data_changed": False}))


if __name__ == "__main__":
    main()
