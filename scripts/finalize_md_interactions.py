"""发布补充分析解读、预算和文件链接验收，不重复数值处理。"""

import argparse
import json
import shutil
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        self.paths.extend(v for k, v in attrs if k in ("src", "href"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    final = args.root / "finalization_v1"
    final.mkdir(exist_ok=False)
    page = args.root / "analysis_v1/figures/index.html"
    original = page.read_text(encoding="utf-8")
    (final / "index_before_readout.html").write_text(original, encoding="utf-8")
    prose = (
        '<section id="scientific-readout"><h2>主要发现</h2>'
        "<p><strong>CHEMBL120691：短程姿势较一致，主要看到持续的非极性接触，"
        "直接氢键较少。</strong>它的三条轨迹在1.5Å中心半径下归为同一大簇；"
        "该簇描述当前短程轨迹，长期稳定性与活性待进一步评价。</p>"
        "<p><strong>CHEMBL1086046：氢键较多，但一条重复后段出现局部转向和接触重排。</strong>"
        "20261011重复的SER77–O34氢键由首段72%降到末段39%，TRP360–O35仍保持；"
        "C4–C5–S17–O18角由约−40°移到约−4°，其余两条变化小。"
        "氢键数量用于描述当前几何接触。</p>"
        "<p>外围CF3转动会增加固定原子对应下的差异。结果后核心子集核对显示聚类对尺度敏感，"
        "该差异按局部构象变化解释，稳定结合态需更长采样与独立验证。"
        "原三组阈值与全重原子结果均保留。</p>"
        '<p><a href="../../core_cluster_audit_v1/cluster_summary.csv">核心聚类敏感性</a> · '
        '<a href="../../core_cluster_audit_v1/cluster_window_fractions.csv">核心聚类首末窗口</a> · '
        '<a href="../../finalization_v1/report.md">完整中文报告</a> · '
        '<a href="../../independent_verification.json">独立复核</a></p></section>'
    )
    assert 'id="scientific-readout"' not in original
    page.write_text(original.replace("<h2>", prose + "<h2>", 1), encoding="utf-8")
    shutil.copyfile(args.report, final / "report.md")
    manifests = [
        "pilot_50_v1/manifest.json",
        "pilot_50_v2/manifest.json",
        "pilot_verification.json",
        "analysis_v1/manifest.json",
        "independent_verification.json",
        "analysis_v1/figures/plot_manifest.json",
        "core_cluster_audit_v1/manifest.json",
        "torsion_review_v2/manifest.json",
    ]
    stages = []
    for name in manifests:
        obj = json.loads((args.root / name).read_text())
        stages.append({"record": name, "status": obj["status"], "seconds": obj["wall_seconds"]})
    total = sum(s["seconds"] for s in stages)
    assert total < 3600
    reader = References()
    reader.feed(page.read_text(encoding="utf-8"))
    for value in reader.paths:
        parsed = urlparse(value)
        if not parsed.scheme and parsed.path:
            assert (page.parent / unquote(parsed.path)).resolve().exists(), value
    # 原始分析快照不覆盖；增加此次核对和展示修正的完整实现副本。
    code = Path(__file__).parents[1]
    for name in (
        "verify_md_interactions.py",
        "audit_md_core_clustering.py",
        "curate_md_torsion_scope.py",
        "analyze_completed_md_poses.py",
    ):
        shutil.copyfile(code / "scripts" / name, final / name)
    shutil.copyfile(
        code / "src/translock_screening/md_interactions.py", final / "md_interactions_final.py"
    )
    shutil.copyfile(code / "docs/md-interactions-protocol.md", final / "protocol.md")
    shutil.copyfile(__file__, final / Path(__file__).name)
    result = {
        "status": "complete_verified",
        "finished_utc": datetime.now(UTC).isoformat(),
        "frames": 4500,
        "replicates": 9,
        "test_count": 14,
        "measured_stage_wall_seconds": total,
        "includes_failed_pilot": True,
        "runtime_startup_and_tests_in_main_prior_reserve_seconds": 35,
        "budget_limit_seconds": 3600,
        "stages": stages,
        "HTML_references_checked": len(reader.paths),
        "current_torsions": "torsion_review_v2/torsion_summary.csv",
        "reused_existing_RMSD": True,
        "new_MD": False,
        "Obsidian_writes": False,
        "hashes": False,
        "cloud_instance_untouched": True,
    }
    with (args.root / "completion_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
