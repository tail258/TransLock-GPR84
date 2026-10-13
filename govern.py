"""从随包原活动和审计表回放准入与单位合并；需要environments/data环境。"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
import yaml
from translock_screening.measurements import admit_measurements, assay_graph_units


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--from-raw", action="store_true", help="同时回放原始活动的语境与结构审计")
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    audit = pd.read_parquet(ROOT / "data/source/activity_audit.parquet")
    if args.from_raw:
        sys.path.insert(0, str(ROOT / "legacy/src"))
        from translock_gpr84.data.activity_audit import audit_records, load_jsonl
        audit_config = yaml.safe_load((ROOT / "configs/original_activity_audit.yaml").read_text())
        source = ROOT / "data/source"
        audit = pd.DataFrame(audit_records(
            *[load_jsonl(source / f"{name}.jsonl") for name in
              ["activities", "assays", "molecules", "documents", "sources"]],
            audit_config["audit"],
        ))
    raw = {int(r["activity_id"]): r for r in
           (json.loads(s) for s in (ROOT / "data/source/activities.jsonl").read_text().splitlines())}
    manifest = json.loads((ROOT / "data/training/manifest.json").read_text(encoding="utf-8"))
    # 已扩展的隔离别名是原运行输入记录，不重新指定或放宽隔离范围。
    measurements, exclusions, summary = admit_measurements(audit, raw, manifest["expanded_policy"])
    units, conflicts = assay_graph_units(measurements)
    expected = pd.read_csv(ROOT / "data/training/training_units.csv")
    pd.testing.assert_frame_equal(units.reset_index(drop=True), expected, check_dtype=False)
    if (len(measurements), len(units), units.smiles.nunique()) != (285, 277, 248):
        raise ValueError("回放数量与正式准入不同")
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in [("admitted_measurements", measurements), ("training_units", units),
                        ("exclusions", exclusions), ("conflicts", conflicts)]:
        frame.to_csv(output / f"{name}.csv", index=False)
    if args.from_raw:
        audit.to_csv(output / "activity_audit_replay.csv", index=False)
    summary.update({"units": len(units), "graphs": int(units.smiles.nunique()),
                    "archived_units_match": True, "hashes_computed": False})
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    print("准入回放通过：2070条原记录→285条测量→277个单位→248个图。")


if __name__ == "__main__":
    main()
