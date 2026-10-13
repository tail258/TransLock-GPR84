"""按固定来源范围检查十项候选骨架，并保存可回读表。"""

import argparse
import json
import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import rdkit
import yaml

from translock_screening.scaffold_audit import candidate_pairs, compare, prepare_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    inputs = [args.config, args.root / cfg["candidates"]]
    candidates = pd.read_csv(inputs[-1])
    if len(candidates) != 10 or candidates.candidate_id.duplicated().any():
        raise ValueError("固定十项候选身份不完整")
    references, failures = [], []
    for name, spec in cfg["references"].items():
        path = args.root / spec["path"]
        inputs.append(path)
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "expected_rows" in spec and len(frame) != spec["expected_rows"]:
            raise ValueError(f"{name}参照数量改变")
        ref, bad = prepare_reference(frame, name, spec["id_column"], spec["smiles_column"])
        references.append(ref)
        failures.extend(bad)
    stats_before = [
        {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in inputs
    ]
    output = args.output.resolve()
    if not output.is_relative_to((args.root / "artifacts").resolve()):
        raise ValueError("输出必须在产物根目录下")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    repository = Path(__file__).resolve().parents[1]
    for path in [
        args.config,
        Path(__file__),
        repository / "src/translock_screening/scaffold_audit.py",
        repository / "src/translock_screening/screening_application.py",
        repository / "tests/test_scaffold_audit.py",
        repository / "docs/candidate-structure-protocol.md",
        repository / "pyproject.toml",
        repository / "uv.lock",
    ]:
        shutil.copyfile(path, snapshot / path.name)
    reference = pd.concat(references, ignore_index=True)
    comparisons = pd.concat([compare(candidates, ref) for ref in references], ignore_index=True)
    pairs = candidate_pairs(candidates)
    reference.to_csv(output / "reference_structures.csv", index=False)
    comparisons.to_csv(output / "candidate_comparisons.csv", index=False)
    pairs.to_csv(output / "candidate_pairs.csv", index=False)
    shutil.copyfile(inputs[1], output / "input_candidates.csv")
    summary = {
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": cfg,
        "runtime": {"python": platform.python_version(), "rdkit": rdkit.__version__},
        "inputs": stats_before,
        "candidate_count": len(candidates),
        "failures": failures,
        "scope_summary": [
            {
                "scope": name,
                "rows": len(ref),
                "graphs": ref.connection.nunique(),
                "scaffolds": ref.scaffold.nunique(),
                "frameworks": ref.generic_framework.nunique(),
                "candidates_with_connection_match": int((table.connection_match_rows > 0).sum()),
                "candidates_with_scaffold_match": int((table.scaffold_match_rows > 0).sum()),
                "candidates_with_framework_match": int(
                    (table.generic_framework_match_rows > 0).sum()
                ),
            }
            for name, ref, table in [
                (r.iloc[0].scope, r, compare(candidates, r)) for r in references
            ]
        ],
        "pair_summary": {col: int(pairs[col].sum()) for col in pairs if col.startswith("same_")},
        "limits": "已取得结构范围的比较；非全球查新、功能方向或实测活性验证",
    }
    if stats_before != [
        {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in inputs
    ]:
        raise ValueError("输入运行中改变")
    (output / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in ("scope_summary", "pair_summary", "failures")}))


if __name__ == "__main__":
    main()
