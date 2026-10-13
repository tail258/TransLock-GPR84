"""用已有数据环境导出完整身份列，供无Parquet依赖的模型环境读取。"""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = [args.audit.stat().st_size, args.audit.stat().st_mtime_ns]
    columns = ["molecule_chembl_id", "standardized_smiles"]
    frame = pd.read_parquet(args.audit)[columns]
    frame.to_csv(args.output / "legacy_identity.csv", index=False)
    restored = pd.read_csv(args.output / "legacy_identity.csv")
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), restored, check_dtype=False)
    assert before == [args.audit.stat().st_size, args.audit.stat().st_mtime_ns]
    shutil.copy2(Path(__file__), args.output / Path(__file__).name)
    manifest = {
        "status": "complete",
        "source": str(args.audit),
        "source_stats": before,
        "rows": len(frame),
        "columns": columns,
        "retained_all_rows": True,
        "projection_readback": "pass",
        "utc": datetime.now(UTC).isoformat(),
        "hashes_computed": False,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
