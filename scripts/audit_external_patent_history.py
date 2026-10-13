"""补充审计旧研究中未全部进入数据库训练表的已见结构。"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from translock_screening.external_validation import identity_keys  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--storage-root", type=Path, required=True)
parser.add_argument("--admission", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(exist_ok=False)
r = args.storage_root
rows = []
for folder, filename, col, id_col, name in [
    (
        "acs.jmedchem.5c03367/2026-09-06",
        "jm5c03367_si_004.csv",
        "Smiles",
        "Compound ",
        "chromenopyrrole_author82",
    ),
    (
        "binop_identity_2026-09-27_v1",
        "jm4c00359_si_001.csv",
        "SMILES",
        "compound",
        "BINOP_author51",
    ),
]:
    path = r / "downloads/literature" / folder / filename
    data = pd.read_csv(path)
    for row in data.to_dict("records"):
        rows.append(
            {
                "source": name,
                "sample_id": str(row[id_col]),
                "source_path": str(path),
                "source_smiles": row[col],
                **identity_keys(row[col]),
            }
        )
path = r / "artifacts/data/patent_66335_sample_review_2026-09-07/manual_structures.json"
for row in json.loads(path.read_text())["records"]:
    rows.append(
        {
            "source": "patent66335_reviewed8",
            "sample_id": str(row["example"]),
            "source_path": str(path),
            "source_smiles": row["smiles"],
            **identity_keys(row["smiles"]),
        }
    )
path = r / "artifacts/data/patent_55117_review_2026-09-07/manual_I277.json"
row = json.loads(path.read_text())
rows.append(
    {
        "source": "patent55117_I277",
        "sample_id": row["compound_id"],
        "source_path": str(path),
        "source_smiles": row["manual_smiles"],
        **identity_keys(row["manual_smiles"]),
    }
)
seen = pd.DataFrame(rows)
test = pd.read_csv(args.admission / "admission_all_61.csv")
hits = test[["sample_id", "broad_identity"]].merge(
    seen, on="broad_identity", suffixes=("_test", "_seen")
)
seen.to_csv(args.output / "historical_structures.csv", index=False)
hits.to_csv(args.output / "overlaps.csv", index=False)
summary = {
    "source_counts": seen.source.value_counts().to_dict(),
    "total": len(seen),
    "matches": len(hits),
    "all_test_samples": len(test),
    "scope": "历史机器可读结构及已手工转录样品；未转录的专利其余结构未声称逐图匹配",
}
(args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False))
