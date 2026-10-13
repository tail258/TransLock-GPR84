"""在 WSL 中检查手工专利结构与公开自动抽取结构是否一致。"""

import argparse
import json
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bank = json.loads(args.bank.read_text())
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=3)
    prepared = []
    for row in bank:
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is not None:
            prepared.append((row, Chem.MolToSmiles(mol), generator.GetSparseCountFingerprint(mol)))
    results = []
    for row in json.loads(args.manual.read_text()):
        mol = Chem.MolFromSmiles(row["manual_smiles"])
        if mol is None:
            raise ValueError(row["sample_id"])
        smiles = Chem.MolToSmiles(mol)
        fp = generator.GetSparseCountFingerprint(mol)
        exact = [x for x, s, _ in prepared if s == smiles]
        nearest = sorted(
            [(DataStructs.TanimotoSimilarity(fp, f), x) for x, _, f in prepared],
            key=lambda x: x[0],
            reverse=True,
        )[:3]
        row.update(
            canonical_smiles=smiles,
            public_exact_matches=exact,
            nearest=[{"similarity": v, "record": r} for v, r in nearest],
        )
        results.append(row)
        print(
            row["sample_id"],
            "exact",
            len(exact),
            "nearest",
            round(nearest[0][0], 4),
            "stereo",
            Chem.FindMolChiralCenters(mol, includeUnassigned=True),
        )
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
