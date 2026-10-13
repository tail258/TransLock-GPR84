"""从已验收坐标导出新的展示SDF，剔除原含氢参数的陈旧属性数组。"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from rdkit import Chem

from translock_screening.md_interpretation import coordinate_only_molecule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = json.loads((args.analysis / "summaries.json").read_text(encoding="utf-8"))
    records = []
    for row in rows:
        cid, seed = row["ligand_id"], row["seed"]
        source = args.bundle / "inputs" / cid / "parameters/ligand.sdf"
        molecule = Chem.RemoveHs(Chem.SDMolSupplier(str(source), removeHs=False)[0])
        data = np.load(args.analysis / cid / str(seed) / "aligned_ligand_frames.npz")
        folder = args.output / cid / str(seed)
        folder.mkdir(parents=True)
        for label, index in [
            ("representative", row["representative_frame"] - 1),
            ("late_representative", row["late_representative_frame"] - 1),
        ]:
            xyz = data["xyz_A"][index]
            exported = coordinate_only_molecule(molecule, xyz, f"{cid}_{seed}_{label}")
            exported.SetProp("time_ps", str((index + 1) * 10))
            exported.SetProp("reference_frame", "native_receptor_CA_aligned_to_original")
            path = folder / f"{label}.sdf"
            with Chem.SDWriter(str(path)) as writer:
                writer.write(exported)
            readback = Chem.SDMolSupplier(str(path), removeHs=True)[0]
            assert Chem.MolToInchiKey(readback) == Chem.MolToInchiKey(molecule)
            assert not readback.HasProp("atom.dprop.PartialCharge")
            assert np.allclose(readback.GetConformer().GetPositions(), xyz, atol=6e-5)
            records.append(
                {
                    "ligand_id": cid,
                    "seed": seed,
                    "time_ps": (index + 1) * 10,
                    "file": str(path),
                    "source_parameters": str(source),
                }
            )
    result = {
        "status": "passed",
        "exports": records,
        "coordinates_changed": False,
        "old_exports_preserved": True,
        "reason": "Heavy-only visualization export must not retain full explicit-H charge list",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copyfile(__file__, args.output / "source_snapshot.py")
    shutil.copyfile(
        Path(__file__).parents[1] / "src/translock_screening/md_interpretation.py",
        args.output / "md_interpretation.py",
    )
    print(json.dumps({"status": "passed", "clean_exports": len(records)}))


if __name__ == "__main__":
    main()
