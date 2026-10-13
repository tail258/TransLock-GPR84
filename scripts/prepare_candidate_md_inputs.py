"""把已选对接姿势转入已验收v5构建体坐标系，冻结两候选输入。"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from rdkit import Chem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("construct", "poses", "config", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    original = json.loads((args.construct / "manifest.json").read_text())
    assert original["status"] == "construct_geometry_passed"
    orientation = json.loads((args.construct / "orientation.json").read_text())
    rotation = np.array(orientation["rotation"])
    origin = np.array(orientation["origin_A"])
    assert np.allclose(rotation @ rotation.T, np.eye(3)) and np.linalg.det(rotation) > 0.999
    jobs = json.loads((args.poses / "exported_complexes.json").read_text())
    receptor_xyz = np.array(
        [
            [float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))]
            for line in (args.construct / "receptor_minimized.pdb").read_text().splitlines()
            if line.startswith("ATOM  ") and line[76:78].strip() != "H"
        ]
    )
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output / "source_snapshot.py")
    records = []
    for candidate in ("CHEMBL120691", "CHEMBL1086046"):
        job = next(j for j in jobs if j["job_id"] == f"{candidate}_s01_20261002")
        source = args.poses / job["sdf"]
        mol = Chem.SDMolSupplier(str(source), removeHs=True)[0]
        expected = Chem.MolToInchiKey(Chem.MolFromSmiles(job["smiles"]))
        assert mol is not None and Chem.MolToInchiKey(mol) == expected
        xyz = (mol.GetConformer().GetPositions() - origin) @ rotation.T
        for i, point in enumerate(xyz):
            mol.GetConformer().SetAtomPosition(i, point)
        coordinate_copy = Chem.Mol(mol)
        Chem.RemoveStereochemistry(coordinate_copy)
        Chem.AssignStereochemistryFrom3D(coordinate_copy, replaceExistingTags=True)
        assert Chem.MolToInchiKey(coordinate_copy) == expected
        closest = float(np.linalg.norm(receptor_xyz[:, None] - xyz[None, :], axis=2).min())
        assert closest > 1.5
        target = args.output / candidate / "construct"
        target.mkdir(parents=True)
        for name in (
            "receptor_minimized.pdb",
            "residue_map.json",
            "orientation.json",
            "retained_oriented_heavy.pdb",
        ):
            shutil.copyfile(args.construct / name, target / name)
        with Chem.SDWriter(str(target / "ligand_oriented.sdf")) as writer:
            writer.write(mol)
        cfg = yaml.safe_load(args.config.read_text())
        cfg.update(
            ligand_id=candidate,
            ligand_smiles=job["smiles"],
            ligand=str(source),
            pose_reference="docked_pose",
        )
        (target.parent / "config.yaml").write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        record = dict(
            status="construct_geometry_passed",
            source_construct=str(args.construct),
            receptor_rebuilt=False,
            ligand_id=candidate,
            source_pose=job,
            pose_path=str(source),
            pose_bytes=source.stat().st_size,
            pose_mtime_ns=source.stat().st_mtime_ns,
            inchikey=expected,
            minimum_ligand_receptor_heavy_distance_A=closest,
            config=cfg,
            scope="沿用对照v5受体；仅转换目录候选既有姿势，非新对接或活性确证",
        )
        (target / "manifest.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        records.append(record)
    (args.output / "inputs.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in ("ligand_id", "inchikey", "minimum_ligand_receptor_heavy_distance_A")
                }
                for r in records
            ]
        )
    )


if __name__ == "__main__":
    main()
