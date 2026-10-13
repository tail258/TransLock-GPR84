"""独立读取构建体PDB，核对连接与取向并生成检查图。"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from scipy.spatial.distance import cdist


def pdb_atoms(path):
    return {
        (int(line[22:26]), line[12:16].strip()): np.array(
            [float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))]
        )
        for line in path.read_text().splitlines()
        if line.startswith("ATOM  ")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    assert manifest["status"] == "construct_geometry_passed"
    mapping = json.loads((args.run / "residue_map.json").read_text())
    xyz = pdb_atoms(args.run / "receptor_minimized.pdb")
    original = pdb_atoms(args.run / "retained_oriented_heavy.pdb")
    assert len([k for k in xyz if k[1] == "CA"]) == 291
    lengths = [float(np.linalg.norm(xyz[(i, "C")] - xyz[(i + 1, "N")])) for i in range(1, 291)]
    assert min(lengths) > 1.1 and max(lengths) < 1.6
    volumes, hydrogen_volumes = [], []
    for i in range(1, 292):
        if (i, "CB") in xyz:
            n, ca, c, cb = [xyz[(i, n)] for n in ("N", "CA", "C", "CB")]
            volumes.append(float(np.linalg.det(np.stack([n - ca, c - ca, cb - ca]))))
            ha = xyz[(i, "HA")]
            hydrogen_volumes.append(float(np.linalg.det(np.stack([n - ca, c - ca, ha - ca]))))
            angle = np.rad2deg(
                np.arccos(
                    np.clip(
                        np.dot(cb - ca, ha - ca)
                        / (np.linalg.norm(cb - ca) * np.linalg.norm(ha - ca)),
                        -1,
                        1,
                    )
                )
            )
            assert 80 < angle < 140
    assert min(volumes) > 1
    assert max(hydrogen_volumes) < 0
    retained = [r["model_id"] for r in mapping if r["native_id"] is not None]
    native_ca_rmsd = float(
        np.sqrt(np.mean([np.sum((xyz[(i, "CA")] - original[(i, "CA")]) ** 2) for i in retained]))
    )
    original_ss = manifest["disulfides"]
    native_to_model = {r["native_id"]: r["model_id"] for r in mapping if r["native_id"] is not None}
    for pair in original_ss:
        a, b = [native_to_model[n] for n in pair["native_pair"]]
        d = float(np.linalg.norm(xyz[(a, "SG")] - xyz[(b, "SG")]))
        assert abs(d - pair["distance_A"]) < 0.002
    ligand = Chem.SDMolSupplier(str(args.run / "ligand_oriented.sdf"), removeHs=True)[0]
    ligxyz = ligand.GetConformer().GetPositions()
    heavyxyz = np.array([p for (_, name), p in xyz.items() if not name.startswith("H")])
    minimum = float(cdist(heavyxyz, ligxyz).min())
    assert minimum > 1.5
    synthetic = [r["model_id"] for r in mapping if r["native_id"] is None]
    ca = np.array([xyz[(i, "CA")] for i in range(1, 292)])
    synth = ca[np.array(synthetic) - 1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, (a, b) in zip(axes, [(0, 2), (1, 2), (0, 1)], strict=True):
        ax.plot(ca[:, a], ca[:, b], color="#8c99a5", lw=1, label="Retained receptor CA")
        ax.plot(
            synth[:, a], synth[:, b], "o-", color="#c65042", ms=4, lw=2, label="Synthetic 13-ALA"
        )
        for bond in ligand.GetBonds():
            pts = ligxyz[[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
            ax.plot(pts[:, a], pts[:, b], color="#285f88", lw=2)
        ax.scatter(ligxyz[:, a], ligxyz[:, b], s=10, color="#285f88", label="Experimental GLPG1205")
        if b == 2:
            ax.axhline(-15, color="#8e9d78", ls="--")
            ax.axhline(15, color="#8e9d78", ls="--")
        ax.set_xlabel("XYZ"[a] + " (Angstrom)")
        ax.set_ylabel("XYZ"[b] + " (Angstrom)")
        ax.set_aspect("equal")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "9WIJ-derived control construct | geometric membrane orientation, synthetic ICL3",
        fontsize=15,
    )
    fig.tight_layout()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), args.output / "source_snapshot.py")
    for ext in ("png", "svg"):
        fig.savefig(args.output / f"construct_geometry.{ext}", dpi=180)
    result = {
        "status": "passed",
        "CA_count": len(ca),
        "synthetic_count": len(synthetic),
        "peptide_CN_range_A": [min(lengths), max(lengths)],
        "alpha_volume_min_A3": min(volumes),
        "alpha_hydrogen_volume_max_A3": max(hydrogen_volumes),
        "retained_CA_RMSD_to_input_A": native_ca_rmsd,
        "ligand_receptor_minimum_heavy_distance_A": minimum,
        "synthetic_CA_z_range_A": [float(synth[:, 2].min()), float(synth[:, 2].max())],
        "scope": "构建体几何与取向检查；非膜插入或MD验收",
    }
    (args.output / "independent_readback.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
