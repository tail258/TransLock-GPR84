"""独立读取体系XML/阶段坐标和诊断，核对膜小样并绘图。"""

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

from translock_screening.dcd_reader import read_openmm_dcd, time_for_step


def state_positions(path):
    state = ET.parse(path).getroot()
    positions = np.array(
        [[float(p.get(k)) for k in ("x", "y", "z")] for p in state.find("Positions")]
    )
    vectors = np.array(
        [
            [float(state.find("PeriodicBoxVectors/" + name).get(k)) for k in ("x", "y", "z")]
            for name in ("A", "B", "C")
        ]
    )
    assert np.isfinite(positions).all() and np.isfinite(vectors).all()
    assert np.linalg.det(vectors) > 0
    return positions, vectors, dict(state.find("Parameters").attrib), float(state.get("time"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    assert manifest["status"] in {"complete_1ns_technical_sample", "failed_complete_system"}
    construct = Path(manifest["construct"])
    parameters = Path(manifest["parameters"])
    pm = json.loads((parameters / "manifest.json").read_text())
    ligand_count = pm["atoms"]
    ligand_id = manifest["config"].get("ligand_id", "GLPG1205")
    assert pm["status"] == "ligand_parameters_passed"
    ligand = Chem.SDMolSupplier(str(parameters / "ligand.sdf"), removeHs=False)[0]
    assert ligand is not None and Chem.MolToInchiKey(Chem.RemoveHs(ligand)) == pm["inchikey"]
    ligand_parameters = json.loads((parameters / "atom_parameters.json").read_text())
    input_ligand = Chem.SDMolSupplier(str(construct / "ligand_oriented.sdf"), removeHs=True)[0]
    parameter_heavy = [a for a in ligand_parameters if a["atomic_number"] != 1]
    assert input_ligand is not None and Chem.MolToInchiKey(input_ligand) == pm["inchikey"]
    assert [a.GetAtomicNum() for a in input_ligand.GetAtoms()] == [
        a["atomic_number"] for a in parameter_heavy
    ]
    assert np.allclose(
        input_ligand.GetConformer().GetPositions(),
        [a["position_A"] for a in parameter_heavy],
        atol=0.0001,
    )
    ligand_system = ET.parse(parameters / "ligand_system.xml").getroot()
    ligand_nb = next(f for f in ligand_system.find("Forces") if f.get("type") == "NonbondedForce")
    ligand_q = np.array([float(p.get("q")) for p in ligand_nb.find("Particles")])
    assert len(ligand_q) == ligand_count and np.allclose(
        ligand_q, [a["charge_e"] for a in ligand_parameters], atol=1e-7
    )
    assert abs(ligand_q.sum()) < 1e-5
    atoms = json.loads((args.run / "atom_map.json").read_text())
    assert [a["index"] for a in atoms] == list(range(len(atoms)))
    system = ET.parse(args.run / "system.xml").getroot()
    assert len(system.find("Particles")) == len(atoms) == manifest["atom_count"]
    assert not any(f.get("type") == "CustomCompoundBondForce" for f in system.find("Forces"))
    nb = next(f for f in system.find("Forces") if f.get("type") == "NonbondedForce")
    charges = np.array([float(p.get("q")) for p in nb.find("Particles")])
    assert abs(charges.sum()) < 1e-3
    protein_atoms = [a for a in atoms if a["chain"] == "R"]
    protein_n = len(protein_atoms)
    assert np.allclose(charges[protein_n : protein_n + ligand_count], ligand_q, atol=1e-7)
    by_name = {(int(a["residue_id"]), a["name"]): a["index"] for a in protein_atoms}
    mapping = json.loads((construct / "residue_map.json").read_text())
    assert len(mapping) == 291
    cn = np.array([[by_name[(i, "C")], by_name[(i + 1, "N")]] for i in range(1, 291)])
    native_to_model = {r["native_id"]: r["model_id"] for r in mapping if r["native_id"] is not None}
    ss = np.array(
        [
            [by_name[(native_to_model[a], "SG")], by_name[(native_to_model[b], "SG")]]
            for a, b in [(11, 166), (93, 168)]
        ]
    )
    q = np.array(
        [
            [by_name[(i, n)] for n in ["N", "CA", "C", "CB"]]
            for i in range(1, 292)
            if (i, "CB") in by_name
        ]
    )
    qh = np.array([by_name[(i, "HA")] for i in range(1, 292) if (i, "CB") in by_name])
    alpha_ids = [i for i in range(1, 292) if (i, "CB") in by_name]
    map_by_id = {r["model_id"]: r for r in mapping}
    original_ca = {}
    for line in (construct / "retained_oriented_heavy.pdb").read_text().splitlines():
        if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
            original_ca[int(line[22:26])] = (
                np.array([float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))]) / 10
            )
    ca_ids = sorted(original_ca)
    ca_reference = np.array([original_ca[i] for i in ca_ids])
    ca_indices = np.array([by_name[(i, "CA")] for i in ca_ids])
    lig_heavy_local = [i for i, a in enumerate(ligand_parameters) if a["atomic_number"] != 1]
    lig_indices = np.array(lig_heavy_local) + protein_n
    crystal = np.array([ligand_parameters[i]["position_A"] for i in lig_heavy_local]) / 10

    def coordinate_check(xyz, vectors, label, time_ps):
        assert len(xyz) == len(atoms)
        inverse = np.linalg.inv(vectors)

        def periodic(delta, inverse=inverse, vectors=vectors):
            frac = delta @ inverse
            return (frac - np.rint(frac)) @ vectors

        cn_lengths = np.linalg.norm(periodic(xyz[cn[:, 0]] - xyz[cn[:, 1]]), axis=1) * 10
        ss_lengths = np.linalg.norm(periodic(xyz[ss[:, 0]] - xyz[ss[:, 1]]), axis=1) * 10
        v = [periodic(xyz[q[:, i]] - xyz[q[:, 1]]) * 10 for i in (0, 2, 3)]
        volumes = np.linalg.det(np.stack(v, axis=1))
        vh = periodic(xyz[qh] - xyz[q[:, 1]]) * 10
        hydrogen_volumes = np.linalg.det(np.stack([v[0], v[1], vh], axis=1))
        angles = np.rad2deg(
            np.arccos(
                np.clip(
                    np.einsum("ij,ij->i", v[2], vh)
                    / (np.linalg.norm(v[2], axis=1) * np.linalg.norm(vh, axis=1)),
                    -1,
                    1,
                )
            )
        )
        bad = np.where((volumes <= 1) | (hydrogen_volumes >= 0) | (angles <= 80) | (angles >= 140))[
            0
        ]
        ca_xyz = xyz[ca_indices]
        center, reference_center = ca_xyz.mean(axis=0), ca_reference.mean(axis=0)
        u, _, vt = np.linalg.svd((ca_xyz - center).T @ (ca_reference - reference_center))
        sign = np.eye(3)
        sign[-1, -1] = np.linalg.det(u @ vt)
        rotation = u @ sign @ vt
        ligand_fit = (xyz[lig_indices] - center) @ rotation + reference_center
        ligand_rmsd = np.sqrt(np.mean(np.sum((ligand_fit - crystal) ** 2, axis=1))) * 10
        coordinate_ligand = Chem.Mol(ligand)
        conformer = coordinate_ligand.GetConformer()
        for i in range(ligand_count):
            conformer.SetAtomPosition(i, xyz[protein_n + i] * 10)
        Chem.RemoveStereochemistry(coordinate_ligand)
        Chem.AssignStereochemistryFrom3D(coordinate_ligand, replaceExistingTags=True)
        coordinate_identity = Chem.MolToInchiKey(Chem.RemoveHs(coordinate_ligand))
        ligand_identity_passed = coordinate_identity == pm["inchikey"]
        geometry_passed = bool(
            cn_lengths.min() > 1.1
            and cn_lengths.max() < 1.6
            and ss_lengths.min() > 1.8
            and ss_lengths.max() < 2.3
            and not len(bad)
            and ligand_identity_passed
        )
        if manifest["status"] == "complete_1ns_technical_sample":
            assert geometry_passed, f"独立几何检查失败：{label}"
        return {
            "stage": label,
            "time_ps": float(time_ps),
            "geometry_passed": geometry_passed,
            "CN_range_A": [float(cn_lengths.min()), float(cn_lengths.max())],
            "SS_range_A": [float(ss_lengths.min()), float(ss_lengths.max())],
            "alpha_volume_min_A3": float(volumes.min()),
            "alpha_hydrogen_volume_max_A3": float(hydrogen_volumes.max()),
            "alpha_CB_CA_HA_angle_range_deg": [float(angles.min()), float(angles.max())],
            "failed_alpha_centers": [
                {
                    "model_id": alpha_ids[i],
                    "native_id": map_by_id[alpha_ids[i]]["native_id"],
                    "CB_volume_A3": float(volumes[i]),
                    "HA_volume_A3": float(hydrogen_volumes[i]),
                    "CB_CA_HA_angle_deg": float(angles[i]),
                }
                for i in bad
            ],
            "ligand_mapped_heavy_RMSD_to_input_pose_after_native_CA_fit_A": float(ligand_rmsd),
            "ligand_coordinate_identity_matches": ligand_identity_passed,
            "ligand_coordinate_inchikey": coordinate_identity,
            "box_nm": vectors.tolist(),
        }

    records = []
    elapsed = 0
    for stage in manifest["stages"]:
        elapsed += stage["time_ps"]
        xyz, vectors, parameters_actual, time_actual = state_positions(
            args.run / f"{stage['name']}_state.xml"
        )
        assert abs(time_actual - elapsed) < 1e-5
        assert float(parameters_actual["k"]) == stage["restraint_k"]
        record = coordinate_check(xyz, vectors, stage["name"], elapsed)
        record["context_restraint_k"] = float(parameters_actual["k"])
        records.append(record)
    frames = []
    final_frame = None
    for frame in read_openmm_dcd(args.run / "trajectory.dcd"):
        final_frame = frame
        frames.append(
            coordinate_check(
                frame.positions_nm,
                frame.box_vectors_nm,
                "dcd",
                time_for_step(frame.step, manifest["config"]["stages"]),
            )
        )
    assert frames
    if manifest["status"] == "complete_1ns_technical_sample":
        assert abs(frames[-1]["time_ps"] - 1000) < 1e-6
    diagnostics = pd.read_csv(args.run / "diagnostics.csv")
    assert np.isfinite(diagnostics.select_dtypes(include="number").to_numpy()).all()
    assert diagnostics.time_ps.is_monotonic_increasing
    if manifest["status"] == "complete_1ns_technical_sample":
        assert len(records) == len(manifest["config"]["stages"]) == 8
        assert abs(diagnostics.time_ps.iloc[-1] - 1000) < 1e-5
        assert abs(sum(s["time_ps"] for s in manifest["stages"]) - 1000) < 1e-8
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), args.output / "source_snapshot.py")
    shutil.copyfile(
        Path(__file__).parents[1] / "src/translock_screening/dcd_reader.py",
        args.output / "dcd_reader.py",
    )
    (args.output / "frame_coordinate_checks.json").write_text(
        json.dumps(frames, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        [{k: v for k, v in frame.items() if not isinstance(v, list)} for frame in frames]
    ).to_csv(args.output / "frame_coordinate_checks.csv", index=False)
    result = {
        "status": "passed"
        if manifest["status"] == "complete_1ns_technical_sample"
        else "partial_failure_reviewed",
        "source_status": manifest["status"],
        "particles": len(atoms),
        "ligand_charge_sum_e": float(ligand_q.sum()),
        "system_charge_sum_e": float(charges.sum()),
        "stage_coordinate_checks": records,
        "trajectory_frames": len(frames),
        "trajectory_last_time_ps": frames[-1]["time_ps"],
        "failed_trajectory_geometry_frames": sum(not f["geometry_passed"] for f in frames),
        "input_pose_reference_ligand_final_RMSD_A": frames[-1][
            "ligand_mapped_heavy_RMSD_to_input_pose_after_native_CA_fit_A"
        ],
        "diagnostic_records": len(diagnostics),
        "final_time_ps": float(diagnostics.time_ps.iloc[-1]),
        "checks": [
            "SDF身份与电荷映射",
            "参数坐标与当前构建体输入配体共同坐标系",
            "XML粒子/电中性",
            "几何修复力未进入MD",
            "阶段XML重新计算PBC键长与CA立体",
            "State中的实际时间/位置约束参数，末段k=0",
            "DCD全部保存帧回读，CA重原子/氢方向与输入姿势参照配体RMSD",
            "DCD时间按分段实际步长计算，未用单一头部dt",
            "CSV有限性/时间/阶段成员",
        ],
        "scope": "完整膜技术与初步平衡小样，非充分平衡或活性确证",
    }
    (args.output / "independent_readback.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    frame_table = pd.DataFrame(frames)
    fig, axs = plt.subplots(3, 3, figsize=(16, 11))
    series = [
        ("temperature_K", "Temperature (K)"),
        ("potential_energy_kJ_mol", "Potential energy (kJ/mol)"),
        ("area_nm2", "XY area (nm2)"),
        ("native_CA_aligned_RMSD_A", "Native CA RMSD (Angstrom)"),
        (
            "ligand_mapped_heavy_RMSD_to_input_pose_after_native_CA_fit_A",
            f"{ligand_id} RMSD to input pose (Angstrom)",
        ),
        ("contact_recall", "Contact recall"),
        ("alpha_volume_min_A3", "Minimum CA-CB signed volume (Angstrom3)"),
        ("alpha_hydrogen_volume_max_A3", "Maximum CA-HA signed volume (Angstrom3)"),
        ("phosphate_layer_separation_nm", "Phosphate layer separation (nm)"),
    ]
    for ax, (col, label) in zip(axs.ravel(), series, strict=True):
        data = frame_table if col in frame_table else diagnostics
        ax.plot(data.time_ps / 1000, data[col], color="#376c8c", lw=1.4)
        if col == "alpha_volume_min_A3":
            ax.axhline(1, color="#bd593c", ls="--", lw=1)
        elif col == "alpha_hydrogen_volume_max_A3":
            ax.axhline(0, color="#bd593c", ls="--", lw=1)
        ax.set_xlabel("Simulation time (ns)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"{ligand_id} complete membrane technical sample | not an activity validation", fontsize=15
    )
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(args.output / f"membrane_diagnostics.{ext}", dpi=180)
    plt.close(fig)
    xyz_A = final_frame.positions_nm * 10
    ca_path = np.array([by_name[(r["model_id"], "CA")] for r in mapping])
    synthetic = np.array(
        [by_name[(r["model_id"], "CA")] for r in mapping if r["native_id"] is None]
    )
    phosphorus = np.array(
        [a["index"] for a in atoms if a["residue"] in {"POP", "POPC"} and a["element"] == "P"]
    )
    carbon = np.array(
        [a["index"] for a in atoms if a["residue"] in {"POP", "POPC"} and a["element"] == "C"]
    )
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    for ax, (x, y) in zip(axs, [(0, 2), (1, 2), (0, 1)], strict=True):
        ax.scatter(xyz_A[carbon, x], xyz_A[carbon, y], s=1, alpha=0.09, color="#92ad91")
        ax.scatter(
            xyz_A[phosphorus, x],
            xyz_A[phosphorus, y],
            s=6,
            color="#65826a",
            alpha=0.7,
            label="POPC phosphorus",
        )
        ax.plot(
            xyz_A[ca_path, x],
            xyz_A[ca_path, y],
            color="#687583",
            lw=1.3,
            label="Receptor CA",
        )
        ax.plot(
            xyz_A[synthetic, x],
            xyz_A[synthetic, y],
            "o-",
            color="#bd593c",
            ms=3,
            label="Synthetic 13-ALA",
        )
        ax.scatter(
            xyz_A[lig_indices, x],
            xyz_A[lig_indices, y],
            s=24,
            color="#315b91",
            label=f"{ligand_id} heavy atoms",
        )
        ax.set_xlabel("XYZ"[x] + " (Angstrom)")
        ax.set_ylabel("XYZ"[y] + " (Angstrom)")
        ax.set_aspect("equal")
    axs[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Complete membrane saved frame at {frames[-1]['time_ps'] / 1000:g} ns")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(args.output / f"membrane_geometry.{ext}", dpi=180)
    plt.close(fig)
    html = (
        """<!doctype html><html lang="zh"><meta charset="utf-8"><title>GLPG1205膜小样</title>
        <style>body{max-width:1100px;margin:40px auto;padding:0 24px;
        font:17px/1.6 system-ui;color:#23334a}img{width:100%}
        table{font-size:13px;border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}
        </style><h1>GLPG1205完整膜技术小样</h1>
        <p>13ALA连接肽、几何膜取向、AMBER/Lipid17及Sage/NAGL。
        1ns小样用于体系运行和初步平衡检查。</p>"""
        + f"<p>运行状态：{manifest['status']}；独立回读：{result['status']}；"
        + f"{len(atoms):,}原子。</p>"
        + '<img src="membrane_diagnostics.png">'
        + '<img src="membrane_geometry.png">'
        + pd.DataFrame(manifest["stages"]).to_html(index=False)
        + '<p><a href="../manifest.json">运行清单</a> · '
        + '<a href="independent_readback.json">独立回读</a> · '
        + '<a href="../system_initial.pdb">初始完整体系</a></p></html>'
    )
    (args.output / "index.html").write_text(html.replace("GLPG1205", ligand_id), encoding="utf-8")
    print(
        json.dumps(
            {k: result[k] for k in ["status", "particles", "final_time_ps", "diagnostic_records"]}
        )
    )


if __name__ == "__main__":
    main()
