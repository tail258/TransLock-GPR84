"""以已知GLPG1205控制先验收路径，再建立重构目录候选的复合物假设。"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import rdMolAlign

from translock_screening.candidate_poses import (
    conditional_stereoisomers,
    first_acceptable_pose,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("root", "legacy-root", "config", "gnina", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.legacy_root / "src"))
    # 复用结构准备、姿势检查与PDB导出的技术组件。
    from translock_gpr84.docking.mixture_diagnostic import pose_audit
    from translock_gpr84.docking.mixture_inputs import validate_coordinates
    from translock_gpr84.docking.pilot import fixed_command, prepare_smiles, run_command, runtime
    from translock_gpr84.docking.pose_evaluation import (
        contact_recovery,
        read_receptor_heavy_atoms,
        residue_contacts,
    )
    from translock_gpr84.reporting.candidate_poses import complex_text, minimum_distance

    def save(path, value):
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = {
        name: args.root / cfg[name]
        for name in ("candidates", "receptor", "geometry", "control_crystal")
    }
    source_stats = [
        {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in paths.values()
    ]
    candidates = pd.read_csv(paths["candidates"])
    if len(candidates) != 10 or candidates.candidate_id.duplicated().any():
        raise ValueError("候选名单不是固定10项")
    crystal = Chem.SDMolSupplier(str(paths["control_crystal"]), removeHs=False)[0]
    if crystal is None:
        raise ValueError("GLPG实验坐标无法读取")
    control_smiles = Chem.MolToSmiles(Chem.RemoveHs(crystal))
    samples = [
        {
            "candidate_id": "GLPG1205",
            "smiles": control_smiles,
            "role": "known_antagonist_technical_control",
        }
    ]
    samples += [
        {
            "candidate_id": r.candidate_id,
            "smiles": r.smiles,
            "role": "directory_candidate_hypothesis",
            "nomination_order": r.nomination_order,
        }
        for r in candidates.itertuples()
    ]
    states = []
    for sample in samples:
        for index, smiles in enumerate(conditional_stereoisomers(sample["smiles"]), 1):
            states.append(
                {
                    **sample,
                    "state_id": f"{sample['candidate_id']}_s{index:02d}",
                    "state_smiles": smiles,
                    "state_scope": "conditional_neutral_registered_tautomer_not_population",
                }
            )
    output = args.output.resolve()
    if not output.is_relative_to((args.root / "artifacts").resolve()):
        raise ValueError("输出必须在产物目录")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    repository = Path(__file__).resolve().parents[1]
    for path in [
        args.config,
        Path(__file__),
        repository / "docs/candidate-structure-protocol.md",
        repository / "src/translock_screening/candidate_poses.py",
        repository / "tests/test_candidate_poses.py",
        repository / "pyproject.toml",
        repository / "uv.lock",
    ]:
        shutil.copyfile(path, snapshot / path.name)
    for relative in (
        "src/translock_gpr84/docking/pilot.py",
        "src/translock_gpr84/docking/mixture_inputs.py",
        "src/translock_gpr84/docking/mixture_diagnostic.py",
        "src/translock_gpr84/docking/pose_evaluation.py",
        "src/translock_gpr84/reporting/candidate_poses.py",
        "src/translock_gpr84/reporting/diagnostic_export.py",
        "tests/unit/test_candidate_poses.py",
        "tests/unit/test_pose_evaluation.py",
        "tests/unit/test_mixture_inputs.py",
        "tests/unit/test_mixture_diagnostic.py",
    ):
        path = args.legacy_root / relative
        dest = snapshot / "legacy" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    for name, path in paths.items():
        shutil.copyfile(path, output / (name + path.suffix))
    prepared, complexes = output / "prepared", output / "complexes"
    prepared.mkdir()
    complexes.mkdir()
    manifest = {
        "status": "preparing",
        "config": cfg,
        "runtime": runtime(),
        "inputs": source_stats,
        "sample_count": len(samples),
        "state_count": len(states),
        "planned_jobs": len(states) * len(cfg["docking_seeds"]),
        "failures": [],
        "reused_legacy_scope": "structure_tools_only_no_activity_or_direction_evaluation",
    }
    save(output / "manifest.json", manifest)
    for state in states:
        sdf = prepared / (state["state_id"] + ".sdf")
        try:
            state["preparation"] = prepare_smiles(state["state_smiles"], sdf, cfg["embedding_seed"])
            mol = Chem.SDMolSupplier(str(sdf), removeHs=False)[0]
            state["geometry"] = validate_coordinates(mol, state["state_smiles"])
            state["status"] = "prepared"
            state["prepared_sdf"] = str(sdf.relative_to(output))
        except (ValueError, RuntimeError) as error:
            state["status"] = "preparation_failed"
            state["error"] = str(error)
            manifest["failures"].append({"state": state["state_id"], "error": str(error)})
    save(output / "states.json", states)
    print(
        json.dumps(
            {"state_count": len(states), "prepared": sum(s["status"] == "prepared" for s in states)}
        ),
        flush=True,
    )
    engine = {
        **cfg["engine"],
        **{k: json.loads(paths["geometry"].read_text())[k] for k in ("center", "size")},
    }
    receptor = paths["receptor"].read_text()
    receptor_xyz, labels = read_receptor_heavy_atoms(paths["receptor"])
    crystal_contacts = residue_contacts(
        receptor_xyz, labels, Chem.RemoveHs(crystal).GetConformer().GetPositions()
    )
    jobs, exported, control_checks = [], [], []
    started = time.perf_counter()
    for state in states:
        if state["status"] != "prepared":
            if state["candidate_id"] == "GLPG1205":
                raise ValueError("GLPG控制准备失败")
            continue
        if state["candidate_id"] != "GLPG1205" and (
            len(control_checks) != len(cfg["docking_seeds"])
            or not all(c["passed"] for c in control_checks)
        ):
            manifest["status"] = "stopped_control_failure"
            break
        for seed in cfg["docking_seeds"]:
            if shutil.disk_usage(args.root).free < cfg["minimum_free_GiB"] * 1024**3:
                raise RuntimeError("E盘空闲低于停止线")
            temperatures = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                text=True,
            )
            temperature = max(float(v) for v in temperatures.split())
            if temperature >= cfg["max_gpu_temperature_C"]:
                raise RuntimeError("GPU温度触及停止线")
            identifier = f"{state['state_id']}_{seed}"
            directory = output / "docking" / identifier
            directory.mkdir(parents=True, exist_ok=False)
            poses = directory / "poses.sdf"
            command = fixed_command(
                args.gnina,
                paths["receptor"],
                output / state["prepared_sdf"],
                poses,
                {**engine, "seed": seed},
            )
            execution = run_command(command, directory, cfg["max_job_seconds"])
            job = {
                "job_id": identifier,
                "candidate_id": state["candidate_id"],
                "state_id": state["state_id"],
                "seed": seed,
                "execution": execution,
                "gpu_temperature_before_C": temperature,
                "selected_rank": None,
            }
            if execution["status"] == "completed":
                audits = pose_audit(
                    poses,
                    state["state_smiles"],
                    engine,
                    {"grid_spacing_A": 0.375, "grid_tolerance_A": 0.001},
                )
                molecules = list(Chem.SDMolSupplier(str(poses), removeHs=False))
                for audit, mol in zip(audits, molecules, strict=True):
                    if audit["valid"]:
                        audit["minimum_receptor_distance_A"] = minimum_distance(mol, receptor_xyz)
                        audit["contacts_4A"] = residue_contacts(
                            receptor_xyz, labels, Chem.RemoveHs(mol).GetConformer().GetPositions()
                        )
                job["audits"] = audits
                rank = first_acceptable_pose(audits, cfg["minimum_receptor_heavy_distance_A"])
                job["selected_rank"] = rank
                if state["candidate_id"] == "GLPG1205":
                    checks = []
                    for audit, mol in zip(audits[:5], molecules[:5], strict=True):
                        if not audit["valid"]:
                            continue
                        rmsd = rdMolAlign.CalcRMS(Chem.RemoveHs(mol), Chem.RemoveHs(crystal))
                        recall = contact_recovery(set(crystal_contacts), set(audit["contacts_4A"]))[
                            "recall"
                        ]
                        checks.append(
                            {
                                "rank": audit["rank"],
                                "rmsd_A": rmsd,
                                "contact_recall": recall,
                                "passed": rmsd <= 2 and recall >= 0.75,
                            }
                        )
                    control_checks.append(
                        {
                            "seed": seed,
                            "poses_top5": checks,
                            "passed": any(c["passed"] for c in checks),
                        }
                    )
                if rank is not None:
                    mol = molecules[rank - 1]
                    text, chain = complex_text(receptor, mol, identifier)
                    pdb = complexes / (identifier + ".pdb")
                    sdf = complexes / (identifier + ".sdf")
                    pdb.write_text(text, encoding="utf-8")
                    with Chem.SDWriter(str(sdf)) as writer:
                        writer.write(mol)
                    exported.append(
                        {
                            "job_id": identifier,
                            "candidate_id": state["candidate_id"],
                            "state_id": state["state_id"],
                            "seed": seed,
                            "smiles": state["state_smiles"],
                            "role": state["role"],
                            "pose_rank": rank,
                            "ligand_chain": chain,
                            "pdb": str(pdb.relative_to(output)),
                            "sdf": str(sdf.relative_to(output)),
                            "contacts_4A": audits[rank - 1]["contacts_4A"],
                        }
                    )
            elif state["candidate_id"] == "GLPG1205":
                control_checks.append(
                    {"seed": seed, "passed": False, "execution_status": execution["status"]}
                )
            if job["selected_rank"] is None:
                manifest["failures"].append(
                    {
                        "job_id": identifier,
                        "execution_status": execution["status"],
                        "error": "no_exportable_pose",
                    }
                )
            jobs.append(job)
            save(output / "jobs.json", jobs)
            save(output / "exported_complexes.json", exported)
            save(output / "control_checks.json", control_checks)
            print(
                json.dumps(
                    {
                        "job": identifier,
                        "status": execution["status"],
                        "export_rank": job["selected_rank"],
                    }
                ),
                flush=True,
            )
    variations = []
    for state in states:
        pair = [e for e in exported if e["state_id"] == state["state_id"]]
        if len(pair) == 2:
            a, b = [Chem.SDMolSupplier(str(output / e["sdf"]), removeHs=True)[0] for e in pair]
            variations.append(
                {
                    "state_id": state["state_id"],
                    "candidate_id": state["candidate_id"],
                    "seed_pose_rmsd_A": rdMolAlign.CalcRMS(a, b),
                    "contact_jaccard": contact_recovery(
                        set(pair[0]["contacts_4A"]), set(pair[1]["contacts_4A"])
                    )["jaccard"],
                }
            )
    save(output / "seed_variations.json", variations)
    if manifest["status"] != "stopped_control_failure":
        manifest["status"] = (
            "completed_with_recorded_failures" if manifest["failures"] else "complete"
        )
    manifest.update(
        run_seconds=time.perf_counter() - started,
        actual_jobs=len(jobs),
        exported_complexes=len(exported),
        control_passed=all(c["passed"] for c in control_checks),
        geometry_valid_poses=sum(a["valid"] for j in jobs for a in j.get("audits", [])),
        total_poses=sum(len(j.get("audits", [])) for j in jobs),
    )
    if source_stats != [
        {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in paths.values()
    ]:
        raise ValueError("源输入改变")
    save(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "status",
                    "actual_jobs",
                    "exported_complexes",
                    "geometry_valid_poses",
                    "total_poses",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
